
#include "distributed.h"

#include "rpc.h"

#include "rdma.h"

#define ZSTD_STATIC_LINKING_ONLY
#include "zstd/lib/zstd.h"

#include <fmt/printf.h>
#include <torch/torch.h>

#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace distributed {

struct NetStats {
  bool hasData = false;
  double sent = 0.0;
  double received = 0.0;
  double rpcCalls = 0.0;
  double latency = 0.0;
  std::chrono::steady_clock::time_point lastprint{};
  std::mutex m;
};
inline NetStats netstats;

struct NetStatsCounter {
  std::chrono::steady_clock::time_point timestamp{};
  size_t sent = 0;
  size_t received = 0;
  size_t rpcCalls = 0;
};

template <typename T>
void addnetworkstats(const T& obj, NetStatsCounter& counter) {
  auto now = std::chrono::steady_clock::now();
  auto elapsed = now - counter.timestamp;
  if (elapsed <
      (netstats.hasData ? std::chrono::seconds(1) : std::chrono::seconds(10))) {
    return;
  }
  std::unique_lock l(netstats.m, std::try_to_lock);
  if (!l.owns_lock()) {
    return;
  }
  counter.timestamp = now;
  double t = std::chrono::duration_cast<
                 std::chrono::duration<double, std::ratio<1, 1>>>(elapsed)
                 .count();
  size_t newSent = obj.bytesSent();
  size_t newReceived = obj.bytesReceived();
  size_t newCalls = obj.numRpcCalls();
  double sent = (newSent - std::exchange(counter.sent, newSent)) / t;
  double recv =
      (newReceived - std::exchange(counter.received, newReceived)) / t;
  double calls = (newCalls - std::exchange(counter.rpcCalls, newCalls)) / t;

  double alpha = std::pow(0.99, t);
  if (!netstats.hasData) {
    alpha = 0.0;
    netstats.hasData = true;
  }
  netstats.sent = netstats.sent * alpha + sent * (1.0 - alpha);
  netstats.received = netstats.received * alpha + recv * (1.0 - alpha);
  netstats.rpcCalls = netstats.rpcCalls * alpha + calls * (1.0 - alpha);

  constexpr bool haslatency = std::is_same_v<T, rpc::Client>;
  if constexpr (haslatency) {
    double ll = std::chrono::duration_cast<
                    std::chrono::duration<double, std::ratio<1, 1000>>>(
                    obj.lastLatency())
                    .count();
    netstats.latency = netstats.latency * alpha + ll * (1.0 - alpha);
  }

  if (now - netstats.lastprint >= std::chrono::seconds(60)) {
    netstats.lastprint = now;
    if (haslatency) {
      printf("Network stats: in: %.02fM/s out: %.02fM/s  RPC calls: %.02f/s "
             "latency: %.02fms\n",
             netstats.received / 1024 / 1024, netstats.sent / 1024 / 1024,
             netstats.rpcCalls, netstats.latency);
    } else {
      printf("Network stats: in: %.02fM/s out: %.02fM/s  RPC calls: %.02f/s\n",
             netstats.received / 1024 / 1024, netstats.sent / 1024 / 1024,
             netstats.rpcCalls);
    }
  }
}

inline rpc::Rpc& getRpc() {
  static std::unique_ptr<rpc::Rpc> rpc = []() {
    auto rpc = std::make_unique<rpc::Rpc>();
    rpc->asyncRun(40);
    return rpc;
  }();
  return *rpc;
}

struct RDMAModelInfo {
  uint32_t checksum;
  uint32_t key;
  uintptr_t address;
  size_t size;
};

struct Crc32 {
  std::array<uint32_t, 256> lut;
  Crc32() {
    for (uint32_t i = 0; i != 256; ++i) {
      uint32_t v = i;
      for (size_t b = 0; b != 8; ++b) {
        v = (v >> 1) ^ (v & 1 ? 0xedb88320 : 0);
      }
      lut[i] = v;
    }
  }
  uint32_t operator()(const void* ptr, size_t size) {
    uint32_t r = 0xffffffff;
    unsigned char* c = (unsigned char*)ptr;
    unsigned char* end = c + size;
    while (c != end) {
      r = (r >> 8) ^ lut[(r ^ *c++) & 0xff];
    }
    return r;
  }
} crc32;

class ServerImpl {

  std::shared_ptr<rpc::Server> server;

  std::minstd_rand rng{std::random_device()()};

  float rollChance(std::string_view id) {
    auto i = models.find(id);
    if (i == models.end()) {
      return 0.0f;
    }
    float rating = i->second.rating;
    float max = 0.0f;
    std::vector<std::pair<float, std::string_view>> sorted;
    for (auto& [id, m] : models) {
      sorted.emplace_back(m.rating, id);
      max = std::max(max, m.rating);
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    float lo = 1.0f;
    float ret = 0.0f;
    for (size_t i = 0; i != sorted.size(); ++i) {
      auto [r, n] = sorted[i];
      float x = r - max;
      float o =
          x == 0 ? 1.0f : std::min(std::log(1 - (2.0f * 200) / x) / 4, 1.0f);
      if (r < rating) {
        ret += (lo - o) / i;
      }
      lo = o;
    }
    ret += lo / sorted.size();
    return ret;
  }

  std::string_view sampleModelId() {
    if (models.empty() ||
        std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.5) {
      return "dev";
    }
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.01) {
      auto it = models.begin();
      std::advance(
          it, std::uniform_int_distribution<size_t>(0, models.size() - 1)(rng));
      return it->first;
    }

    float max = 0.0f;
    for (auto& [id, m] : models) {
      max = std::max(max, m.rating);
    }
    double x = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double target = -(2.0f / (std::exp(x * 4) - 1)) * 200;
    std::vector<std::string_view> pool;
    for (auto& [id, m] : models) {
      double diff = m.rating - max;
      if (diff >= target) {
        pool.push_back(id);
      }
    }
    if (!pool.empty()) {
      return pool.at(
          std::uniform_int_distribution<size_t>(0, pool.size() - 1)(rng));
    }
    return "dev";
  }

  std::chrono::steady_clock::time_point lastRatingPrint =
      std::chrono::steady_clock::now();

  void addResult(std::string_view id, float ratio, float reward) {
    if (ratio < 0.9f) {
      return;
    }
    auto i = models.find(id);
    if (i == models.end()) {
      return;
    }
    auto di = models.find("dev");
    if (di == models.end()) {
      return;
    }

    if (i == di) {
      return;
    }

    float rating = i->second.rating;
    float devrating = di->second.rating;

    auto calc = [&](float reward, float diff) {
      float k = 6;
      float scale = 400;
      float offset = 0.5f;
      if (reward > 0) {
        offset = 1.0f;
      } else if (reward < 0) {
        offset = 0.0f;
      }
      return k * (offset - 1.0 / (1.0 + std::pow(10.0f, diff / scale)));
    };

    float delta = calc(reward, devrating - rating) * ratio;
    float delta2 = calc(-reward, rating - devrating) * ratio;

    rating += delta;
    devrating += delta2;

    i->second.rating = rating;
    di->second.rating = devrating;

    ++i->second.ngames;
    ++di->second.ngames;

    i->second.rewardsum += reward;
    di->second.rewardsum -= reward;

    i->second.avgreward = i->second.rewardsum / i->second.ngames;
    di->second.avgreward = di->second.rewardsum / di->second.ngames;

    auto now = std::chrono::steady_clock::now();
    if (now - lastRatingPrint >= std::chrono::seconds(120)) {
      lastRatingPrint = now;
      std::vector<std::pair<float, std::string_view>> sorted;
      for (auto& [id, m] : models) {
        sorted.emplace_back(m.rating, id);

        m.curgames = m.ngames - m.prevngames;
        m.curreward = (m.rewardsum - m.prevrewardsum) / m.curgames;

        m.prevngames = m.ngames;
        m.prevrewardsum = m.rewardsum;
      }
      std::sort(sorted.begin(), sorted.end(), std::greater<>());
      int devrank = 0;
      float devrating = 0;
      for (size_t i = 0; i != sorted.size(); ++i) {
        if (sorted[i].second == "dev") {
          devrank = (int)i + 1;
          devrating = sorted[i].first;
          break;
        }
      }
      if (sorted.size() > 20) {
        sorted.resize(20);
      }
      std::string str;
      int rank = 1;
      auto stringify = [&](int rank, float rating, std::string_view id) {
        return fmt::sprintf("%d. %g %s (roll chance %f) (total %d games, %f "
                            "avg reward) (diff %d games, %f avg reward)\n",
                            rank, rating, id, rollChance(id), models[id].ngames,
                            models[id].avgreward, models[id].curgames,
                            models[id].curreward);
      };
      for (auto& [rating, id] : sorted) {
        str += stringify(rank, rating, id);
        ++rank;
      }
      if (devrank > 20) {
        str += stringify(devrank, devrating, "dev");
      }
      fmt::printf("Top 20:\n%s", str);
    }
  }

  std::pair<std::string_view, int> requestModel(bool wantsNewModelId,
                                                std::string_view modelId) {
    std::unique_lock l(mut);
    if (wantsNewModelId) {
      modelId = sampleModelId();
    }
    int version = -1;
    auto i = models.find(modelId);
    if (i == models.end()) {
      modelId = "dev";
      i = models.find(modelId);
    }
    if (i != models.end()) {
      version = i->second.version;
    } else {
      version = -1;
    }
    addnetworkstats(*server, netstatsCounter);
    return {modelId, version};
  }

  std::optional<std::unordered_map<std::string, torch::Tensor>>
  requestStateDict(std::string_view modelId) {
    std::unique_lock l(mut);
    auto i = models.find(modelId);
    if (i == models.end()) {
      return {};
    } else {
      return i->second.stateDict;
    }
  }

  std::optional<std::vector<char>> requestCompressedStateDict(
      std::string_view modelId) {
    std::unique_lock l(mut);
    auto i = models.find(modelId);
    if (i == models.end()) {
      return {};
    } else {
      if (i->second.compressedStateDict.empty()) {
        for (int n = 0; n != 500 && i->second.compressing.exchange(true); ++n) {
          l.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          l.lock();
          i = models.find(modelId);
          if (i == models.end()) {
            return {};
          }
          if (!i->second.compressedStateDict.empty()) {
            return i->second.compressedStateDict;
          }
        }
        auto copy = i->second.stateDict;
        l.unlock();
        auto start = std::chrono::steady_clock::now();
        rpc::Serializer s;
        rpc::Serialize ser(s);
        ser(copy);
        auto now = std::chrono::steady_clock::now();
        double t1 =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::ratio<1, 1000>>>(now - start)
                .count();
        start = now;
        size_t oldsize = s.size();
        s.compress(15);
        size_t newsize = s.size();
        s.buf.shrink_to_fit();
        now = std::chrono::steady_clock::now();
        double t2 =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::ratio<1, 1000>>>(now - start)
                .count();
        start = now;

        fmt::printf("State dict serialized in %gms, compressed (from %gM to "
                    "%gM) in %gms\n",
                    t1, oldsize / 1024.0 / 1024.0, newsize / 1024.0 / 1024.0,
                    t2);

        l.lock();
        i = models.find(modelId);
        if (i == models.end()) {
          return {};
        }
        i->second.compressedStateDict = std::move(s.buf);
        i->second.compressing = false;
      }
      return i->second.compressedStateDict;
    }
  }

  void trainData(const std::unordered_map<std::string, torch::Tensor> data) {
    onTrainData(std::move(data));
  }

  void gameResult(
      std::vector<std::pair<float, std::unordered_map<std::string_view, float>>>
          result) {
    std::lock_guard l(mut);
    for (auto& [reward, models] : result) {
      for (auto& [id, ratio] : models) {
        addResult(id, ratio, reward);
      }
    }
  }

  struct rdmaClient {
    std::chrono::steady_clock::time_point timestamp;
    std::unique_ptr<rdma::Host> host;
    rdma::Endpoint localEp;
    rdma::Endpoint remoteEp;
  };

  std::unique_ptr<rdma::Context> rdmaContext;
  std::unique_ptr<rdma::CompletionQueue> rdmaCq;
  std::list<rdmaClient> rdmaClients;
  std::mutex rdmaMut;

  rdma::Endpoint rdmaConnect(rdma::Endpoint ep) {
    auto host = rdmaContext->createHost();
    auto localEp = host->init(*rdmaCq);
    host->connect(ep);

    // fmt::printf("rdmaConnect, remoteEp %d:%d localEp %d:%d\n", ep.lid,
    // ep.qpnum, localEp.lid, localEp.qpnum);

    std::lock_guard l(rdmaMut);
    auto now = std::chrono::steady_clock::now();
    for (auto i = rdmaClients.begin(); i != rdmaClients.end();) {
      if (now - i->timestamp >= std::chrono::minutes(1)) {
        // fmt::printf("RDMA client %d:%d timed out\n", i->remoteEp.lid,
        // i->remoteEp.qpnum);
        i = rdmaClients.erase(i);
      } else {
        ++i;
      }
    }
    rdmaClients.emplace_back();
    auto& c = rdmaClients.back();
    c.host = std::move(host);
    c.localEp = localEp;
    c.remoteEp = ep;
    c.timestamp = now;
    return localEp;
  }

  bool rdmaKeepalive(rdma::Endpoint remoteEp) {
    std::unique_lock rl(rdmaMut);
    for (auto i = rdmaClients.begin(); i != rdmaClients.end(); ++i) {
      if (i->remoteEp == remoteEp) {
        // fmt::printf("keepalive: rdma client %d:%d found, timestamp
        // updated\n", remoteEp.lid, remoteEp.qpnum);
        i->timestamp = std::chrono::steady_clock::now();
        return true;
      }
    }
    return false;
  }

  std::optional<RDMAModelInfo> rdmaGetModel(rdma::Endpoint remoteEp,
                                            std::string_view modelId) {
    try {
      std::unique_lock rl(rdmaMut);
      rdma::Endpoint localEp;
      for (auto i = rdmaClients.begin(); i != rdmaClients.end(); ++i) {
        if (i->remoteEp == remoteEp) {
          i->timestamp = std::chrono::steady_clock::now();
          localEp = i->localEp;
        }
      }
      rl.unlock();
      std::unique_lock l(mut);
      auto i = models.find(modelId);
      if (i == models.end()) {
        return {};
      }
      for (int n = 0;; ++n) {
        if (!i->second.rdmaBuffer ||
            i->second.rdmaBufferVersion != i->second.version) {
          if (n < 500 && i->second.rdmaSerializing.exchange(true)) {
            l.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            l.lock();
            i = models.find(modelId);
            if (i == models.end()) {
              return {};
            }
          } else {
            auto copy = i->second.stateDict;
            int version = i->second.version;
            l.unlock();
            auto start = std::chrono::steady_clock::now();
            rpc::Serializer s;
            rpc::Serialize ser(s);
            ser((uint32_t)0);
            ser(copy);

            uint32_t checksum =
                s.size() > 4 ? crc32(s.data() + 4, s.size() - 4) : 0;
            std::memcpy((void*)s.data(), &checksum, 4);

            l.lock();
            i = models.find(modelId);
            if (i == models.end()) {
              return {};
            }

            auto buffer = std::move(i->second.rdmaBuffer);
            auto storage = std::move(i->second.rdmaBufferStorage);
            // if (buffer && storage.size() >= s.size()) {
            if (false) {

              storage.resize(s.size());
              std::memcpy(storage.data(), s.data(), s.size());

              auto now = std::chrono::steady_clock::now();
              double t1 =
                  std::chrono::duration_cast<
                      std::chrono::duration<double, std::ratio<1, 1000>>>(now -
                                                                          start)
                      .count();
              fmt::printf("State dict serialized and RDMA buffer updated in "
                          "%gms, %gM (checksum %#x)\n",
                          t1, storage.size() / 1024.0 / 1024.0, checksum);
            } else {
              i->second.rdmaBuffer_01 = std::move(buffer);
              i->second.rdmaBufferStorage_01 = std::move(storage);

              storage = std::move(s.buf);
              buffer = rdmaContext->createBuffer(
                  (void*)storage.data(), storage.size());

              auto now = std::chrono::steady_clock::now();
              double t1 =
                  std::chrono::duration_cast<
                      std::chrono::duration<double, std::ratio<1, 1000>>>(now -
                                                                          start)
                      .count();
              fmt::printf("State dict serialized and RDMA buffer created in "
                          "%gms, %gM (checksum %#x)\n",
                          t1, storage.size() / 1024.0 / 1024.0, checksum);
            }

            i->second.rdmaBuffer = std::move(buffer);
            i->second.rdmaBufferStorage = std::move(storage);
            i->second.rdmaSerializing = false;
            i->second.rdmaBufferVersion = version;
          }
        } else {
          RDMAModelInfo r;
          r.key = i->second.rdmaBuffer->keyFor(localEp);
          r.checksum = i->second.rdmaBufferChecksum;
          r.address = (uintptr_t)i->second.rdmaBufferStorage.data();
          r.size = i->second.rdmaBufferStorage.size();
          return r;
        }
      }
    } catch (const std::exception& e) {
      fmt::printf("rdmaGetModel error: %s\n", e.what());
      return {};
    }
  }

  struct ModelInfo {
    std::string id;
    int version = 0;
    float rating = 0.0f;
    std::unordered_map<std::string, torch::Tensor> stateDict;
    std::vector<char> compressedStateDict;
    std::atomic<bool> compressing{false};
    uint64_t ngames = 0;
    double rewardsum = 0.0;
    float avgreward = 0.0f;

    uint64_t prevngames = 0;
    double prevrewardsum = 0.0;

    uint64_t curgames = 0;
    float curreward = 0.0f;

    double rollChance = 0.0;

    std::atomic<bool> rdmaSerializing{false};
    std::vector<char> rdmaBufferStorage;
    std::unique_ptr<rdma::Buffer> rdmaBuffer;
    int rdmaBufferVersion = -1;
    uint32_t rdmaBufferChecksum = 0;
    std::vector<char> rdmaBufferStorage_01;
    std::unique_ptr<rdma::Buffer> rdmaBuffer_01;
  };

  std::mutex mut;
  std::unordered_map<std::string_view, ModelInfo> models;

  std::mutex timemut;
  std::unordered_map<std::string, float> calltimes;
  std::chrono::steady_clock::time_point lasttimereport;

 public:
  std::function<void(const std::unordered_map<std::string, torch::Tensor>)>
      onTrainData;
  NetStatsCounter netstatsCounter;

  template <typename R, typename... Args>
  auto define(std::string name, R (ServerImpl::*f)(Args...)) {
    server->define(
        name, std::function<R(Args...)>([this, f, name](Args&&... args) {
          auto begin = std::chrono::steady_clock::now();
          auto finish = [&]() {
            auto end = std::chrono::steady_clock::now();
            double t = std::chrono::duration_cast<
                           std::chrono::duration<double, std::ratio<1, 1000>>>(
                           end - begin)
                           .count();
            {
              std::unique_lock l(timemut);
              auto i = calltimes.find(name);
              if (i == calltimes.end()) {
                i = calltimes.emplace(name, t).first;
              }
              float& v = i->second;
              v = v * 0.99 + t * 0.01;

              if (end - lasttimereport >= std::chrono::seconds(60)) {
                lasttimereport = end;
                std::string s = "RPC call times (running average):\n";
                for (auto& v : calltimes) {
                  s += fmt::sprintf("  %s  %fms\n", v.first, v.second);
                }
                l.unlock();
                fmt::printf("%s", s);
              }
            }
          };
          if constexpr (std::is_same_v<R, void>) {
            (this->*f)(std::forward<Args>(args)...);
            finish();
          } else {
            auto rv = (this->*f)(std::forward<Args>(args)...);
            finish();
            return rv;
          }
        }));
  }

  void start(std::string_view endpoint) {
    if (endpoint.substr(0, 6) == "tcp://") {
      endpoint.remove_prefix(6);
    }
    printf("actual listen endpoint is %s\n", std::string(endpoint).c_str());
    server = getRpc().listen("");

    define("requestModel", &ServerImpl::requestModel);
    define("requestStateDict", &ServerImpl::requestStateDict);
    define(
        "requestCompressedStateDict", &ServerImpl::requestCompressedStateDict);
    define("trainData", &ServerImpl::trainData);
    define("gameResult", &ServerImpl::gameResult);

    try {
      rdmaContext = rdma::create();
      if (!rdmaContext) {
        fmt::printf("RDMA/IB is not supported\n");
      } else {
        rdmaCq = rdmaContext->createCQ(4);
        auto testHost = rdmaContext->createHost();

        define("rdmaConnect", &ServerImpl::rdmaConnect);
        define("rdmaKeepalive", &ServerImpl::rdmaKeepalive);
        define("rdmaGetModel", &ServerImpl::rdmaGetModel);

        fmt::printf("RDMA over IB supported\n");
      }
    } catch (const std::exception& e) {
      fmt::printf("RDMA error: %s\nRDMA/IB will not be used\n", e.what());
    }

    server->listen(endpoint);
  }

  void updateModel(const std::string& id,
                   std::unordered_map<std::string, torch::Tensor> stateDict) {
    std::unique_lock l(mut);
    auto i = models.try_emplace(id);
    if (i.second) {
      i.first->second.version =
          std::uniform_int_distribution<int>(0, 10000)(rng) * 1000;
      i.first->second.id = id;
      (std::string_view&)i.first->first = i.first->second.id;
      auto idev = models.find("dev");
      if (idev != models.end()) {
        i.first->second.rating = idev->second.rating;
      }
    }
    auto& m = i.first->second;
    m.stateDict = std::move(stateDict);
    ++m.version;
    m.compressedStateDict.clear();
  }
};

class ClientImpl {

  std::shared_ptr<rpc::Client> client;

  mutable std::mutex mut;
  std::unordered_set<std::string> allModelIds;
  std::string_view currentModelId = *allModelIds.emplace("dev").first;
  int currentModelVersion = -1;
  int gamesDoneWithCurrentModel = 0;
  bool wantsNewModelId = false;
  bool wantsTournamentResult_ = false;

  std::chrono::steady_clock::time_point lastCheckTournamentResult =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point lastTournamentResult =
      std::chrono::steady_clock::now();

  std::vector<std::pair<float, std::unordered_map<std::string_view, float>>>
      resultQueue;

  NetStatsCounter netstatsCounter;

  std::unique_ptr<rdma::Context> rdmaContext;
  std::unique_ptr<rdma::Host> rdmaHost;
  std::unique_ptr<rdma::Buffer> rdmaBuffer;
  size_t rdmaBufferSize = 0;
  std::vector<char> rdmaBufferStorage;
  std::optional<rdma::Endpoint> rdmaEndpoint;
  std::unique_ptr<rdma::CompletionQueue> rdmaCq;

  std::mutex trainDataMut;
  std::vector<std::future<void>> trainDataFutures;

  struct Bandit {
    std::mutex mut;
    std::unordered_map<std::string, float> value;
    std::minstd_rand rng{std::random_device{}()};
    float sample(std::string name, float weight = 1.0f) {
      std::lock_guard l(mut);
      return std::uniform_real_distribution<float>(
          0.0f, std::exp(value[name] * 4) * weight)(rng);
    }
    float get(std::string name) {
      std::lock_guard l(mut);
      return value[name];
    }
  };

  struct BanditResultCounter {
    Bandit& b;
    std::string name;
    bool succeeded_ = false;
    BanditResultCounter(Bandit& b, std::string name)
        : b(b)
        , name(std::move(name)) {
    }
    void success(bool succeeded = true) {
      succeeded_ = succeeded;
    }
    ~BanditResultCounter() {
      std::lock_guard l(b.mut);
      float& v = b.value[name];
      v = v * 0.95f + (succeeded_ ? 1.0f : -1.0f) * 0.05f;
    }
  };

  Bandit bandit;

  bool createRdmaHost() {
    if (!rdmaContext) {
      return false;
    }
    if (rdmaHost) {
      return true;
    }
    try {
      rdmaHost = rdmaContext->createHost();
      return true;
    } catch (const std::exception& e) {
      fmt::printf("RDMA error: %s\nRDMA/IB will not be used\n", e.what());
      return false;
    }
  }

  void requestModelStateDict(std::string modelId, int modelVersion) {
    try {

      auto start = std::chrono::steady_clock::now();

      float rdmaValue = bandit.get("rdma");
      float rpcValue = bandit.get("rpc");

      fmt::printf("bandit values: rdma %g rpc %g\n", rdmaValue, rpcValue);

      if ((rdmaHost || createRdmaHost()) &&
          (rdmaValue >= 0.75f || (rdmaValue >= 0.0f && rpcValue < 0.5f) ||
           bandit.sample("rdma", 4.0f) > bandit.sample("rpc"))) {

        BanditResultCounter bc(bandit, "rdma");

        try {

          if (!rdmaEndpoint) {

            rdmaCq = rdmaContext->createCQ(4);
            rdmaEndpoint = rdmaHost->init(*rdmaCq);

            fmt::printf("local endpoint is %d:%d\n", rdmaEndpoint->lid,
                        rdmaEndpoint->qpnum);

            auto remoteEp =
                client->sync<rdma::Endpoint>("rdmaConnect", *rdmaEndpoint);
            addnetworkstats(*client, netstatsCounter);

            fmt::printf(
                "remote endpoint is %d:%d\n", remoteEp.lid, remoteEp.qpnum);

            rdmaHost->connect(remoteEp);
          }

          auto result = client->async<std::optional<RDMAModelInfo>>(
              "rdmaGetModel", *rdmaEndpoint, modelId);
          auto mi = result.get();

          if (!mi) {
            std::lock_guard l(mut);
            currentModelId = "dev";
            currentModelVersion = -1;
          } else {

            if (!rdmaBuffer || rdmaBufferSize < mi->size) {
              rdmaBufferStorage.resize(mi->size);
              rdmaBuffer =
                  rdmaContext->createBuffer(rdmaBufferStorage.data(), mi->size);
            }

            rdmaHost->read(*rdmaBuffer, rdmaBufferStorage.data(), mi->key,
                           mi->address, mi->size);
            rdmaHost->wait();

            std::unordered_map<std::string, torch::Tensor> stateDict;
            rpc::Deserializer d(rdmaBufferStorage.data(), mi->size);
            rpc::Deserialize des(d);
            uint32_t checksum = 0;
            des(checksum);
            if (mi->size > 4 &&
                crc32(rdmaBufferStorage.data() + 4, mi->size - 4) == checksum) {
              fmt::printf("RDMA model checksum OK (%#x)\n", checksum);
              des(stateDict);
              onUpdateModel(modelId, stateDict);
              std::lock_guard l(mut);
              if (currentModelId != modelId) {
                currentModelId = *allModelIds.emplace(modelId).first;
                gamesDoneWithCurrentModel = 0;
              }
              currentModelVersion = modelVersion;
              fmt::printf("Got model '%s' version %d\n", modelId, modelVersion);
              bc.success();
            } else {
              fmt::printf("RDMA model checksum error\n");
              return;
            }
          }

        } catch (const rdma::Error& e) {
          fmt::printf("RDMA error: %s\n", e.what());

          rdmaEndpoint.reset();
          rdmaHost.reset();
          rdmaCq.reset();
          return;
        }

      } else {

        BanditResultCounter bc(bandit, "rpc");

        auto result = client->async<std::optional<std::vector<char>>>(
            "requestCompressedStateDict", modelId);
        auto compressed = result.get();
        addnetworkstats(*client, netstatsCounter);
        if (!compressed) {
          std::unique_lock l(mut);
          currentModelId = "dev";
          currentModelVersion = -1;
        } else {
          std::unordered_map<std::string, torch::Tensor> stateDict;
          rpc::Deserializer d(compressed->data(), compressed->size());
          d.decompress();
          rpc::Deserialize des(d);
          des(stateDict);
          onUpdateModel(modelId, stateDict);
          std::unique_lock l(mut);
          if (currentModelId != modelId) {
            currentModelId = *allModelIds.emplace(modelId).first;
            gamesDoneWithCurrentModel = 0;
          }
          currentModelVersion = modelVersion;
          fmt::printf("Got model '%s' version %d\n", modelId, modelVersion);
          bc.success();
        }
      }

      double t = std::chrono::duration_cast<
                     std::chrono::duration<double, std::ratio<1, 1000>>>(
                     std::chrono::steady_clock::now() - start)
                     .count();
      fmt::printf("State dict received and updated in %gms\n", t);

    } catch (const rpc::RPCException& e) {
      fmt::printf("RPC exception: %s\n", e.what());
    }
  }

 public:
  std::function<void(
      std::string_view, std::unordered_map<std::string, torch::Tensor>)>
      onUpdateModel;

  ClientImpl() {
    try {
      rdmaContext = rdma::create();
      if (!rdmaContext) {
        fmt::printf("RDMA/IB is not supported\n");
      } else {
        rdmaHost = rdmaContext->createHost();

        fmt::printf("Using RDMA over IB for model transfers\n");
      }
    } catch (const std::exception& e) {
      fmt::printf("RDMA error: %s\nRDMA/IB will not be used\n", e.what());
    }
  }

  void requestModel(bool isTournamentOpponent) {
    try {
      std::unique_lock l(mut);
      if (!resultQueue.empty()) {
        client->async("gameResult", resultQueue);
        resultQueue.clear();
      }

      //      fmt::printf(
      //          "Request model, isTournamentOpponent %d, wantsNewModelId
      //          %d\n", isTournamentOpponent, wantsNewModelId);

      auto result = client->async<std::pair<std::string, int>>(
          "requestModel",
          isTournamentOpponent ? std::exchange(wantsNewModelId, false) : false,
          currentModelId);
      l.unlock();

      if (rdmaEndpoint) {
        if (!client->sync<bool>("rdmaKeepalive", *rdmaEndpoint)) {
          rdmaEndpoint.reset();
          rdmaHost.reset();
          rdmaCq.reset();
        }
      }

      auto [newId, version] = result.get();
      addnetworkstats(*client, netstatsCounter);

      // fmt::printf("Got model '%s'\n", newId);

      l.lock();
      auto now = std::chrono::steady_clock::now();
      if (isTournamentOpponent &&
          now - lastCheckTournamentResult >= std::chrono::minutes(2)) {
        lastCheckTournamentResult = now;
        wantsTournamentResult_ =
            now - lastTournamentResult >= std::chrono::minutes(5);
        if (!wantsTournamentResult_) {
          wantsNewModelId = true;
        }
        //        fmt::printf("wantsTournamentResult_ is %d, wantsNewModelId is
        //        %d\n",
        //                    wantsTournamentResult_,
        //                    wantsNewModelId);
      } else if (!isTournamentOpponent) {
        wantsTournamentResult_ = false;
      }
      if (currentModelId != newId || version != currentModelVersion) {
        l.unlock();
        requestModelStateDict(newId, version);
      } else {
        l.unlock();
      }
    } catch (const rpc::RPCException& e) {
      fmt::printf("RPC exception: %s\n", e.what());
    }
  }

  void connect(std::string_view endpoint) {
    if (endpoint.substr(0, 6) == "tcp://") {
      endpoint.remove_prefix(6);
    }
    printf("actual connect endpoint is %s\n", std::string(endpoint).c_str());
    client = getRpc().connect(endpoint);
  }

  void sendTrainData(
      const std::unordered_map<std::string, torch::Tensor>& data) {
    try {
      std::unique_lock l(trainDataMut);
      std::future<void> fut;
      if (trainDataFutures.size() >= 32) {
        fut = std::move(trainDataFutures.front());
        trainDataFutures.erase(trainDataFutures.begin());
      }
      l.unlock();
      if (fut.valid()) {
        fut.get();
      }
    } catch (const rpc::RPCException& e) {
      fmt::printf("RPC exception: %s\n", e.what());
    }
    try {
      auto fut = client->async<void>("trainData", data);
      std::lock_guard l(trainDataMut);
      trainDataFutures.push_back(std::move(fut));
    } catch (const rpc::RPCException& e) {
      fmt::printf("RPC exception: %s\n", e.what());
    }
  }

  void sendResult(float reward,
                  std::unordered_map<std::string_view, float> models) {
    std::unique_lock l(mut);
    auto i = models.find(currentModelId);
    if (i != models.end()) {
      if (i->second >= 0.9f) {
        ++gamesDoneWithCurrentModel;
        if (gamesDoneWithCurrentModel >= 20) {
          lastTournamentResult = std::chrono::steady_clock::now();
          wantsNewModelId = true;
        }
      }
    }
    resultQueue.emplace_back(reward, std::move(models));
  }

  bool wantsTournamentResult() const {
    std::unique_lock l(mut);
    return wantsTournamentResult_;
  }

  std::string_view getModelId() const {
    std::unique_lock l(mut);
    return currentModelId;
  }
};

Server::Server() {
  impl = std::make_unique<ServerImpl>();
}
Server::~Server() {
}

void Server::setOnTrainData(
    std::function<void(std::unordered_map<std::string, torch::Tensor>)>
        onTrainData) {
  impl->onTrainData = std::move(onTrainData);
}

void Server::start(std::string endpoint) {
  impl->start(endpoint);
}

void Server::updateModel(
    const std::string& id,
    std::unordered_map<std::string, torch::Tensor> stateDict) {
  impl->updateModel(id, std::move(stateDict));
}

Client::Client() {
  impl = std::make_unique<ClientImpl>();
}

Client::~Client() {
}

void Client::setOnUpdateModel(
    std::function<void(std::string_view,
                       std::unordered_map<std::string, torch::Tensor>)>
        onUpdateModel) {
  impl->onUpdateModel = std::move(onUpdateModel);
}

void Client::connect(std::string endpoint) {
  impl->connect(endpoint);
}

void Client::requestModel(bool isTournamentOpponent) {
  impl->requestModel(isTournamentOpponent);
}

void Client::sendTrainData(
    const std::unordered_map<std::string, torch::Tensor>& data) {
  impl->sendTrainData(data);
}

bool Client::wantsTournamentResult() {
  return impl->wantsTournamentResult();
}

std::string_view Client::getModelId() {
  return impl->getModelId();
}

void Client::sendResult(float reward,
                        std::unordered_map<std::string_view, float> models) {
  impl->sendResult(reward, std::move(models));
}

}  // namespace distributed
