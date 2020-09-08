
#include "model_manager.h"

#include "common/async.h"
#include "common/thread_id.h"
#include "distributed/distributed.h"
#include "distributed/rpc.h"
#include "replay_buffer.h"
#include "tube/src_cpp/data_channel.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <fmt/printf.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace core {

std::unordered_map<std::string, at::Tensor> convertIValueToMap(
    const c10::IValue& value) {
  std::unordered_map<std::string, torch::Tensor> map;
  auto dict = value.toGenericDict();

#ifdef PYTORCH12
  for (auto& name2tensor : dict) {
    auto name = name2tensor.key().toString();
    torch::Tensor tensor = name2tensor.value().toTensor();
#else
  auto ivalMap = dict->elements();
  for (auto& name2tensor : ivalMap) {
    auto name = name2tensor.first.toString();
    torch::Tensor tensor = name2tensor.second.toTensor();
#endif

    tensor = tensor.detach();
    map.insert({name->string(), tensor});
  }
  return map;
}

// A mutex where threads are strictly ordered by their priority when waiting to
// acquire the mutex.
// Threads must call PriorityMutex::setThreadPriority to set their priority.
// Lower priority values will acquire the mutex first.
class PriorityMutex {
  struct TLData {
    TLData* next = nullptr;
    int priority = 0;
    std::condition_variable cv;
    bool waiting = false;
  };

  static inline thread_local std::unique_ptr<TLData> tldata;

  std::mutex mut;
  TLData* queue = nullptr;
  TLData* owner = nullptr;

  static TLData& getTLData() {
    if (!tldata) {
      tldata = std::make_unique<TLData>();
    }
    return *tldata;
  }

 public:
  static void setThreadPriority(int priority) {
    getTLData().priority = priority;
  }

  void lock() {
    std::unique_lock<std::mutex> l(mut);
    auto& tld = getTLData();
    if (!owner) {
      owner = &tld;
      return;
    } else {
      if (!queue) {
        queue = &tld;
      } else {
        TLData** insert = &queue;
        TLData* next = queue;
        while (next && next->priority <= tld.priority) {
          insert = &next->next;
          next = next->next;
        }
        tld.next = next;
        *insert = &tld;
      }
    }
    tld.waiting = true;
    while (tld.waiting) {
      tld.cv.wait(l);
    }
  }
  void unlock() {
    std::unique_lock<std::mutex> l(mut);
    if (queue) {
      auto* next = queue;
      next->waiting = false;
      queue = next->next;
      owner = next;
      next->next = nullptr;
      l.unlock();
      next->cv.notify_all();
    } else {
      owner = nullptr;
    }
  }
};

namespace {
template <typename T> struct StreamBuffer : std::streambuf {
  T& buf;
  StreamBuffer(T& buf)
      : buf(buf) {
  }
  virtual std::streamsize xsputn(const char_type* s,
                                 std::streamsize count) override {
    if (buf.capacity() < buf.size() + count) {
      buf.reserve(std::max(buf.size() * 2, buf.size() + count + 16));
    }
    size_t prevSize = buf.size();
    buf.resize(buf.size() + count);
    char* dst = buf.data() + prevSize;
    memcpy(dst, s, count);
    return count;
  }

  void write(const std::string& str) {
    size_t len = str.size();
    xsputn((char*)&len, sizeof(len));
    xsputn(str.data(), str.size());
  }
};

template <typename T> struct StreamReadBuffer : std::streambuf {
  T& buf;
  size_t readPos = 0;
  StreamReadBuffer(T& buf)
      : buf(buf) {
  }
  virtual std::streamsize xsgetn(char_type* s, std::streamsize count) override {
    count = std::min((size_t)count, buf.size() - readPos);
    memcpy(s, buf.data() + readPos, count);
    readPos += count;
    return count;
  }

  std::string_view read() {
    size_t len = 0;
    size_t n = xsgetn((char*)&len, sizeof(len));
    if (n != sizeof(len)) {
      return {};
    }
    if (buf.size() - readPos < len) {
      return {};
    }
    std::string_view r{buf.data() + readPos, len};
    readPos += len;
    return r;
  }
};

std::mutex deviceMutexMutex;
std::unordered_map<std::string, PriorityMutex> deviceMutex;
PriorityMutex* getDeviceMutex(std::string device) {
  std::lock_guard l(deviceMutexMutex);
  return &deviceMutex[device];
}

}  // namespace

using TorchJitModel = torch::jit::script::Module;

class ModelManagerImpl {
 public:
  ModelManagerImpl(int actBatchsize,
                   const std::string& device,
                   int replayCapacity,
                   int seed,
                   const std::string& jitModel,
                   int trainChannelTimeoutMs,
                   int trainChannelNumSlots)
      : jitModel_(jitModel)
      , device_(device)
      , replayBuffer_(replayCapacity, seed) {
    trainChannel_ = std::make_shared<tube::DataChannel>(
        "train", trainChannelNumSlots, trainChannelTimeoutMs);
    actChannel_ = std::make_shared<tube::DataChannel>("act", actBatchsize, -1);

#ifdef PYTORCH12
    model_ =
        std::make_shared<TorchJitModel>(torch::jit::load(jitModel_, device));
#else
    model_ = torch::jit::load(jitModel_, device);
#endif
    model_->eval();

    dtype_ = at::ScalarType::Float;

    model_->to(dtype_);

    modelMutex_ = getDeviceMutex(device);
  }

  ~ModelManagerImpl() {
    terminate_ = true;
    actChannel_->terminate();
    trainChannel_->terminate();
    for (auto& v : threads_) {
      v.join();
    }
    if (modelUpdateThread.joinable()) {
      modelUpdateThread.join();
    }
  }

  void startServer(std::string serverListenEndpoint) {
    server_.emplace();
    server_->setOnTrainData([this](std::unordered_map<std::string, torch::Tensor> batch) {
      replayBuffer_.add(std::move(batch));
    });
    server_->start(serverListenEndpoint);
    fmt::printf("Listening on %s\n", serverListenEndpoint);
  }

  void startClient(std::string serverConnectHostname) {
    auto firstUpdate = std::make_shared<std::promise<bool>>();
    auto firstUpdateFuture = firstUpdate->get_future();
    client_.emplace();
    client_->setOnUpdateModel(
        [this, firstUpdate](
            std::string_view id,
            std::unordered_map<std::string, torch::Tensor> dict) mutable {
          if (firstUpdate) {
            firstUpdate->set_value(true);
            firstUpdate.reset();
          }
          if (!dontRequestModelUpdates_) {
            fmt::printf("onUpdateModel '%s'\n", id);
            updateModel(dict);
          }
        });
    client_->connect(serverConnectHostname);
    fmt::printf("Connected to %s\n", serverConnectHostname);

    modelUpdateThread = std::thread([this]() {
      while (!terminate_ && !trainChannel_->terminated()) {
        if (!dontRequestModelUpdates_) {
          client_->requestModel(isTournamentOpponent_);
        }
        for (int i = 0; i != 2 && !terminate_ && !trainChannel_->terminated();
             ++i) {
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
      }
    });

    if (!dontRequestModelUpdates_) {
      fmt::printf("Waiting for model\n");
      firstUpdateFuture.wait();
      fmt::printf("Received model\n");
    } else {
      fmt::printf("Not requesting model updates\n");
    }
  }

  std::unique_ptr<rpc::Rpc> replayBufferRpc;
  std::shared_ptr<rpc::Server> replayBufferRpcServer;
  std::shared_ptr<rpc::Client> replayBufferRpcClient;

  void startReplayBufferServer(std::string endpoint) {
    if (endpoint.substr(0, 6) == "tcp://") {
      endpoint.erase(0, 6);
    }
    if (!replayBufferRpc) {
      replayBufferRpc = std::make_unique<rpc::Rpc>();
      replayBufferRpc->asyncRun(8);
    }

    replayBufferRpcServer = replayBufferRpc->listen("");
    replayBufferRpcServer->define("sample", &ModelManagerImpl::sample, this);

    replayBufferRpcServer->listen(endpoint);
  }

  void startReplayBufferClient(std::string endpoint) {
    if (endpoint.substr(0, 6) == "tcp://") {
      endpoint.erase(0, 6);
    }
    if (!replayBufferRpc) {
      replayBufferRpc = std::make_unique<rpc::Rpc>();
      replayBufferRpc->asyncRun(8);
    }

    replayBufferRpcClient = replayBufferRpc->connect(endpoint);
  }

  SampleResult remoteSample(int sampleSize) {
    return {replayBufferRpcClient
                ->async<std::unordered_map<std::string, torch::Tensor>>(
                    "sample", sampleSize)};
  }

  std::shared_ptr<tube::DataChannel> getTrainChannel() {
    return trainChannel_;
  }

  std::shared_ptr<tube::DataChannel> getActChannel() {
    return actChannel_;
  }

  std::unordered_map<std::string, torch::Tensor> cloneStateDict(
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    torch::NoGradGuard ng;
    std::unordered_map<std::string, torch::Tensor> r;
    for (auto& [name, tensor] : stateDict) {
      r[name] = tensor.detach().to(
          torch::TensorOptions().device(torch::kCPU).dtype(dtype_), false, true);
    }
    return r;
  }

  void addTournamentModel(
      std::string id,
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    if (server_) {
      fmt::printf(" -- ADD MODEL %s --\n", id);
      server_->updateModel(id, cloneStateDict(stateDict));
    }
  }

#ifdef PYTORCH15
  void loadModelStateDict(
      TorchJitModel& model,
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    std::unordered_map<std::string, torch::Tensor> dst;
    for (const auto& v : model.named_parameters()) {
      dst[v.name] = v.value;
    }
    for (const auto& v : model.named_buffers()) {
      dst[v.name] = v.value;
    }

    for (auto& [k, v] : stateDict) {
      auto i = dst.find(k);
      if (i == dst.end()) {
        throw std::runtime_error(
            fmt::sprintf("key '%s' not found in destination state dict", k));
      } else if (i->second.sizes() != v.sizes()) {
        throw std::runtime_error(
            fmt::sprintf("state dict key '%s' shape mismatch", k));
      }
    }

    for (auto& [k, v] : dst) {
      auto i = stateDict.find(k);
      if (i == stateDict.end()) {
        throw std::runtime_error(
            fmt::sprintf("key '%s' not found in source state dict", k));
      }
    }

    fmt::printf("loadModelStateDict: state dicts OK\n");

    for (auto& v : stateDict) {
      auto i = dst.find(v.first);
      if (i != dst.end()) {
        dst.at(v.first).copy_(v.second).detach();
      } else {
        fmt::printf(
            "copyModelStateDict: Unknown state dict entry '%s'\n", v.first);
        std::abort();
      }
    }
    model.eval();
  }
#else
  void loadModelStateDict(
      TorchJitModel& model,
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    for (auto& [name, tensor] : stateDict) {
      const char* ptr = name.c_str();
      std::string memberNameString;
      const char* memberNamePtr = ptr;
      auto* currentModule = &model;
      decltype(currentModule->find_module(memberNameString)) subModule;
      while (*ptr) {
        if (*ptr == '.') {
          memberNameString.assign(memberNamePtr, ptr - memberNamePtr);
          subModule = currentModule->find_module(memberNameString);
          if (!subModule) {
            fmt::printf(
                "copyModelStateDict: Unknown state dict entry '%s' -- could "
                "not find module '%s'\n",
                name, memberNameString);
            std::abort();
          }
          currentModule = &*subModule;
          ++ptr;
          memberNamePtr = ptr;
        } else {
          ++ptr;
        }
      }
      memberNameString.assign(memberNamePtr, ptr - memberNamePtr);

      if (auto p = currentModule->find_parameter(memberNameString); p) {
        p->value().toTensor().copy_(tensor);
      } else if (auto b = currentModule->find_buffer(memberNameString); b) {
        b->value().toTensor().copy_(tensor);
      } else {
        fmt::printf(
            "copyModelStateDict: Unknown state dict entry '%s' -- could not "
            "find parameter/buffer '%s'\n",
            name, memberNameString);
        std::abort();
      }
    }
    model.eval();
  }
#endif

  void updateModel(
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    torch::NoGradGuard ng;
    fmt::printf(" -- UPDATE MODEL --\n");
    if (server_) {
      server_->updateModel("dev", cloneStateDict(stateDict));
    }
    PriorityMutex::setThreadPriority(-9);
    std::lock_guard<PriorityMutex> lk(*modelMutex_);
    loadModelStateDict(*model_, stateDict);
  }

  int bufferSize() const {
    return replayBuffer_.size();
  }

  bool bufferFull() const {
    return replayBuffer_.full();
  }

  std::unordered_map<std::string, torch::Tensor> sample(int sampleSize) {
    return replayBuffer_.sample(sampleSize);
  }

  void start() {
    threads_.emplace_back(&ModelManagerImpl::trainThread, this);

    threads_.emplace_back(&ModelManagerImpl::actThread, this);
  }

  void trainThread() {
    torch::NoGradGuard ng;
    if (client_) {
      std::atomic<bool> qdone{false};
      int qwaiters = 0;
      std::mutex qmut;
      std::condition_variable qcv;
      std::deque<std::unordered_map<std::string, torch::Tensor>> queue;
      std::vector<std::thread> qthreads;
      for (int i = 0; i != 4; ++i) {
        qthreads.emplace_back([&]() {
          while (true) {
            std::unique_lock l(qmut);
            ++qwaiters;
            while (queue.empty()) {
              if (qdone) {
                --qwaiters;
                return;
              }
              qcv.wait(l);
            }
            --qwaiters;
            auto batch = std::move(queue.front());
            queue.pop_front();
            l.unlock();
            client_->sendTrainData(batch);
          }
        });
      }
      while (true) {
        auto batch = trainChannel_->getInput();
        if (terminate_ || trainChannel_->terminated()) {
          break;
        }
        trainChannel_->setReply({});
        std::lock_guard l(qmut);
        if (queue.size() < 128) {
          queue.push_back(batch);
        } else {
          fmt::printf("Warning: train data queue is full, discarding data\n");
        }
        if (qwaiters) {
          qcv.notify_one();
        }
      }
      std::lock_guard l(qmut);
      qdone = true;
      qcv.notify_all();
    } else {
      while (true) {
        auto batch = trainChannel_->getInput();
        if (terminate_ || trainChannel_->terminated()) {
          break;
        }
        replayBuffer_.add(batch);
        trainChannel_->setReply({});
      }
    }
  }

  void actThread() {
    torch::NoGradGuard ng;
    while (true) {
      auto batch = actChannel_->getInput();
      if (terminate_ || actChannel_->terminated()) {
        break;
      }
      // TODO[hengyuan]: temp hard code
      auto s = batch["s"].to(device_);
      std::vector<torch::jit::IValue> input;
      input.push_back(s);
      PriorityMutex::setThreadPriority(-1);
      std::unique_lock<PriorityMutex> lk(*modelMutex_);
      auto output = model_->forward(input);
      lk.unlock();
      auto reply = convertIValueToMap(output);
      actChannel_->setReply(reply);
    }
  }

  void testAct() {
    torch::NoGradGuard ng;
    std::vector<torch::jit::IValue> inputs;
    auto x = torch::ones({1, 6 * 7 * 2}, torch::kFloat32);
    inputs.push_back(x);
    auto y = model_->forward(inputs);
    auto reply = convertIValueToMap(y);
    for (auto& name2tensor : reply) {
      std::cout << name2tensor.first << ": " << std::endl;
      std::cout << name2tensor.second << std::endl;
    }
  }

  void batchAct(torch::Tensor input,
                torch::Tensor v,
                torch::Tensor pi,
                torch::Tensor rnnState = {},
                torch::Tensor* rnnStateOut = nullptr) {
    torch::NoGradGuard ng;
    bool isCuda = device_.is_cuda();
    PriorityMutex::setThreadPriority(common::getThreadId());
    std::optional<c10::cuda::CUDAStreamGuard> g;
    if (isCuda) {
      g.emplace(c10::cuda::getStreamFromPool(false, device_.index()));
    }
    std::vector<torch::jit::IValue> inp;
    inp.push_back(input.to(device_, dtype_, true));
    if (rnnState.defined()) {
      inp.push_back(rnnState.to(device_, dtype_, true));
    }
    std::unique_lock<PriorityMutex> lk(*modelMutex_);
    auto output = model_->forward(inp);
    if (isCuda) {
      g->current_stream().synchronize();
    }
    lk.unlock();
    auto reply = convertIValueToMap(output);
    v.copy_(reply["v"], true);
    pi.copy_(reply["pi_logit"], true);
    if (rnnStateOut) {
      *rnnStateOut = reply["rnn_state"];
    }
    //    if (rnnStateOut.defined()) {
    //      rnnStateOut.copy_(reply["rnn_state"], true);
    //    }
    if (isCuda) {
      g->current_stream().synchronize();
    }
  }

  struct Timer {
    std::chrono::steady_clock::time_point start;
    Timer() {
      reset();
    }
    void reset() {
      start = std::chrono::steady_clock::now();
    }
    float elapsedAt(std::chrono::steady_clock::time_point now) {
      return std::chrono::duration_cast<
                 std::chrono::duration<double, std::ratio<1, 1>>>(now - start)
          .count();
    }
    float elapsed() {
      return elapsedAt(std::chrono::steady_clock::now());
    }
    float elapsedReset() {
      auto now = std::chrono::steady_clock::now();
      float r = elapsedAt(now);
      start = now;
      return r;
    }
  };

  int findBatchSize(torch::Tensor input, torch::Tensor rnnState = {}) {
    if (hasFoundBatchSize_) {
      return foundBatchSize_;
    }
    torch::NoGradGuard ng;
    bool isCuda = device_.is_cuda();
    PriorityMutex::setThreadPriority(common::getThreadId());
    std::optional<c10::cuda::CUDAStreamGuard> g;
    if (isCuda) {
      g.emplace(c10::cuda::getStreamFromPool(false, device_.index()));
    } else {
      return 1;
    }
    std::vector<torch::jit::IValue> inp;
    torch::Tensor gpuinput = input.to(device_, dtype_, true);
    torch::Tensor gpurnnState;
    if (rnnState.defined()) {
      gpurnnState = rnnState.to(device_, dtype_, true);
    }
    auto prep = [&](int bs) {
      inp.clear();
      std::vector<torch::Tensor> batch;
      for (int i = 0; i != bs; ++i) {
        batch.push_back(gpuinput);
      }
      inp.push_back(torch::stack(batch).to(device_, dtype_, true));
      if (rnnState.defined()) {
        batch.clear();
        for (int i = 0; i != bs; ++i) {
          batch.push_back(gpurnnState);
        }
        inp.push_back(torch::stack(batch).to(device_, dtype_, true));
      }
      g->current_stream().synchronize();
    };
    std::unique_lock<PriorityMutex> lk(*modelMutex_);
    if (hasFoundBatchSize_) {
      return foundBatchSize_;
    }
    auto call = [&]() {
      model_->forward(inp);
      if (isCuda) {
        g->current_stream().synchronize();
      }
    };
    fmt::printf("Finding batch size\n");
    prep(1);
    // warm up
    for (int i = 0; i != 10; ++i) {
      call();
    }
    Timer t;
    for (int i = 0; i != 10; ++i) {
      call();
    }
    float call1 = t.elapsed() / 10.0f * 1000.0f;
    fmt::printf("Base latency: %gms\n", call1);

    float maxms = 100.0f;
    int maxbs = 10240;

    struct I {
      float latency = 0.0f;
      float throughput = 0.0f;
      int n = 0;
      float score() {
        return latency / n / 400 - std::log(throughput / n);
      }
    };

    std::map<int, I> li;

    int best = 0;
    float bestScore = std::numeric_limits<float>::infinity();

    auto eval = [&](int i) {
      prep(i);
      int badcount = 0;
      float latency = 0.0f;
      float throughput = 0.0f;
      int n = 2;
      for (int j = 0; j != n; ++j) {
        call();
      }
      for (int j = 0; j != n; ++j) {
        t.reset();
        call();
        float ms = t.elapsed() * 1000;
        latency += ms;
        throughput += i / ms;
        if (ms > maxms || i >= maxbs) {
          ++badcount;
        }
      }
      auto& x = li[i];
      x.latency += latency;
      x.throughput += throughput;
      x.n += n;
      float score = x.score();
      if (badcount < n && score < bestScore) {
        bestScore = score;
        best = i;
      }
      return badcount < n;
    };

    for (int i = 1;; i += (i + 1) / 2) {
      if (!eval(i)) {
        break;
      }
    }
    std::minstd_rand rng(std::random_device{}());

    auto expandNear = [&](int k) {
      int r = 0;
      auto i = li.find(k);
      if (i != li.end()) {
        auto search = [&](auto begin, auto end) {
          int b = begin->first;
          int e;
          if (end == li.end()) {
            e = std::prev(end)->first;
          } else {
            e = end->first;
          }
          b = std::max(b, i->first - 3);
          e = std::max(b, i->first + 6);
          for (int i = b; i != e; ++i) {
            if (li.find(i) != li.end()) {
              continue;
            }
            ++r;
            if (!eval(i)) {
              break;
            }
          }
        };
        search(i, std::next(i));
        if (i != li.begin()) {
          search(std::prev(i), i);
        }
      }
      return r;
    };

    for (int j = 0; j != 4; ++j) {
      int expands = 12;
      for (int k = 0; k != 12; ++k) {
        float sum = 0.0f;
        std::vector<std::tuple<float, int, int>> list;
        float minweight = std::numeric_limits<float>::infinity();
        for (auto& [k, v] : li) {
          minweight = std::min(minweight, v.score());
        }
        for (auto i = li.begin();;) {
          auto next = std::next(i);
          if (next == li.end()) {
            break;
          }
          int from = i->first + 1;
          int to = next->first;
          if (to - from > 0) {
            float weight =
                std::min(i->second.score(), next->second.score()) - minweight;
            weight = 1.0f / std::min(std::exp(weight * 4), 1e9f);
            weight *= to - from;
            list.emplace_back(weight, from, to);
            sum += weight;
          }
          i = next;
        }
        if (list.size() > 0 && sum > 0.0f) {
          float val = std::uniform_real_distribution<float>(0.0f, sum)(rng);
          for (auto& [weight, from, to] : list) {
            val -= weight;
            if (val <= 0) {
              int k = std::uniform_int_distribution<int>(from, to - 1)(rng);
              eval(k);
              if (expands > 0) {
                expands -= expandNear(k);
              }
              break;
            }
          }
        }
      }
      if (best) {
        expandNear(best);
      }
      std::vector<std::tuple<float, int>> sorted;
      for (auto& [k, v] : li) {
        sorted.emplace_back(v.score(), k);
      }
      std::sort(sorted.begin(), sorted.end());
      for (size_t i = 0; i != sorted.size() && i < 10; ++i) {
        int k = std::get<1>(sorted[i]);
        if (li[k].n < 8) {
          eval(k);
        }
      }
    }
    foundBatchSize_ = best;
    hasFoundBatchSize_ = true;

    for (auto& [k, v] : li) {
      fmt::printf(
          "Batch size %d, evals %d latency %fms throughput %g score %g\n", k,
          v.n, v.latency / v.n, v.throughput / v.n, v.score());
    }

    fmt::printf("Found best batch size of %d with evals %d latency %fms "
                "throughput %g score %g\n",
                best, li[best].n, li[best].latency / li[best].n,
                li[best].throughput / li[best].n, li[best].score());
    return best;
  }

  bool isCuda() const {
    return device_.is_cuda();
  }

  torch::Device device() const {
    return device_;
  }

  int64_t bufferNumSample() const {
    return replayBuffer_.numSample();
  }

  int64_t bufferNumAdd() const {
    return replayBuffer_.numAdd();
  }

  void setIsTournamentOpponent(bool mode) {
    isTournamentOpponent_ = mode;
  }
  bool isTournamentOpponent() const {
    return isTournamentOpponent_;
  }
  void setDontRequestModelUpdates(bool v) {
    dontRequestModelUpdates_ = v;
  }

  bool wantsTournamentResult() {
    return client_ ? client_->wantsTournamentResult() : false;
  }

  std::string_view getTournamentModelId() {
    if (client_) {
      return client_->getModelId();
    } else {
      return "dev";
    }
  }

  void result(float reward,
              std::unordered_map<std::string_view, float> models) {
    if (client_ && isTournamentOpponent_ && !dontRequestModelUpdates_) {
      client_->sendResult(reward, std::move(models));
    }
  }

 private:
  const std::string jitModel_;
  torch::Device device_;
  torch::ScalarType dtype_;

  PriorityMutex* modelMutex_;
  std::shared_ptr<TorchJitModel> model_;
  std::shared_ptr<tube::DataChannel> actChannel_;
  std::shared_ptr<tube::DataChannel> trainChannel_;
  std::vector<std::thread> threads_;
  std::atomic_bool terminate_{false};

  ReplayBuffer replayBuffer_;

  std::atomic_size_t nextActIndex_{0};
  std::optional<distributed::Server> server_;
  std::optional<distributed::Client> client_;
  std::thread modelUpdateThread;
  bool isTournamentOpponent_ = false;
  bool dontRequestModelUpdates_ = false;

  std::atomic<bool> hasFoundBatchSize_ = false;
  std::atomic<int> foundBatchSize_ = 0;
};

ModelManager::ModelManager() {
}

ModelManager::ModelManager(int actBatchsize,
                           const std::string& device,
                           int replayCapacity,
                           int seed,
                           const std::string& jitModel,
                           int trainChannelTimeoutMs,
                           int trainChannelNumSlots) {
  impl = std::make_unique<ModelManagerImpl>(
      actBatchsize, device, replayCapacity, seed, jitModel,
      trainChannelTimeoutMs, trainChannelNumSlots);
}

ModelManager::~ModelManager() {
}

std::shared_ptr<tube::DataChannel> ModelManager::getTrainChannel() {
  return impl->getTrainChannel();
}

std::shared_ptr<tube::DataChannel> ModelManager::getActChannel() {
  return impl->getActChannel();
}

void ModelManager::updateModel(
    const std::unordered_map<std::string, at::Tensor>& stateDict) {
  return impl->updateModel(stateDict);
}

int ModelManager::bufferSize() const {
  return impl->bufferSize();
}

bool ModelManager::bufferFull() const {
  return impl->bufferFull();
}

std::unordered_map<std::string, at::Tensor> ModelManager::sample(
    int sampleSize) {
  return impl->sample(sampleSize);
}

void ModelManager::start() {
  return impl->start();
}

void ModelManager::testAct() {
  return impl->testAct();
}

void ModelManager::setIsTournamentOpponent(bool mode) {
  return impl->setIsTournamentOpponent(mode);
}

void ModelManager::addTournamentModel(
    std::string id,
    const std::unordered_map<std::string, at::Tensor>& stateDict) {
  return impl->addTournamentModel(std::move(id), stateDict);
}

void ModelManager::setDontRequestModelUpdates(bool v) {
  return impl->setDontRequestModelUpdates(v);
}

void ModelManager::startServer(std::string serverListenEndpoint) {
  return impl->startServer(serverListenEndpoint);
}

void ModelManager::startClient(std::string serverConnectHostname) {
  return impl->startClient(serverConnectHostname);
}

void ModelManager::startReplayBufferServer(std::string endpoint) {
  return impl->startReplayBufferServer(endpoint);
}

void ModelManager::startReplayBufferClient(std::string endpoint) {
  return impl->startReplayBufferClient(endpoint);
}

SampleResult ModelManager::remoteSample(int sampleSize) {
  return impl->remoteSample(sampleSize);
}

bool ModelManager::isCuda() const {
  return impl->isCuda();
}

torch::Device ModelManager::device() const {
  return impl->device();
}

void ModelManager::batchAct(at::Tensor input,
                            at::Tensor v,
                            at::Tensor pi,
                            at::Tensor rnnState,
                            at::Tensor* rnnStateOut) {
  return impl->batchAct(std::move(input), std::move(v), std::move(pi),
                        std::move(rnnState), rnnStateOut);
}

std::string_view ModelManager::getTournamentModelId() {
  return impl->getTournamentModelId();
}

void ModelManager::result(float reward,
                          std::unordered_map<std::string_view, float> models) {
  return impl->result(reward, std::move(models));
}

int ModelManager::findBatchSize(at::Tensor input, at::Tensor rnnState) {
  return impl->findBatchSize(input, rnnState);
}

int64_t ModelManager::bufferNumSample() const {
  return impl->bufferNumSample();
}

int64_t ModelManager::bufferNumAdd() const {
  return impl->bufferNumAdd();
}

bool ModelManager::isTournamentOpponent() const {
  return impl->isTournamentOpponent();
}

bool ModelManager::wantsTournamentResult() {
  return impl->wantsTournamentResult();
}

}  // namespace core
