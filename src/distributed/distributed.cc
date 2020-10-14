/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "reqrepserver.h"

#include <fmt/printf.h>
#include <torch/torch.h>

#include <cstring>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace tube {

// This is not a cross platform serializer
struct Serializer {
  std::vector<char> buf;
  void write(const void* data, size_t len) {
    size_t offset = buf.size();
    if (buf.capacity() < offset + len) {
      buf.reserve(
          std::max(offset + len, std::max(buf.capacity() * 2, (size_t)16)));
    }
    buf.resize(offset + len);
    std::memcpy(buf.data() + offset, data, len);
  }
  template <typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  void write(T v) {
    write((void*)&v, sizeof(v));
  }

  void write(std::string_view str) {
    write(str.size());
    write(str.data(), str.size());
  }

  template <typename Key, typename Value, typename... A>
  void write(const std::unordered_map<Key, Value, A...>& v) {
    writeMap(v);
  }

  void write(torch::Tensor v) {
    std::ostringstream os;
    torch::save(v, os);
    write(os.str());
  }

  template <typename Key, typename Value, typename... A>
  void writeMap(const std::unordered_map<Key, Value, A...>& v) {
    write(v.size());
    for (auto& x : v) {
      write(x.first);
      write(x.second);
    }
  }

  void clear() {
    buf.clear();
  }
  const char* data() const {
    return buf.data();
  }
  size_t size() const {
    return buf.size();
  }
};
struct Deserializer {
  std::string_view buf;
  Deserializer() = default;
  Deserializer(std::string_view buf)
      : buf(buf) {
  }
  Deserializer(const void* data, size_t len)
      : buf((const char*)data, len) {
  }
  void consume(size_t len) {
    buf = {buf.data() + len, buf.size() - len};
  }
  std::string_view readString() {
    size_t len = read<size_t>();
    if (buf.size() < len) {
      len = buf.size();
    }
    const char* data = buf.data();
    consume(len);
    return {data, len};
  }
  template <typename T, std::enable_if_t<std::is_trivial_v<T>>* = nullptr>
  void read(T& r) {
    if (buf.size() < sizeof(T)) {
      consume(buf.size());
      r = {};
      return;
    }
    std::memcpy(&r, buf.data(), sizeof(T));
    consume(sizeof(T));
  }
  void read(std::string_view& r) {
    r = readString();
  }
  void read(std::string& r) {
    r = readString();
  }
  template <typename Key, typename Value, typename... A>
  void read(std::unordered_map<Key, Value, A...>& v) {
    readMap(v);
  }

  void read(torch::Tensor& v) {
    auto s = read();
    std::string str(s.data(), s.size());
    std::istringstream is(str);
    torch::load(v, is);
  }

  template <typename T> T read() {
    T r;
    read(r);
    return r;
  }
  std::string_view read() {
    return readString();
  }
  template <typename Key, typename Value, typename... A>
  void readMap(std::unordered_map<Key, Value, A...>& v) {
    v.clear();
    size_t n = read<size_t>();
    for (; n; --n) {
      auto k = read<Key>();
      v.emplace(std::move(k), read<Value>());
    }
  }

  bool empty() {
    return buf.empty();
  }
};

enum class MessageID {
  MsgNull = 0,
  MsgRequestModel = 1,
  MsgReplyModel = 2,
  MsgRequestStateDict = 3,
  MsgReplyStateDict = 4,
  MsgTrainData = 5,
  MsgGameResult = 6,
};

class DistributedServer {

  std::optional<cpid::ReqRepServer> server;

  std::minstd_rand rng{std::random_device()()};

  std::string_view sampleModelId() {
    if (models.empty()) {
      return "dev";
    }
    std::vector<std::pair<double, std::string_view>> scores;
    float max = 0.0;
    for (auto& [id, m] : models) {
      max = std::max(max, m.rating);
    }
    for (auto& [id, m] : models) {
      double v = std::exp((m.rating - max) / 400.0);
      scores.emplace_back(v, id);
    }
    std::sort(scores.begin(), scores.end(), std::greater<>());
    if (scores.size() > 24) {
      scores.resize(24);
    }
    double sum = 0.0;
    for (auto& [v, k] : scores) {
      sum += v;
    }
    for (auto& [v, k] : scores) {
      v /= sum;
    }
    double val =
        std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    for (auto& [v, id] : scores) {
      val -= v;
      if (val <= 0) {
        return id;
      }
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
      float k = 30;
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

    auto now = std::chrono::steady_clock::now();
    if (now - lastRatingPrint >= std::chrono::seconds(120)) {
      lastRatingPrint = now;
      std::vector<std::pair<float, std::string_view>> sorted;
      for (auto& [id, m] : models) {
        sorted.emplace_back(m.rating, id);
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
      if (sorted.size() > 10) {
        sorted.resize(10);
      }
      std::string str;
      int rank = 1;
      for (auto& [rating, id] : sorted) {
        str += fmt::sprintf("%d. %g %s\n",
                            rank,
                            rating,
                            id);
        ++rank;
      }
      if (devrank > 10) {
        str += fmt::sprintf("%d. %g %s\n",
                            devrank,
                            devrating,
                            "dev");
      }
      fmt::printf("Top 10:\n%s", str);
    }
  }

  void onData(const void* data,
              size_t len,
              std::function<void(void const* buf, size_t len)> reply) {

    Serializer s;

    Deserializer d(data, len);
    MessageID id = (MessageID)d.read<unsigned char>();

    if (id == MessageID::MsgRequestModel) {
      std::unique_lock l(mut);
      bool wantsNewModelId = d.read<bool>();
      std::string_view modelId = d.read();
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
      s.write<char>((int)MessageID::MsgReplyModel);
      s.write(modelId);
      s.write<int>(version);
    } else if (id == MessageID::MsgRequestStateDict) {
      std::unique_lock l(mut);
      std::string_view modelId = d.read();
      s.write<char>((int)MessageID::MsgReplyStateDict);
      auto i = models.find(modelId);
      if (i == models.end()) {
        s.write<bool>(false);
      } else {
        s.write<bool>(true);
        s.write(i->second.stateDict);
      }
    } else if (id == MessageID::MsgTrainData) {
      auto data = d.readString();
      onTrainData(data.data(), data.size());
    } else if (id == MessageID::MsgGameResult) {
      std::lock_guard l(mut);
      std::unordered_map<std::string_view, float> models;
      while (!d.empty()) {
        float reward = d.read<float>();
        d.readMap(models);
        for (auto& [id, ratio] : models) {
          addResult(id, ratio, reward);
        }
      }
    }

    if (s.size() == 0) {
      s.write<char>((int)MessageID::MsgNull);
    }
    reply(s.data(), s.size());
  }

  struct ModelInfo {
    std::string id;
    int version = 0;
    float rating = 0.0f;
    std::unordered_map<std::string, torch::Tensor> stateDict;
  };

  std::mutex mut;
  std::unordered_map<std::string_view, ModelInfo> models;

 public:
  std::function<void(const void* data, size_t len)> onTrainData;

  void start(std::string endpoint) {
    server.emplace(std::bind(&DistributedServer::onData,
                             this,
                             std::placeholders::_1,
                             std::placeholders::_2,
                             std::placeholders::_3),
                   4,
                   endpoint);
  }

  void updateModel(const std::string& id,
                   std::unordered_map<std::string, torch::Tensor> stateDict) {
    std::unique_lock l(mut);
    auto i = models.try_emplace(id);
    if (i.second) {
      i.first->second.id = id;
      (std::string_view&)i.first->first = i.first->second.id;
      auto idev = models.find("dev");
      if (idev != models.end()) {
        i.first->second.rating = idev->second.rating - 100;
      }
    }
    auto& m = i.first->second;
    m.stateDict = std::move(stateDict);
    ++m.version;
  }
};

class DistributedClient {

  std::optional<cpid::ReqRepClient> client;

  std::mutex mut;
  std::unordered_set<std::string> allModelIds;
  std::string_view currentModelId = *allModelIds.emplace("dev").first;
  int currentModelVersion = -1;
  int gamesDoneWithCurrentModel = 0;
  bool wantsNewModelId = false;

  void send(Serializer& s) {
    recv(client->request(std::move(s.buf)).get());
  }

  void recv(std::vector<char> buf) {
    Deserializer d(buf.data(), buf.size());
    MessageID id = (MessageID)d.read<unsigned char>();
    if (id == MessageID::MsgReplyModel) {
      std::string_view newId = d.readString();
      std::unique_lock l(mut);
      if (currentModelId != newId) {
        currentModelId = *allModelIds.emplace(newId).first;
        currentModelVersion = -1;
        gamesDoneWithCurrentModel = 0;
      }
      int version = d.read<int>();
      if (version != currentModelVersion) {
        currentModelVersion = version;
        l.unlock();
        requestModelStateDict();
      } else {
        l.unlock();
      }
    } else if (id == MessageID::MsgReplyStateDict) {
      std::unordered_map<std::string, torch::Tensor> stateDict;
      bool success = d.read<bool>();
      if (!success) {
        std::lock_guard l(mut);
        currentModelId = "dev";
        currentModelVersion = -1;
      } else {
        std::unique_lock l(mut);
        auto id = currentModelId;
        l.unlock();
        d.read(stateDict);
        onUpdateModel(id, stateDict);
      }
    }
  }

  std::vector<std::pair<float, std::unordered_map<std::string_view, float>>>
      resultQueue;

  void requestModelStateDict() {
    Serializer s;
    s.write<char>((int)MessageID::MsgRequestStateDict);
    std::unique_lock l(mut);
    s.write(currentModelId);
    l.unlock();
    send(s);
  }

 public:
  std::function<void(
      std::string_view, std::unordered_map<std::string, torch::Tensor>)>
      onUpdateModel;

  void requestModel(bool isTournamentOpponent) {
    std::unique_lock l(mut);
    Serializer s;
    if (!resultQueue.empty()) {
      s.write<char>((int)MessageID::MsgGameResult);
      for (auto& v : resultQueue) {
        s.write<float>(v.first);
        s.writeMap(v.second);
      }
      resultQueue.clear();
      l.unlock();
      send(s);
      s.clear();
      l.lock();
    }

    s.write<char>((int)MessageID::MsgRequestModel);
    s.write<bool>(isTournamentOpponent ? std::exchange(wantsNewModelId, false)
                                       : false);
    s.write(currentModelId);
    l.unlock();
    send(s);
  }

  void connect(std::string host) {
    client.emplace(50, std::vector<std::string>{host});
    requestModel(false);
  }

  void sendTrainData(const void* data, size_t len) {
    Serializer s;
    s.write<char>((int)MessageID::MsgTrainData);
    s.write(std::string_view((const char*)data, len));
    send(s);
  }

  void sendResult(float reward,
                  std::unordered_map<std::string_view, float> models) {
    std::unique_lock l(mut);
    auto i = models.find(currentModelId);
    if (i != models.end()) {
      if (i->second >= 0.9f) {
        ++gamesDoneWithCurrentModel;
        if (gamesDoneWithCurrentModel >= 8) {
          wantsNewModelId = true;
        }
      }
    }
    resultQueue.emplace_back(reward, std::move(models));
  }

  std::string_view getModelId() {
    std::unique_lock l(mut);
    return currentModelId;
  }
};

}  // namespace tube
