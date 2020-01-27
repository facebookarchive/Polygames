/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "data_channel.h"
#include "distributed.h"
#include "replay_buffer.h"

#include <fmt/printf.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <fmt/printf.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace tube {

using TorchJitModel = torch::jit::script::Module;

inline std::atomic_int threadIdCounter{0};
inline thread_local int threadId = ++threadIdCounter;

inline std::unordered_map<std::string, torch::Tensor> convertIValueToMap(
    const torch::IValue& value) {
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

    tensor = tensor.to(torch::kCPU).detach();
    // if (device != nullptr) {
    //   tensor = tensor.to(device);
    // }
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

inline std::vector<PriorityMutex> mModels_(30);

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

}  // namespace

class ChannelAssembler {
 public:
  ChannelAssembler(int actBatchsize,
                   int numActChannel,
                   const std::vector<std::string>& actDevices,
                   int replayCapacity,
                   int seed,
                   const std::string& jitModel,
                   int trainChannelTimeoutMs,
                   int trainChannelNumSlots)
      : jitModel_(jitModel)
      //, mModels_(numActChannel)
      , replayBuffer_(replayCapacity, seed) {
    trainChannel_ = std::make_shared<DataChannel>(
        "train", trainChannelNumSlots, trainChannelTimeoutMs);
    assert((int)actDevices.size() >= numActChannel);
    for (int i = 0; i < numActChannel; ++i) {
      std::string device =
          (size_t)i >= actDevices.size() && actDevices.size() > 0
              ? actDevices.back()
              : actDevices.at(i);
      auto deviceString = torch::Device(device);
      actDevices_.push_back(deviceString);

#ifdef PYTORCH12
      models_.push_back(std::make_shared<TorchJitModel>(torch::jit::load(jitModel_)));
#else
      models_.push_back(torch::jit::load(jitModel_));
#endif
      models_.back()->to(device);
      models_.back()->eval();

      auto channel = std::make_shared<DataChannel>(
          std::string("act") + std::to_string(i), actBatchsize, -1);
      actChannels_.push_back(std::move(channel));
    }
  }

  ~ChannelAssembler() {
    terminate_ = true;
    for (auto& v : actChannels_) {
      v->terminate();
    }
    trainChannel_->terminate();
    for (auto& v : threads_) {
      v.join();
    }
    if (modelUpdateThread.joinable()) {
      modelUpdateThread.join();
    }
  }

  void startServer(std::string serverListenEndpoint) {
    if (!actChannels_.empty()) {
      throw std::runtime_error(
          "Server should be run with no actors! (--num_game 0)");
    }
    server_.emplace();
    server_->onTrainData = [this](const void* data, size_t len) {
      std::unordered_map<std::string, torch::Tensor> batch;
      std::string_view view((const char*)data, len);
      StreamReadBuffer<decltype(view)> buf(view);
      while (true) {
        auto k = buf.read();
        auto v = buf.read();
        if (v == std::string_view{}) {
          break;
        }
        std::string str(v.data(), v.size());
        std::istringstream is(str);
        torch::Tensor t;
        torch::load(t, is);
        batch[std::string(k)] = t;
      }
      replayBuffer_.add(std::move(batch));
    };
    server_->start(serverListenEndpoint);
    fmt::printf("Listening on %s\n", serverListenEndpoint);
  }

  void startClient(std::string serverConnectHostname) {
    client_.emplace();
    client_->onUpdateModel =
        [this](std::string_view id,
               std::unordered_map<std::string, torch::Tensor> dict) {
          if (!dontRequestModelUpdates_) {
            fmt::printf("onUpdateModel '%s'\n", id);
            updateModel(dict);
          }
        };
    client_->connect(serverConnectHostname);
    fmt::printf("Connected to %s\n", serverConnectHostname);

    modelUpdateThread = std::thread([this]() {
      while (!terminate_ && !trainChannel_->terminated()) {
        if (!dontRequestModelUpdates_) {
          client_->requestModel(isTournamentOpponent_);
        }
        for (int i = 0; i != 20 && !terminate_ && !trainChannel_->terminated();
             ++i) {
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
      }
    });
  }

  std::shared_ptr<DataChannel> getTrainChannel() {
    return trainChannel_;
  }

  std::vector<std::shared_ptr<DataChannel>> getActChannels() {
    return actChannels_;
  }

  std::unordered_map<std::string, torch::Tensor> cloneStateDict(
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    torch::NoGradGuard ng;
    std::unordered_map<std::string, torch::Tensor> r;
    for (auto& [name, tensor] : stateDict) {
      r[name] = tensor.clone().detach().cpu();
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

  void loadModelStateDict(
      TorchJitModel& model,
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    for (auto& [name, tensor] : stateDict) {
      const char* ptr = name.c_str();
      auto* currentModule = &model;
      std::string memberNameString;
      const char* memberNamePtr = ptr;
      while (*ptr) {
        if (*ptr == '.') {
          memberNameString.assign(memberNamePtr, ptr - memberNamePtr);
          auto subModule = currentModule->find_module(memberNameString);
          if (!subModule) {
            fmt::printf("copyModelStateDict: Unknown state dict entry '%s' -- could "
                   "not find module '%s'\n",
                   name,
                   memberNameString);
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

#ifdef PYTORCH12
      if (c10::optional<torch::jit::script::Slot> p = currentModule->find_parameter(memberNameString); p) {
        p->value().toTensor().copy_(tensor);
      } else if (c10::optional<torch::jit::script::Slot> b = currentModule->find_buffer(memberNameString); b) {
        b->value().toTensor().copy_(tensor);
      } else {
#else
      if (auto* p = currentModule->find_parameter(memberNameString); p) {
        p->value().toTensor().copy_(tensor);
      } else if (auto* b = currentModule->find_buffer(memberNameString); b) {
        b->value().toTensor().copy_(tensor);
      } else {
#endif

        fmt::printf("copyModelStateDict: Unknown state dict entry '%s' -- could not "
               "find parameter/buffer '%s'\n",
               name,
               memberNameString);
        std::abort();
      }
    }
    model.eval();
  }

  void updateModel(
      const std::unordered_map<std::string, torch::Tensor>& stateDict) {
    torch::NoGradGuard ng;
    fmt::printf(" -- UPDATE MODEL --\n");
    if (server_) {
      server_->updateModel("dev", cloneStateDict(stateDict));
    }
    for (size_t i = 0; i < models_.size(); ++i) {
      PriorityMutex::setThreadPriority(-9);
      std::lock_guard<PriorityMutex> lk(mModels_[i]);
      loadModelStateDict(*models_[i], stateDict);
    }
  }

  int bufferSize() const {
    return replayBuffer_.size();
  }

  bool bufferFull() const {
    return replayBuffer_.full();
  }

  ReplayBuffer::SerializableState getBufferState() {
    return replayBuffer_.toState();
  }

  void setBufferState(const ReplayBuffer::SerializableState& state) {
    replayBuffer_.initFromState(state);
  }

  std::unordered_map<std::string, torch::Tensor> sample(int sampleSize) {
    return replayBuffer_.sample(sampleSize);
  }

  void start() {
    threads_.emplace_back(&ChannelAssembler::trainThread, this);

    for (int i = 0; i < (int)actChannels_.size(); ++i) {
      threads_.emplace_back(&ChannelAssembler::actThread, this, i);
    }
  }

  void trainThread() {
    torch::NoGradGuard ng;
    if (client_) {
      while (true) {
        auto batch = trainChannel_->getInput();
        if (terminate_ || trainChannel_->terminated()) {
          break;
        }
        std::vector<char> buf;
        StreamBuffer<decltype(buf)> obuf(buf);
        for (auto& v : batch) {
          obuf.write(v.first);
          std::ostringstream stream;
          torch::save(v.second, stream);
          obuf.write(stream.str());
        }
        client_->sendTrainData(buf.data(), buf.size());
        trainChannel_->setReply({});
      }
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

  void actThread(const int threadIdx) {
    torch::NoGradGuard ng;
    while (true) {
      auto batch = actChannels_[threadIdx]->getInput();
      if (terminate_ || actChannels_[threadIdx]->terminated()) {
        break;
      }
      // TODO[hengyuan]: temp hard code
      auto s = batch["s"].to(actDevices_[threadIdx]);
      std::vector<torch::jit::IValue> input;
      input.push_back(s);
      PriorityMutex::setThreadPriority(-1);
      std::unique_lock<PriorityMutex> lk(mModels_[threadIdx]);
      auto output = models_[threadIdx]->forward(input);
      lk.unlock();
      auto reply = convertIValueToMap(output);
      actChannels_[threadIdx]->setReply(reply);
    }
  }

  void testAct() {
    torch::NoGradGuard ng;
    std::vector<torch::jit::IValue> inputs;
    auto x = torch::ones({1, 6 * 7 * 2}, torch::kFloat32);
    inputs.push_back(x);
    auto y = models_[0]->forward(inputs);
    auto reply = convertIValueToMap(y);
    for (auto& name2tensor : reply) {
      std::cout << name2tensor.first << ": " << std::endl;
      std::cout << name2tensor.second << std::endl;
    }
  }

  std::string_view batchAct(torch::Tensor input,
                            torch::Tensor v,
                            torch::Tensor pi) {
    torch::NoGradGuard ng;
    size_t n = nextActIndex_++ % models_.size();
    std::vector<torch::jit::IValue> inp;
    PriorityMutex::setThreadPriority(threadId);
    inp.push_back(input.to(actDevices_[n]));
    std::unique_lock<PriorityMutex> lk(mModels_[n]);
    std::string_view id = getTournamentModelId();
    auto output = models_[n]->forward(inp);
    lk.unlock();
    auto reply = convertIValueToMap(output);
    v.copy_(reply["v"]);
    pi.copy_(reply["pi"]);
    return id;
  }

  int bufferNumSample() const {
    return replayBuffer_.numSample();
  }

  int bufferNumAdd() const {
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
  std::vector<torch::Device> actDevices_;

  // std::vector<PriorityMutex> mModels_;
  std::vector<std::shared_ptr<TorchJitModel>> models_;
  std::vector<std::shared_ptr<DataChannel>> actChannels_;
  std::shared_ptr<DataChannel> trainChannel_;
  std::vector<std::thread> threads_;
  std::atomic_bool terminate_{false};

  ReplayBuffer replayBuffer_;

  std::atomic_size_t nextActIndex_{0};
  std::optional<DistributedServer> server_;
  std::optional<DistributedClient> client_;
  std::thread modelUpdateThread;
  bool isTournamentOpponent_ = false;
  bool dontRequestModelUpdates_ = false;
  int clientModelVersion_ = -1;
};

}  // namespace tube
