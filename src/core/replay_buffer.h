/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/torch.h>
#include <mutex>
#include <random>
#include <vector>
#include <unordered_map>
#include <string>

namespace core {

class ReplayBuffer {
 public:
  ReplayBuffer(int capacity, int seed)
      : capacity(capacity)
      , buffer(capacity) {
    rng_.seed(seed);
  }

  ~ReplayBuffer() {
    if (!sampleThreads.empty()) {
      std::unique_lock l(mut);
      sampleThreadDie = true;
      cv.notify_all();
      l.unlock();
      for (auto& v : sampleThreads) {
        v.join();
      }
    }
  }

  struct Key {
    std::string name;
    std::vector<int64_t> shape;
    caffe2::TypeMeta dtype;
  };

  struct BufferEntry {
    size_t datasize;
    std::vector<char> data;
  };

  /*
   * add to the circular array the elements of input
   * updates input if size is greater than capacity
   * initializes if the buffer is empty
   */
  void add(std::unordered_map<std::string, torch::Tensor> input);

  template <typename T> static std::string ss(T&& sizes) {
    std::string r = "[";
    for (int64_t v : sizes) {
      if (r != "[")
        r += ", ";
      r += std::to_string(v);
    }
    return r + "]";
  }

  /*
   * sample sampleSize elements from the replayBuffer
   */
  std::unordered_map<std::string, torch::Tensor> sampleImpl(int sampleSize);

  std::mutex mut;
  std::condition_variable cv;
  std::condition_variable cv2;
  std::deque<std::unordered_map<std::string, torch::Tensor>> results;
  int resultsSampleSize = 0;
  bool sampleThreadDie = false;

  std::vector<std::thread> sampleThreads;

  std::unordered_map<std::string, torch::Tensor> sample(int sampleSize);

  int size() const {
    return (int)std::min((int64_t)numAdd_, (int64_t)capacity);
  }

  bool full() const {
    return size() == capacity;
  }

  int64_t numAdd() const {
    return numAdd_;
  }

  int64_t numSample() const {
    return numSample_;
  }

  const int capacity;

 private:
  std::vector<std::atomic<BufferEntry*>> buffer;

  size_t sampleOrderIndex = 0;
  std::vector<size_t> sampleOrder;

  std::vector<Key> keys;
  std::mutex keyMutex;
  std::atomic<bool> hasKeys = false;
  std::mutex sampleMutex;
  int64_t prevSampleNumAdd_ = 0;
  std::atomic_int64_t numAdd_ = 0;
  std::atomic_int64_t numSample_ = 0;

  std::mt19937 rng_;
};

}  // namespace tube
