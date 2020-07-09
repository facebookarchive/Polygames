/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <random>
#include <vector>

#include "data_block.h"

namespace tube {

class ReplayBuffer2 {
 public:
  struct SerializableState {
    int capacity;
    int size;
    int nextIdx;
    std::string rngState;
    std::unordered_map<std::string, torch::Tensor> buffer;
  };

  /*
   * Convert replay buffer to a serializable state (thread-safe)
   */
  SerializableState toState() {
    return {};
  }

  /*
   * Restore replay buffer from a serializable state
   */
  void initFromState(const SerializableState& state) {
  }

  ReplayBuffer2(int capacity, int seed)
      : capacity(capacity)
      , buffer(capacity) {
    rng_.seed(seed);
  }

  ~ReplayBuffer2() {
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

  std::unordered_map<std::string, torch::Tensor> sample(int sampleSize) {
    // return sampleImpl(sampleSize);
    std::unique_lock l(mut);
    if (sampleThreads.empty()) {
      for (int i = 0; i != 8; ++i) {
        sampleThreads.emplace_back([this]() {
          std::unique_lock l(mut);
          while (true) {
            while (results.size() >= 8 || resultsSampleSize == 0) {
              cv.wait(l);
              if (sampleThreadDie) {
                return;
              }
            }
            l.unlock();
            auto tmp = sampleImpl(resultsSampleSize);
            l.lock();
            results.push_back(std::move(tmp));
            cv2.notify_all();
          }
        });
      }
    }
    resultsSampleSize = sampleSize;
    while (results.empty()) {
      cv.notify_all();
      cv2.wait(l);
    }
    auto r = std::move(results.front());
    results.pop_front();
    cv.notify_all();
    return r;
  }

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

using ReplayBuffer = ReplayBuffer2;

namespace old {
class ReplayBuffer {
 public:
  struct SerializableState {
    int capacity;
    int size;
    int nextIdx;
    std::string rngState;
    std::unordered_map<std::string, torch::Tensor> buffer;
  };

  ReplayBuffer(int capacity, int seed)
      : capacity(capacity) {
    rng_.seed(seed);
  }

  /*
   * add to the circular array the elements of input
   * updates input if size is greater than capacity
   * initializes if the buffer is empty
   */
  void add(std::unordered_map<std::string, torch::Tensor> input);

  /*
   * sample sampleSize elements from the replayBuffer
   */
  std::unordered_map<std::string, torch::Tensor> sample(int sampleSize);

  int size() const {
    return size_;
  }

  bool full() const {
    return size_ == capacity;
  }

  int64_t numAdd() const {
    return numAdd_;
  }

  int64_t numSample() const {
    return numSample_;
  }

  /*
   * Convert replay buffer to a serializable state (thread-safe)
   */
  SerializableState toState();

  /*
   * Restore replay buffer from a serializable state
   */
  void initFromState(const SerializableState& state);

  const int capacity;

 private:
  /*
   * given an input to the replay buffer, calculate the indices of buffer_ to
   * copy the input.
   * At the same time, we verify that the sizes of each tensor in input are the
   * same and that
   * the input is no larger than than the capacity of the buffer.
   */
  torch::Tensor getNextIndices(
      std::unordered_map<std::string, torch::Tensor>& input);

  // mutex for buffer_
  std::mutex mBuf_;
  // the actual circular replay buffer
  std::unordered_map<std::string, torch::Tensor> buffer_;
  // how many items in buffer
  int size_ = 0;
  // the next index to write to
  int nextIdx_ = 0;
  int64_t numAdd_ = 0;
  int64_t numSample_ = 0;

  std::mt19937 rng_;
};

}  // namespace old

// using ReplayBuffer = old::ReplayBuffer;

}  // namespace tube
