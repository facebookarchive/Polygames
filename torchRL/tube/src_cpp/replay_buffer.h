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

class ReplayBuffer {
 public:
  struct SerializableState {
    int capacity;
    int size;
    int nextIdx;
    std::string rngState;
    std::unordered_map<std::string, torch::Tensor> buffer;
  };

  /**
   * Helper to track exponential moving average.
   */
  struct ExpMovingAverage {
    float alpha = 0.05f;  // Weight assigned to most recent point, 0 = all
                          // points weighted equally
    float runningMean = 0.f;
    float runningDenominator = 0.f;

    void observe(float data) {
      runningDenominator = (1.f - alpha) * runningDenominator + 1.f;
      runningMean += (1.f / runningDenominator) * (data - runningMean);
    }
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

  float averageEpisodeDuration() const {
    return epDurationTracker_.runningMean;
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
  // keeps track of average duration of episodes
  ExpMovingAverage epDurationTracker_;

  // how many items in buffer
  int size_ = 0;
  // the next index to write to
  int nextIdx_ = 0;
  int64_t numAdd_ = 0;
  int64_t numSample_ = 0;

  std::mt19937 rng_;
};
}  // namespace tube
