/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>

#include "replay_buffer.h"

using tube::ReplayBuffer;

std::vector<int64_t> sampleKfromN(int k, int n, std::mt19937& rng) {
  std::unordered_set<int> samples;
  while ((int)samples.size() < k && (int)samples.size() < n) {
    int s = rng() % n;
    samples.insert(s);
  }
  std::vector<int64_t> ret;
  for (int s : samples) {
    ret.push_back((int64_t)s);
  }
  return ret;
}

void ReplayBuffer::add(std::unordered_map<std::string, torch::Tensor> input) {
  std::unique_lock<std::mutex> lk(mBuf_);
  if (size_ == 0) {
    for (auto& it : input) {
      auto t = it.second.sizes();
      std::vector<int64_t> sizes(t.begin(), t.end());
      sizes[0] = capacity;
      buffer_.insert({it.first, torch::zeros(sizes, it.second.dtype())});
    }
  }
  assert(input.size() == buffer_.size());

  // now perform the copying
  torch::Tensor tensorIndices = getNextIndices(input);
  numAdd_ += tensorIndices.size(0);
  for (auto& b : buffer_) {
    const std::string name = b.first;
    auto in = input.find(name);
    assert(in != input.end());
    // Explicit size checking
    auto s1 = b.second.sizes();
    auto s2 = in->second.sizes();
    assert(s1.size() == s2.size());
    for (size_t i = 1; i < s1.size(); i++) {
      assert(s1[i] == s2[i]);
    }
    b.second.index_copy_(0, tensorIndices, in->second);
  }
}

std::unordered_map<std::string, torch::Tensor> ReplayBuffer::sample(
    int sampleSize) {
  std::unique_lock<std::mutex> lk(mBuf_);
  numSample_ += sampleSize;
  assert(sampleSize <= size_);
  auto sampleIndices = torch::tensor(sampleKfromN(sampleSize, size_, rng_));

  std::unordered_map<std::string, torch::Tensor> result;
  for (auto& b : buffer_) {
    const std::string& name = b.first;
    result.insert({name, torch::index_select(b.second, 0, sampleIndices)});
  }
  return result;
}

ReplayBuffer::SerializableState ReplayBuffer::toState() {
  std::unique_lock<std::mutex> lk(mBuf_);
  ReplayBuffer::SerializableState state;
  state.capacity = capacity;
  state.size = size_;
  state.nextIdx = nextIdx_;
  std::ostringstream oss;
  oss << rng_;
  state.rngState = oss.str();
  state.buffer = buffer_;
  return state;
}

void ReplayBuffer::initFromState(const SerializableState& state) {
  std::unique_lock<std::mutex> lk(mBuf_);
  if (state.capacity != capacity) {
    std::ostringstream oss;
    oss << "Attempt to initialize a buffer of capacity " << capacity
        << " from buffer state of capacity " << state.capacity;
    throw std::runtime_error(oss.str());
  }
  size_ = state.size;
  nextIdx_ = state.nextIdx;
  std::istringstream iss(state.rngState);
  iss >> rng_;
  buffer_ = state.buffer;
}

torch::Tensor ReplayBuffer::getNextIndices(
    std::unordered_map<std::string, torch::Tensor>& input) {
  int inSize = -1;
  // these are indices of replay buffer that we will copy into
  std::vector<int64_t> copyIndices;
  for (auto& it : input) {
    if (inSize < 0) {
      inSize = it.second.size(0);
      if (inSize > capacity) {
        std::cerr << "inSize=" << inSize << ", capacity=" << capacity
                  << std::endl;
        assert(inSize <= capacity);
      }
      for (int i = 0; i < inSize; i++) {
        copyIndices.push_back((i + nextIdx_) % capacity);
      }
      if (size_ < capacity) {
        size_ = size_ + inSize > capacity ? capacity : size_ + inSize;
      }
      nextIdx_ = (nextIdx_ + inSize) % capacity;
    } else {
      // all the names should have the same input size
      assert(inSize == it.second.size(0));
    }
  }
  return torch::tensor(copyIndices);
}
