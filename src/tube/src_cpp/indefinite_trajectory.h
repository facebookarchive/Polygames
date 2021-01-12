/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "data_block.h"
#include <deque>
#include <torch/extension.h>

namespace tube {

class IndefiniteTrajectory {
 public:
  IndefiniteTrajectory(const std::string& name,
                       int blockLen,
                       const std::vector<int64_t>& sizes,
                       torch::ScalarType dtype)
      : name(name)
      , blockLen(blockLen)
      , dtype(dtype)
      , sizes(sizes)
      , buffer(std::make_shared<DataBlock>(name, sizes, dtype))
      , trajectory(std::make_shared<DataBlock>(
            name, utils::pushLeft(blockLen, sizes), dtype)) {
  }

  torch::Tensor& getBuffer() {
    return buffer->data;
  }

  const torch::Tensor& getBuffer() const {
    return buffer->data;
  }

  int pushBufferToTrajectory() {
    // user might accidentally change the tensor
    // TODO: better ways to prevent it?
    assert(buffer->dtype() == dtype);
    assert(buffer->sizes() == sizes);
    trajectory_.push_back(buffer->data.clone());

    return len();
  }

  bool prepareForSend() {
    if ((int)trajectory_.size() < blockLen) {
      return false;
    }
    for (int i = 0; i < blockLen; ++i) {
      trajectory->data[i].copy_(trajectory_.front());
      trajectory_.pop_front();
    }
    return true;
  }

  int len() {
    return (int)trajectory_.size();
  }

  const std::string name;
  const int blockLen;
  const torch::ScalarType dtype;
  const std::vector<int64_t> sizes;

  std::shared_ptr<DataBlock> buffer;
  std::shared_ptr<DataBlock> trajectory;

 private:
  std::deque<torch::Tensor> trajectory_;
};
}  // namespace tube
