/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "data_block.h"

namespace tube {

class EpisodicTrajectory {
 public:
  EpisodicTrajectory(const std::string& name,
                     // int blockLen,
                     const std::vector<int64_t>& sizes,
                     torch::ScalarType dtype)
      : name(name)
      // , blockLen(blockLen)
      , dtype(dtype)
      , sizes(sizes)
      , buffer(std::make_shared<DataBlock>(name, sizes, dtype)) {}

  int pushBack(torch::Tensor t) {
    assert(t.dtype() == dtype);
    assert(t.sizes() == sizes);
    trajectory_.push_back(t);
    return (int)trajectory_.size();
  }

  bool prepareForSend() {
    if (trajectory_.empty()) {
      return false;
    }

    buffer->data.copy_(trajectory_.back());
    // buffer->data = std::move(trajectory_.back());
    trajectory_.pop_back();
    return true;
  }

  int len() {
    return (int)trajectory_.size();
  }

  const std::string name;
  // const int blockLen;
  const torch::ScalarType dtype;
  const std::vector<int64_t> sizes;

  std::shared_ptr<DataBlock> buffer;

 private:
  std::vector<torch::Tensor> trajectory_;
};
}
