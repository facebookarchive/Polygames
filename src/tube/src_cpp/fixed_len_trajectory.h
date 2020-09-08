/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "data_block.h"

namespace tube {

class FixedLengthTrajectory {
 public:
  FixedLengthTrajectory(const std::string& name,
                        int len,
                        const std::vector<int64_t>& sizes,
                        torch::ScalarType dtype)
      : name(name)
      , len(len)
      , sizes(sizes)
      , dtype(dtype)
      , buffer(std::make_shared<DataBlock>(name, sizes, dtype))
      , trajectory(std::make_shared<DataBlock>(
            name, utils::pushLeft(len, sizes), dtype))
      , nextSlot_(0) {
  }

  torch::Tensor& getBuffer() {
    return buffer->data;
  }

  const torch::Tensor& getBuffer() const {
    return buffer->data;
  }

  int pushBufferToTrajectory() {
    int pushedSlot = nextSlot_;

    // user might accidentally change the tensor
    // TODO: better ways to prevent it?
    assert(buffer->dtype() == dtype);
    assert(buffer->sizes() == sizes);
    trajectory->data[pushedSlot].copy_(buffer->data);

    nextSlot_ = (nextSlot_ + 1) % len;
    return pushedSlot;
  }

  const std::string name;
  const int len;
  const std::vector<int64_t> sizes;
  const torch::ScalarType dtype;

  // TODO: not good to be public, but need to be shared with dispatcher anyway
  std::shared_ptr<DataBlock> buffer;
  std::shared_ptr<DataBlock> trajectory;

 private:
  int nextSlot_;
};
}  // namespace tube
