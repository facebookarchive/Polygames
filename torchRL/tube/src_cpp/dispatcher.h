/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "data_block.h"
#include "data_channel.h"

namespace tube {

class Dispatcher {
 public:

  static constexpr int DISPATCH_ERR_DC_TERM = -2;
  static constexpr int DISPATCH_ERR_NO_SLOT = -1;
  static constexpr int DISPATCH_NOERR = 0;

  Dispatcher(std::shared_ptr<DataChannel> dc)
      : dc_(std::move(dc)) {
  }

  void addDataBlocks(const std::vector<std::shared_ptr<DataBlock>>& send,
                     const std::vector<std::shared_ptr<DataBlock>>& reply) {
    for (auto b : send) {
      auto ret = sendTensors_.insert({b->name, b->data});
      if (!ret.second) {
        std::cout << "Error: duplicated sendkey for dispatcher, "
                  << "key=" << b->name << ", DataChannel=" << dc_->name;
        assert(false);
      }
    }

    for (auto b : reply) {
      auto ret = replyTensors_.insert({b->name, b->data});
      if (!ret.second) {
        std::cout << "Error: duplicated replykey for dispatcher, "
                  << "key=" << b->name << ", DataChannel=" << dc_->name;
        assert(false);
      }
    }
    dc_->createOrCheckBuffers(send, reply);
  }

  // send data and get reply
  int dispatch() {
    int slot = -1;
    if (dc_->terminated()) {
      return DISPATCH_ERR_DC_TERM;
    }
    std::unordered_map<std::string, torch::Tensor> sendBuffers =
        dc_->getSlot(&slot);
    if (slot == -1) {
      return DISPATCH_ERR_NO_SLOT;
    }
    assert(slot >= 0 && slot < dc_->batchsize);
    utils::copyTensors(sendTensors_, sendBuffers);

    dc_->markSlotFilled(slot);

    std::unordered_map<std::string, torch::Tensor> replyBuffers =
        dc_->getReply(slot);
    utils::copyTensors(replyBuffers, replyTensors_);

    dc_->releaseSlot(slot);
    return DISPATCH_NOERR;
  }

  // send data and discard the reply without waiting for it
  int dispatchNoReply() {
    int slot = -1;
    if (dc_->terminated()) {
      return DISPATCH_ERR_DC_TERM;
    }
    std::unordered_map<std::string, torch::Tensor> sendBuffers =
        dc_->getSlot(&slot);
    if (slot == -1) {
      return DISPATCH_ERR_NO_SLOT;
    }
    assert(slot >= 0 && slot < dc_->batchsize);
    utils::copyTensors(sendTensors_, sendBuffers);

    dc_->markSlotFilledAutoRelease(slot);
    return DISPATCH_NOERR;
  }

  void terminate() {
    if (dc_) {
       dc_->terminate();
    }
  }

 private:
  std::shared_ptr<DataChannel> dc_;
  std::unordered_map<std::string, torch::Tensor> sendTensors_;
  std::unordered_map<std::string, torch::Tensor> replyTensors_;
};
}  // namespace tube
