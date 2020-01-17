/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <vector>

#include "data_block.h"
#include "utils.h"

namespace tube {

class DataChannel {
 public:
  DataChannel(std::string name, int batchsize, int timeoutMs)
      : name(name)
      , batchsize(batchsize)
      , timeoutMs(timeoutMs) {
    for (int i = 0; i < batchsize; ++i) {
      availSlots_.push_back(i);
      slotStatus_.push_back(SlotStatus::avail);
    }
  }

  void createOrCheckBuffers(
      const std::vector<std::shared_ptr<DataBlock>>& send,
      const std::vector<std::shared_ptr<DataBlock>>& reply);

  void terminate();
  bool terminated() {
    return terminated_;
  }

  // for consumer (python) to get input
  // timeout < 0, wait until full batch
  // timeout == 0, return immediately
  // timeout > 0, wait until full batch or (timeout && batch not empty)
  const std::unordered_map<std::string, torch::Tensor> getInput();

  // for consumer (python) to set reply
  void setReply(const std::unordered_map<std::string, torch::Tensor>& reply);

  // for dispatchers
  std::unordered_map<std::string, torch::Tensor> getSlot(int* pSlot);

  void markSlotFilled(int slot);
  void markSlotFilledAutoRelease(int slot);

  std::unordered_map<std::string, torch::Tensor> getReply(int slot);

  void releaseSlot(int slot);

  const std::string name;
  const int batchsize;
  const int timeoutMs;

 private:
  const std::unordered_map<std::string, torch::Tensor> sliceTensorsForSend();

  // for slot management
  enum class SlotStatus {
    avail,
    filled,
    filledAutoRelease,
    replied,
  };

  std::unordered_map<std::string, torch::Tensor> sendName2Buffer_;
  std::unordered_map<std::string, torch::Tensor> replyName2Buffer_;

  std::vector<SlotStatus> slotStatus_;
  std::vector<int64_t> sentSlots_;

  std::mutex mAvailSlots_;
  std::condition_variable cvAvailSlots_;
  std::vector<int> availSlots_;

  std::unique_lock<std::mutex> lkFilled_;
  std::mutex mFilled_;
  std::condition_variable cvFilled_;
  int numFilledSlot_ = 0;

  std::mutex mReplied_;
  std::condition_variable cvReplied_;

  bool terminated_ = false;
};
}  // namespace tube
