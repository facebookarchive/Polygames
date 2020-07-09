/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "data_channel.h"

using tube::DataBlock;
using tube::DataChannel;

void createBuffers(int batchsize,
                   const std::vector<std::shared_ptr<DataBlock>>& blocks,
                   std::unordered_map<std::string, torch::Tensor>& buffer) {
  for (const auto& b : blocks) {
    std::vector<int64_t> sizes =
        tube::utils::pushLeft((int64_t)batchsize, b->sizes());
    auto ret = buffer.insert({b->name, torch::zeros(sizes, b->dtype())});
    assert(ret.second);
  }
}

void checkBuffers(
    int batchsize,
    const std::vector<std::shared_ptr<DataBlock>>& blocks,
    const std::unordered_map<std::string, torch::Tensor>& buffer) {
  int numBuffer = 0;
  for (const auto& b : blocks) {
    std::vector<int64_t> sizes =
        tube::utils::pushLeft((int64_t)batchsize, b->sizes());
    auto dtype = b->dtype();

    std::unordered_map<std::string, torch::Tensor>::const_iterator it;
    it = buffer.find(b->name);
    assert(it != buffer.end());
    ++numBuffer;

    auto refSizes = it->second.sizes().vec();
    auto refDtype = it->second.dtype();

    // utils::printVector(refSizes);
    // utils::printVector(sizes);
    assert(sizes == refSizes);
    assert(dtype == refDtype);
  }
  assert(numBuffer == (int)buffer.size());
}

void DataChannel::createOrCheckBuffers(
    const std::vector<std::shared_ptr<DataBlock>>& send,
    const std::vector<std::shared_ptr<DataBlock>>& reply) {
  assert(!send.empty() || !reply.empty());
  if (sendName2Buffer_.empty() && replyName2Buffer_.empty()) {
    // only the first call will create the buffer
    createBuffers(batchsize, send, sendName2Buffer_);
    createBuffers(batchsize, reply, replyName2Buffer_);
  } else {
    checkBuffers(batchsize, send, sendName2Buffer_);
    checkBuffers(batchsize, reply, replyName2Buffer_);
  }
}

void DataChannel::terminate() {
  // called by python, once called, unblock getInput and returns so
  // that the final python future waiting for getting input can end
  // if called, the return value from getInput is undefined
  std::unique_lock<std::mutex> lk1(mFilled_);
  std::unique_lock<std::mutex> lk2(mReplied_);
  std::unique_lock<std::mutex> lk3(mAvailSlots_);
  terminated_ = true;
  cvFilled_.notify_all();
  cvReplied_.notify_all();
  cvAvailSlots_.notify_all();
}

// for consumer (python) to get input
// timeout < 0, wait until full batch
// timeout == 0, return immediately
// timeout > 0, wait until full batch or (timeout && batch not empty)
const std::unordered_map<std::string, torch::Tensor> DataChannel::getInput() {
  std::unique_lock<std::mutex> lk(mFilled_);
  if (timeoutMs < 0) {
    cvFilled_.wait(
        lk, [this] { return terminated_ || numFilledSlot_ == batchsize; });
    return sendName2Buffer_;
  }

  bool returnAll = false;
  do {
    returnAll =
        cvFilled_.wait_for(lk, std::chrono::milliseconds(timeoutMs), [this] {
          return terminated_ || numFilledSlot_ == batchsize;
        });
  } while (numFilledSlot_ == 0 && !terminated_);

  if (returnAll) {
    return sendName2Buffer_;
  }

  // hold the lock to prevent new "mark-as-filled"
  lkFilled_ = std::move(lk);
  return sliceTensorsForSend();
}

// for consumer (python) to set reply
void DataChannel::setReply(
    const std::unordered_map<std::string, torch::Tensor>& reply) {
  if (sentSlots_.empty()) {
    if (numFilledSlot_ != batchsize) {
      std::cout << name << ", setReply: numFilledSlots: " << numFilledSlot_
                << " != batchsize: " << batchsize << std::endl;
      assert(false);
    }
    tube::utils::copyTensors(reply, replyName2Buffer_);
  } else {
    if (numFilledSlot_ >= batchsize) {
      std::cout << name << ", setReply: numFilledSlots: " << numFilledSlot_
                << " >= batchsize: " << batchsize << std::endl;
      assert(false);
    }
    tube::utils::copyTensors(reply, replyName2Buffer_, sentSlots_);
  }

  // lock free, other thread is waiting on cvAvailSlots_ or cvReplied_,
  // or, when timeout >= 0, blocked by mFilled_
  numFilledSlot_ = 0;

  {
    std::lock_guard<std::mutex> lk(mReplied_);
    for (int i = 0; i < (int)slotStatus_.size(); ++i) {
      if (slotStatus_[i] == SlotStatus::filled) {
        slotStatus_[i] = SlotStatus::replied;
      } else if (slotStatus_[i] == SlotStatus::filledAutoRelease) {
        slotStatus_[i] = SlotStatus::replied;
        releaseSlot(i);
      }
    }
  }

  if (!sentSlots_.empty()) {
    lkFilled_.unlock();
    sentSlots_.clear();
  }

  cvReplied_.notify_all();
}

std::unordered_map<std::string, torch::Tensor> DataChannel::getSlot(
    int* pSlot) {
  std::unique_lock<std::mutex> lk(mAvailSlots_);
  cvAvailSlots_.wait(
      lk, [this] { return availSlots_.size() > 0 || terminated_; });
  if (terminated_) {
    return {};
  }
  int slot = availSlots_.back();
  availSlots_.pop_back();
  assert(slotStatus_[slot] == SlotStatus::avail);
  lk.unlock();

  *pSlot = slot;
  std::unordered_map<std::string, torch::Tensor> buffers;
  for (auto& name2tensor : sendName2Buffer_) {
    auto tensor = name2tensor.second.slice(0, slot, slot + 1).squeeze(0);
    buffers[name2tensor.first] = tensor;
  }
  return buffers;
}

void DataChannel::markSlotFilled(int slot) {
  std::unique_lock<std::mutex> lk(mFilled_);

  assert(slotStatus_.at(slot) == SlotStatus::avail);
  slotStatus_.at(slot) = SlotStatus::filled;

  numFilledSlot_ += 1;
  assert(numFilledSlot_ <= batchsize);
  if (numFilledSlot_ == batchsize) {
    lk.unlock();
    cvFilled_.notify_all();  // there should be only one waiting
  }
}

void DataChannel::markSlotFilledAutoRelease(int slot) {
  std::unique_lock<std::mutex> lk(mFilled_);

  assert(slotStatus_.at(slot) == SlotStatus::avail);
  slotStatus_.at(slot) = SlotStatus::filledAutoRelease;

  numFilledSlot_ += 1;
  assert(numFilledSlot_ <= batchsize);
  if (numFilledSlot_ == batchsize) {
    lk.unlock();
    cvFilled_.notify_all();  // there should be only one waiting
  }
}

std::unordered_map<std::string, torch::Tensor> DataChannel::getReply(int slot) {
  std::unique_lock<std::mutex> lk(mReplied_);
  cvReplied_.wait(lk, [this, slot] {
    return slotStatus_[slot] == SlotStatus::replied || terminated_;
  });
  lk.unlock();

  std::unordered_map<std::string, torch::Tensor> buffers;
  for (auto& name2tensor : replyName2Buffer_) {
    auto tensor = name2tensor.second.slice(0, slot, slot + 1).squeeze(0);
    buffers[name2tensor.first] = tensor;
  }
  return buffers;
}

void DataChannel::releaseSlot(int slot) {
  // assert(slotStatus_[slot] == SlotStatus::replied);
  slotStatus_[slot] = SlotStatus::avail;

  std::unique_lock<std::mutex> lk(mAvailSlots_);
  availSlots_.push_back(slot);
  lk.unlock();
  cvAvailSlots_.notify_one();
}

const std::unordered_map<std::string, torch::Tensor>
DataChannel::sliceTensorsForSend() {
  assert(sentSlots_.empty());
  for (int i = 0; i < (int)slotStatus_.size(); ++i) {
    if (slotStatus_[i] == SlotStatus::filled ||
        slotStatus_[i] == SlotStatus::filledAutoRelease) {
      sentSlots_.push_back(i);
    }
  }
  assert((int)sentSlots_.size() < batchsize);

  torch::Tensor indices = torch::from_blob(
      sentSlots_.data(), {(int64_t)sentSlots_.size()}, torch::kInt64);
  std::unordered_map<std::string, torch::Tensor> sliced;
  for (const auto& name2tensor : sendName2Buffer_) {
    const std::string& name = name2tensor.first;
    const torch::Tensor& tensor = name2tensor.second.index_select(0, indices);
    sliced.insert({name, tensor});
  }
  return sliced;
}
