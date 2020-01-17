/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <memory>
#include <vector>

#include "../data_block.h"
#include "../data_channel.h"
#include "../dispatcher.h"
#include "../env_thread.h"

namespace tube {

class ProducerThread : public EnvThread {
public:
  ProducerThread(int threadIdx, std::shared_ptr<DataChannel> dc)
      : threadIdx(threadIdx),
        dispatcher_(std::make_unique<Dispatcher>(std::move(dc))) {
    s_ = std::make_shared<DataBlock>("s", std::initializer_list<int64_t>{1},
                                     torch::kInt32);
    a_ = std::make_shared<DataBlock>("a", std::initializer_list<int64_t>{1},
                                     torch::kInt32);

    dispatcher_->addDataBlocks({s_}, {a_});
    std::cout << "create thread: " << threadIdx << std::endl;
  }

  virtual void mainLoop() override {
    std::cout << "in mainloop, thread: " << threadIdx << std::endl;
    for (int i = 0; i < 10; ++i) {
      s_->data[0] = threadIdx;
      dispatcher_->dispatch();

      std::cout << "thread: " << threadIdx << ", reply: " << a_->data[0]
                << std::endl;
    }
  }

  const int threadIdx;

private:
  std::unique_ptr<Dispatcher> dispatcher_;
  std::shared_ptr<DataBlock> s_;
  std::shared_ptr<DataBlock> a_;
  ;
};

class DualDispatchThread : public EnvThread {
public:
  DualDispatchThread(int threadIdx, int maxStep,
                     std::shared_ptr<DataChannel> dcFast,
                     std::shared_ptr<DataChannel> dcSlow)
      : threadIdx(threadIdx), maxStep(maxStep),
        dispatcherFast_(std::make_unique<Dispatcher>(std::move(dcFast))),
        dispatcherSlow_(std::make_unique<Dispatcher>(std::move(dcSlow))) {
    auto sf = std::make_shared<DataBlock>(
        "s", std::initializer_list<int64_t>{1}, torch::kInt32);
    auto af = std::make_shared<DataBlock>(
        "a", std::initializer_list<int64_t>{1}, torch::kInt32);

    blocksFastSend_.push_back(sf);
    blocksFastReply_.push_back(af);
    dispatcherFast_->addDataBlocks(blocksFastSend_, blocksFastReply_);

    auto ss = std::make_shared<DataBlock>(
        "s", std::initializer_list<int64_t>{1}, torch::kInt32);
    auto as = std::make_shared<DataBlock>(
        "a", std::initializer_list<int64_t>{1}, torch::kInt32);

    blocksSlowSend_.push_back(ss);
    blocksSlowReply_.push_back(as);
    dispatcherSlow_->addDataBlocks(blocksSlowSend_, blocksSlowReply_);

    std::cout << "create thread: " << threadIdx << std::endl;
  }

  virtual void mainLoop() override {
    std::cout << "in mainloop, thread: " << threadIdx << std::endl;
    int i = 0;
    while (i < 3) {
      blocksFastSend_[0]->data[0] = threadIdx;
      dispatcherFast_->dispatch();
      // std::cout << "thread: " << threadIdx << ", stepIdx: " << stepIdx_ << ",
      // reply(fast): "
      //           << blocksFastReply_[0]->data[0] << std::endl;
      ++stepIdx_;
      if (stepIdx_ == maxStep) {
        // std::cout << ">>>thread Slow dispatch" << std::endl;
        blocksSlowSend_[0]->data[0] = threadIdx * threadIdx;
        dispatcherSlow_->dispatch();
        // std::cout << "thread: " << threadIdx << ", reply(slow): "
        //           << blocksSlowReply[0]->data[0] << std::endl;
        stepIdx_ = 0;
        ++i;
      }
    }
    std::cout << "thread: " << threadIdx << " done" << std::endl;
  }

  const int threadIdx;
  const int maxStep;

private:
  int stepIdx_ = 0;
  std::unique_ptr<Dispatcher> dispatcherFast_;
  std::unique_ptr<Dispatcher> dispatcherSlow_;
  std::vector<std::shared_ptr<DataBlock>> blocksFastSend_, blocksFastReply_;
  std::vector<std::shared_ptr<DataBlock>> blocksSlowSend_, blocksSlowReply_;
};

}
