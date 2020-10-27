/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <vector>

#include "../data_channel.h"
#include "test_producer.h"

using tube::DataChannel;
using tube::ProducerThread;

class ConsumerThread {
 public:
  ConsumerThread(std::shared_ptr<DataChannel>& dc)
      : dc_(dc) {
  }

  void mainLoop() {
    int numBatch = 0;
    while (true) {
      const auto& input = dc_->getInput();

      std::cout << ">>>In Consumer mainLoop, batch: " << numBatch << std::endl;
      std::cout << input.at("s") << std::endl;
      std::cout << "========================" << std::endl;

      std::unordered_map<std::string, torch::Tensor> reply;
      for (const auto& name2tensor : input) {
        reply["a"] = name2tensor.second.clone();
      }
      dc_->setReply(reply);
      ++numBatch;
    }
  }

 private:
  std::shared_ptr<DataChannel> dc_;
};

int main() {
  int batchsize = 10;
  int numThread = 10;

  auto dc = std::make_shared<DataChannel>("default_channel", batchsize, -1);

  std::vector<ProducerThread> producers;
  producers.reserve(numThread);
  std::vector<std::thread> tProducers;
  for (int i = 0; i < numThread; ++i) {
    producers.push_back(ProducerThread(i, dc));
    std::thread t(&ProducerThread::mainLoop, std::ref(producers[i]));
    tProducers.push_back(std::move(t));
    std::cout << "add producer: " << i << std::endl;
  }

  ConsumerThread consumer(dc);
  std::thread tConsumer(&ConsumerThread::mainLoop, std::ref(consumer));

  for (int i = 0; i < numThread; ++i) {
    tProducers[i].join();
  }

  tConsumer.join();

  return 0;
}
