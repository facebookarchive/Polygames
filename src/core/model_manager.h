/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tube/src_cpp/data_channel.h>

#include <future>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <unordered_map>

namespace core {

struct SampleResult {
  std::future<std::unordered_map<std::string, torch::Tensor>> fut;

  std::unordered_map<std::string, torch::Tensor> get() {
    return fut.get();
  }
};

class ModelManagerImpl;
class ModelManager {
  std::unique_ptr<ModelManagerImpl> impl;

 public:
  ModelManager();
  ModelManager(int actBatchsize,
               const std::string& device,
               int replayCapacity,
               int seed,
               const std::string& jitModel,
               int trainChannelTimeoutMs,
               int trainChannelNumSlots);
  ~ModelManager();

  std::shared_ptr<tube::DataChannel> getTrainChannel();
  std::shared_ptr<tube::DataChannel> getActChannel();
  void updateModel(
      const std::unordered_map<std::string, torch::Tensor>& stateDict);
  int bufferSize() const;
  bool bufferFull() const;
  std::unordered_map<std::string, torch::Tensor> sample(int sampleSize);
  void start();
  void testAct();
  void setIsTournamentOpponent(bool mode);
  void addTournamentModel(
      std::string id,
      const std::unordered_map<std::string, torch::Tensor>& stateDict);
  void setDontRequestModelUpdates(bool v);
  void startServer(std::string serverListenEndpoint);
  void startClient(std::string serverConnectHostname);
  void startReplayBufferServer(std::string endpoint);
  void startReplayBufferClient(std::string endpoint);
  SampleResult remoteSample(int sampleSize);

  bool isCuda() const;
  torch::Device device() const;
  void batchAct(torch::Tensor input,
                torch::Tensor v,
                torch::Tensor pi,
                torch::Tensor rnnState = {},
                torch::Tensor* rnnStateOut = nullptr);
  std::string_view getTournamentModelId();
  void result(float reward, std::unordered_map<std::string_view, float> models);
  int findBatchSize(torch::Tensor input, torch::Tensor rnnState = {});
  int64_t bufferNumSample() const;
  int64_t bufferNumAdd() const;
  bool isTournamentOpponent() const;
  bool wantsTournamentResult();

  void setFindBatchSizeMaxMs(float ms);
  void setFindBatchSizeMaxBs(int n);
};

}  // namespace core
