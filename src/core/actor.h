/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "tube/src_cpp/data_block.h"
#include "tube/src_cpp/dispatcher.h"
#include "model_manager.h"

#include "state.h"
#include "utils.h"

//#define DEBUG_ACTOR

namespace core {

class PiVal {
 public:
  PiVal() {
    reset();
  }

//  PiVal(int playerId, float value, std::vector<float> policy)
//      : playerId(playerId)
//      , value(value)
//      , policy(std::move(policy)) {}

  void reset() {
    playerId = -999;
    value = 0.0f;
    logitPolicy.reset();
    rnnState.reset();
  }

  int playerId;
  float value;
  torch::Tensor logitPolicy;
  torch::Tensor rnnState;
};

class Actor {
 public:
  Actor(std::shared_ptr<tube::DataChannel> dc,
        const std::vector<int64_t>& featSize,
        const std::vector<int64_t>& actionSize,
        const std::vector<int64_t>& rnnStateSize,
        int rnnSeqlen,
        bool logitValue,
        bool useValue,
        bool usePolicy,
        std::shared_ptr<ModelManager> modelManager)
      : dispatcher_(std::move(dc))
      , useValue_(useValue)
      , usePolicy_(usePolicy)
      , policySize_(actionSize)
      , uniformPolicy_(1.0 / product(actionSize))
      , rnnStateSize_(rnnStateSize)
      , rnnSeqlen_(rnnSeqlen)
      , logitValue_(logitValue)
      , modelManager_(modelManager) {
    if (!useValue && !usePolicy_) {
      return;
    }

    feat_ = std::make_shared<tube::DataBlock>("s", featSize, torch::kFloat32);
    pi_ = std::make_shared<tube::DataBlock>("pi_logit", actionSize, torch::kFloat32);
    value_ = std::make_shared<tube::DataBlock>(
        "v", std::initializer_list<int64_t>{logitValue ? 3 : 1}, torch::kFloat32);

    if (!rnnStateSize.empty()) {
      rnnState_ = std::make_shared<tube::DataBlock>(
          "rnn_state", rnnStateSize, torch::kFloat32);
      rnnStateOut_ = std::make_shared<tube::DataBlock>(
          "rnn_state_out", rnnStateSize, torch::kFloat32);
    }

    if (rnnStateSize.empty()) {
      dispatcher_.addDataBlocks({feat_}, {pi_, value_});
    } else {
      dispatcher_.addDataBlocks(
          {feat_, rnnState_}, {pi_, value_, rnnStateOut_});
    }
  }

  PiVal& evaluate(const core::State& s, PiVal& pival) {
    const auto state = dynamic_cast<const State*>(&s);
    assert(state != nullptr);

    // termination should be handled by mcts
    assert(!state->terminated());

    bool resultsAreValid = false;
    if (useValue_ || usePolicy_) {
      getFeatureInTensor(*state, feat_->data);
      int errcode = dispatcher_.dispatch();
      switch (errcode) {
      case tube::Dispatcher::DISPATCH_ERR_DC_TERM:
#ifdef DEBUG_ACTOR
        std::cout << "actor " << this << ": attempt to dispatch through"
                  << " a terminated data channel " << std::endl;
#endif
        break;
      case tube::Dispatcher::DISPATCH_ERR_NO_SLOT:
#ifdef DEBUG_ACTOR
        std::cout << "actor " << this << ": no slots available to dispatch"
                  << std::endl;
#endif
        break;
      case tube::Dispatcher::DISPATCH_NOERR:
        resultsAreValid = true;
      }
    }

    float val;
    torch::Tensor policy;
    if (useValue_ && resultsAreValid) {
      if (logitValue_) {
        float* begin = value_->data.data_ptr<float>();
        float* end = begin + 3;
        softmax_(begin, end);
      }
      val = logitValue_ ? value_->data[0].item<float>() - value_->data[1].item<float>() : value_->data.item<float>();
    } else {
      val = state->getRandomRolloutReward(state->getCurrentPlayer());
    }
    if (usePolicy_ && resultsAreValid) {
      policy = pi_->data;
    } else {
      policy = torch::zeros(policySize_, torch::kFloat32);
      policy.fill_(uniformPolicy_);
    }

    pival.logitPolicy = policy.clone();
    pival.playerId = state->getCurrentPlayer();
    pival.value = val;
    if (rnnStateOut_) {
      pival.rnnState = rnnStateOut_->data.clone();
    }
    return pival;
  }

  void terminate() {
    dispatcher_.terminate();
  }

  void batchResize(size_t n){
    if (!batchFeat_.defined() || batchFeat_[0].sizes() != feat_->data.sizes() ||
        batchFeat_.size(0) < n) {
      auto allocBatch = [&](auto&& sizes) {
        std::vector<int64_t> s1(sizes.begin(), sizes.end());
        s1.insert(s1.begin(), n);
        if (modelManager_ && modelManager_->isCuda()) {
          return torch::empty(
              s1, at::TensorOptions().pinned_memory(true).requires_grad(false));
        } else {
          return torch::empty(s1, at::TensorOptions().requires_grad(false));
        }
      };
      batchFeat_ = allocBatch(feat_->data.sizes());
      batchPi_ = allocBatch(pi_->data.sizes());
      batchValue_ = allocBatch(value_->data.sizes());

      valueAcc_ = batchValue_.accessor<float, 2>();
      piAcc_ = batchPi_.accessor<float, 4>();
      featAcc_ = batchFeat_.accessor<float, 4>();
    }
    if (rnnState_) {
      rnnStateStack_.resize(n);
    }
  }
  void batchPrepare(size_t index,
                            const core::State& s,
                            torch::Tensor rnnState) {
    if (!modelManager_) {
      if (rnnState.defined()) {
        rnnState_->data.copy_(rnnState);
      }
      return;
    }
    getFeatureInTensor(*dynamic_cast<const State*>(&s), featAcc_[index].data());
    if (!useValue_) {
      batchValue_[index][0] = s.getRandomRolloutReward(s.getCurrentPlayer());
    }
    if (rnnState.defined()) {
      if (rnnState.device() != device()) {
        rnnState = rnnState.to(device());
      }
      rnnStateStack_.at(index) = rnnState;
    }
  }
  void batchEvaluate(size_t n) {
    if (!modelManager_) {
      return;
    }
    if (useValue_ || usePolicy_) {
      if (rnnState_) {
        modelManager_->batchAct(
            batchFeat_.narrow(0, 0, n), batchValue_.narrow(0, 0, n),
            batchPi_.narrow(0, 0, n), torch::stack(rnnStateStack_),
            &batchRnnStateOut_);
      } else {
        modelManager_->batchAct(batchFeat_.narrow(0, 0, n),
                                 batchValue_.narrow(0, 0, n),
                                 batchPi_.narrow(0, 0, n));
      }
    }
  }
  void batchResult(size_t index,
                           const core::State& s,
                           PiVal& pival) {
    if (!modelManager_) {
      evaluate(s, pival);
      return;
    }
    if (logitValue_) {
      float* begin = &valueAcc_[index][0];
      float* end = begin + 3;
      softmax_(begin, end);
    }
    float val = logitValue_ ? valueAcc_[index][0] - valueAcc_[index][1] : valueAcc_[index][0];
    pival.logitPolicy = batchPi_[index].clone();
    pival.playerId = s.getCurrentPlayer();
    pival.value = val;
    if (rnnState_) {
      pival.rnnState = batchRnnStateOut_[index];
    }
  }

  void evaluate(const std::vector<const core::State*>& s,
                std::vector<PiVal*>& pival,
                const std::function<void(size_t, PiVal&)>& resultCallback) {
    // Is this code path dead?
    std::terminate();
    if (!modelManager_) {
      for (size_t i = 0; i != s.size(); ++i) {
        resultCallback(i, evaluate(*s[i], *pival[i]));
      }
      return;
    }

    batchResize(s.size());

    if (useValue_ || usePolicy_) {
      for (size_t i = 0; i != s.size(); ++i) {
        batchPrepare(i, *s[i], {});
      }
      batchEvaluate(s.size());
    }

    if (!usePolicy_) {
      batchPi_.fill_(uniformPolicy_);
    }

    for (size_t i = 0; i != s.size(); ++i) {
      auto* state = s[i];
      auto& pv = *pival[i];
      batchResult(i, *state, pv);
      resultCallback(i, pv);
    }
  }

  void recordMove(const core::State* state) {
    auto id = modelManager_->getTournamentModelId();
    ++modelTrackers_[state][id];
  }

  std::string getModelId() const {
    return modelManager_ ? std::string(modelManager_->getTournamentModelId()) : "dev";
  }

  void result(const core::State* state, float reward) {
    if (modelManager_) {
      auto i = modelTrackers_.find(state);
      if (i != modelTrackers_.end()) {
        auto m = std::move(i->second);
        float sum = 0.0f;
        for (auto& v : m) {
          sum += v.second;
        }
        for (auto& v : m) {
          v.second /= sum;
        }
        modelManager_->result(reward, std::move(m));
        modelTrackers_.erase(i);
      }
    }
  }

  void forget(const core::State* state) {
    if (modelManager_) {
      auto i = modelTrackers_.find(state);
      if (i != modelTrackers_.end()) {
        modelTrackers_.erase(i);
      }
    }
  }

  bool isTournamentOpponent() const {
    return modelManager_ ? modelManager_->isTournamentOpponent() : false;
  }

  bool wantsTournamentResult() const {
    return modelManager_ ? modelManager_->wantsTournamentResult() : false;
  }

  std::vector<int64_t> rnnStateSize() const {
    return rnnStateSize_;
  }

  int rnnSeqlen() const {
    return rnnSeqlen_;
  }

  int vOutputs() const {
    return logitValue_ ? 3 : 1;
  }

  int findBatchSize(const core::State& state) const {
    if (modelManager_) {
      if (rnnState_) {
        return modelManager_->findBatchSize(getFeatureInTensor(*dynamic_cast<const State*>(&state)), torch::zeros(rnnStateSize_));
      } else {
        return modelManager_->findBatchSize(getFeatureInTensor(*dynamic_cast<const State*>(&state)));
      }
    }
    return 0;
  }

  bool isCuda() const {
    return modelManager_ ? modelManager_->isCuda() : false;
  }

  torch::Device device() const {
    return modelManager_ ? modelManager_->device() : torch::Device(torch::kCPU);
  }

 private:
  tube::Dispatcher dispatcher_;

  std::shared_ptr<tube::DataBlock> feat_;
  std::shared_ptr<tube::DataBlock> pi_;
  std::shared_ptr<tube::DataBlock> value_;
  std::shared_ptr<tube::DataBlock> rnnState_;
  std::shared_ptr<tube::DataBlock> rnnStateOut_;

  const bool useValue_;
  const bool usePolicy_;
  const std::vector<int64_t> policySize_;
  const float uniformPolicy_;

  torch::Tensor batchFeat_;
  torch::Tensor batchPi_;
  torch::Tensor batchValue_;

  torch::Tensor batchRnnStateOut_;

  torch::TensorAccessor<float, 2> valueAcc_{nullptr, nullptr, nullptr};
  torch::TensorAccessor<float, 4> piAcc_{nullptr, nullptr, nullptr};
  torch::TensorAccessor<float, 4> featAcc_{nullptr, nullptr, nullptr};

  std::vector<torch::Tensor> rnnStateStack_;
  torch::Tensor rnnStateStackResult_;

  std::unordered_map<const core::State*,
                     std::unordered_map<std::string_view, float>>
      modelTrackers_;

  const std::vector<int64_t> rnnStateSize_;
  int rnnSeqlen_ = 0;
  bool logitValue_ = false;
  std::shared_ptr<ModelManager> modelManager_;
};

}
