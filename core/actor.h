/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "tube/src_cpp/channel_assembler.h"
#include "tube/src_cpp/data_block.h"
#include "tube/src_cpp/dispatcher.h"

#include "mcts/actor.h"

#include "state.h"
#include "utils.h"

//#define DEBUG_ACTOR

class Actor : public mcts::Actor {
 public:
  Actor(std::shared_ptr<tube::DataChannel> dc,
        const std::vector<int64_t>& featSize,
        const std::vector<int64_t>& actionSize,
        const std::vector<int64_t>& rnnStateSize,
        int rnnSeqlen,
        bool useValue,
        bool usePolicy,
        std::shared_ptr<tube::ChannelAssembler> assembler)
      : dispatcher_(std::move(dc))
      , useValue_(useValue)
      , usePolicy_(usePolicy)
      , policySize_(actionSize)
      , uniformPolicy_(1.0 / product(actionSize))
      , rnnStateSize_(rnnStateSize)
      , rnnSeqlen_(rnnSeqlen)
      , assembler_(assembler) {
    if (!useValue && !usePolicy_) {
      return;
    }

    feat_ = std::make_shared<tube::DataBlock>("s", featSize, torch::kFloat32);
    pi_ = std::make_shared<tube::DataBlock>("pi", actionSize, torch::kFloat32);
    value_ = std::make_shared<tube::DataBlock>(
        "v", std::initializer_list<int64_t>{1}, torch::kFloat32);

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

  mcts::PiVal& evaluate(const mcts::State& s, mcts::PiVal& pival) override {
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
      val = value_->data.item<float>();
    } else {
      val = state->getRandomRolloutReward(state->getCurrentPlayer());
    }
    if (usePolicy_ && resultsAreValid) {
      policy = pi_->data;
    } else {
      policy = torch::zeros(policySize_, torch::kFloat32);
      policy.fill_(uniformPolicy_);
    }

    getLegalPi(*state, policy, pival.policy);
    pival.playerId = state->getCurrentPlayer();
    pival.value = val;
    return pival;
  }

  void terminate() override {
    dispatcher_.terminate();
  }

  virtual void batchResize(size_t n) override final {
    if (!batchFeat_.defined() || batchFeat_[0].sizes() != feat_->data.sizes() ||
        batchFeat_.size(0) < n) {
      auto allocBatch = [&](auto&& sizes) {
        std::vector<int64_t> s1(sizes.begin(), sizes.end());
        s1.insert(s1.begin(), n);
        if (assembler_ && assembler_->hasCuda()) {
          return torch::empty(
              s1, at::TensorOptions().pinned_memory(true).requires_grad(false));
        } else {
          return torch::empty(s1, at::TensorOptions().requires_grad(false));
        }
      };
      batchFeat_ = allocBatch(feat_->data.sizes());
      batchPi_ = allocBatch(pi_->data.sizes());
      batchValue_ = allocBatch(value_->data.sizes());
      if (rnnState_) {
        batchRnnState_ = allocBatch(rnnState_->data.sizes());
        batchRnnStateOut_ = allocBatch(rnnStateOut_->data.sizes());
      }

      valueAcc_ = batchValue_.accessor<float, 2>();
      piAcc_ = batchPi_.accessor<float, 4>();
      featAcc_ = batchFeat_.accessor<float, 4>();
      if (rnnState_) {
        rnnStateAcc_ = batchRnnState_.accessor<float, 5>();
        rnnStateOutAcc_ = batchRnnStateOut_.accessor<float, 5>();
      }
    }
  }
  virtual void batchPrepare(size_t index,
                            const mcts::State& s,
                            const std::vector<float>& rnnState) override final {
    if (!assembler_) {
      if (!rnnState.empty()) {
        auto& dst = rnnState_->data;
        auto numel = dst.numel();
        if (rnnState.size() != (size_t)numel) {
          throw std::runtime_error("rnnState size mismatch; got " +
                                   std::to_string(rnnState.size()) +
                                   ", expected " + std::to_string(numel));
        }
        std::memcpy(dst.data_ptr(), rnnState.data(), sizeof(float) * numel);
      }
      return;
    }
    getFeatureInTensor(*dynamic_cast<const State*>(&s), featAcc_[index].data());
    if (!useValue_) {
      batchValue_[index][0] = s.getRandomRolloutReward(s.getCurrentPlayer());
    }
    if (!rnnState.empty()) {
      auto acc = rnnStateAcc_[index];
      auto numel = acc.stride(0) * acc.size(0);
      if (rnnState.size() != (size_t)numel) {
        throw std::runtime_error("rnnState size mismatch; got " +
                                 std::to_string(rnnState.size()) +
                                 ", expected " + std::to_string(numel));
      }
      std::memcpy(acc.data(), rnnState.data(), sizeof(float) * numel);
    }
  }
  virtual void batchEvaluate(size_t n) override final {
    if (!assembler_) {
      return;
    }
    if (useValue_ || usePolicy_) {
      double t;
      if (rnnState_) {
        t = assembler_->batchAct(
            batchFeat_.slice(0, 0, n), batchValue_.slice(0, 0, n),
            batchPi_.slice(0, 0, n), batchRnnState_.slice(0, 0, n),
            batchRnnStateOut_.slice(0, 0, n));
      } else {
        t = assembler_->batchAct(batchFeat_.slice(0, 0, n),
                                 batchValue_.slice(0, 0, n),
                                 batchPi_.slice(0, 0, n));
      }
      batchTiming_ = batchTiming_ * 0.99 + t * 0.01;
    }
  }
  virtual void batchResult(size_t index,
                           const mcts::State& s,
                           mcts::PiVal& pival) override final {
    if (!assembler_) {
      evaluate(s, pival);
      return;
    }
    float val = valueAcc_[index][0];
    getLegalPi(*dynamic_cast<const State*>(&s), piAcc_[index], pival.policy);
    pival.playerId = s.getCurrentPlayer();
    pival.value = val;
    if (rnnState_) {
      auto a = rnnStateOutAcc_[index];
      pival.rnnState.resize(a.stride(0) * a.size(0));
      std::memcpy(pival.rnnState.data(), a.data(), pival.rnnState.size());
    }
  }

  void evaluate(const std::vector<const mcts::State*>& s,
                std::vector<mcts::PiVal*>& pival,
                const std::function<void(size_t, mcts::PiVal&)>& resultCallback)
      override {

    std::terminate();
    if (!assembler_) {
      return mcts::Actor::evaluate(s, pival, resultCallback);
    }

    batchResize(s.size());

    if (useValue_ || usePolicy_) {
      for (size_t i = 0; i != s.size(); ++i) {
        batchPrepare(i, *s[i], {});
      }
      batchEvaluate(s.size());
    }

    auto valueAcc = batchValue_.accessor<float, 2>();
    auto piAcc = batchPi_.accessor<float, 4>();

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

  virtual void recordMove(const mcts::State* state) override {
    auto id = assembler_->getTournamentModelId();
    ++modelTrackers_[state][id];
  }

  virtual std::string getModelId() const override {
    return assembler_ ? std::string(assembler_->getTournamentModelId()) : "dev";
  }

  virtual void result(const mcts::State* state, float reward) override {
    if (assembler_) {
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
        assembler_->result(reward, std::move(m));
        modelTrackers_.erase(i);
      }
    }
  }

  virtual bool isTournamentOpponent() const override {
    return assembler_ ? assembler_->isTournamentOpponent() : false;
  }

  virtual double batchTiming() const override {
    return batchTiming_;
  }

  virtual std::vector<int64_t> rnnStateSize() const override {
    return rnnStateSize_;
  }

  virtual int rnnSeqlen() const override {
    return rnnSeqlen_;
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

  std::shared_ptr<tube::ChannelAssembler> assembler_;

  torch::Tensor batchFeat_;
  torch::Tensor batchPi_;
  torch::Tensor batchValue_;
  torch::Tensor batchRnnState_;
  torch::Tensor batchRnnStateOut_;

  torch::TensorAccessor<float, 2> valueAcc_{nullptr, nullptr, nullptr};
  torch::TensorAccessor<float, 4> piAcc_{nullptr, nullptr, nullptr};
  torch::TensorAccessor<float, 4> featAcc_{nullptr, nullptr, nullptr};
  torch::TensorAccessor<float, 5> rnnStateAcc_{nullptr, nullptr, nullptr};
  torch::TensorAccessor<float, 5> rnnStateOutAcc_{nullptr, nullptr, nullptr};

  std::unordered_map<const mcts::State*,
                     std::unordered_map<std::string_view, float>>
      modelTrackers_;
  double batchTiming_ = 0.0;

  const std::vector<int64_t> rnnStateSize_;
  int rnnSeqlen_ = 0;
};
