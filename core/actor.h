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
        bool useValue,
        bool usePolicy,
        std::shared_ptr<tube::ChannelAssembler> assembler)
      : dispatcher_(std::move(dc))
      , useValue_(useValue)
      , usePolicy_(usePolicy)
      , policySize_(actionSize)
      , uniformPolicy_(1.0 / product(actionSize))
      , assembler_(assembler) {
    if (!useValue && !usePolicy_) {
      return;
    }

    feat_ = std::make_shared<tube::DataBlock>("s", featSize, torch::kFloat32);
    pi_ = std::make_shared<tube::DataBlock>("pi", actionSize, torch::kFloat32);
    value_ = std::make_shared<tube::DataBlock>(
        "v", std::initializer_list<int64_t>{1}, torch::kFloat32);

    dispatcher_.addDataBlocks({feat_}, {pi_, value_});
  }

  mcts::PiVal evaluate(const mcts::State& s) override {
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

    auto a2pi = getLegalPi(*state, policy);
    return mcts::PiVal(state->getCurrentPlayer(), val, std::move(a2pi));
  }

  void terminate() override {
    dispatcher_.terminate();
  }

  void evaluate(
      const std::vector<const mcts::State*>& s,
      const std::function<void(size_t, mcts::PiVal)>& resultCallback) override {

    if (!assembler_) {
      return mcts::Actor::evaluate(s, resultCallback);
    }

    if (!batchFeat_.defined() || batchFeat_[0].sizes() != feat_->data.sizes() ||
        batchFeat_.size(0) < (int)s.size()) {
      auto allocBatch = [&](auto&& sizes) {
        std::vector<int64_t> s1(sizes.begin(), sizes.end());
        s1.insert(s1.begin(), s.size());
        return torch::empty(
            s1, at::TensorOptions().pinned_memory(true).requires_grad(false));
      };
      batchFeat_ = allocBatch(feat_->data.sizes());
      batchPi_ = allocBatch(pi_->data.sizes());
      batchValue_ = allocBatch(value_->data.sizes());
    }

    if (useValue_ || usePolicy_) {
      auto featAcc = batchFeat_.accessor<float, 4>();
      for (size_t i = 0; i != s.size(); ++i) {
        getFeatureInTensor(
            *dynamic_cast<const State*>(s[i]), featAcc[i].data());
      }
      double t = assembler_->batchAct(batchFeat_.slice(0, 0, s.size()),
                           batchValue_.slice(0, 0, s.size()),
                           batchPi_.slice(0, 0, s.size()));
      batchTiming_ = batchTiming_ * 0.99 + t * 0.01;
    }

    auto valueAcc = batchValue_.accessor<float, 2>();
    auto piAcc = batchPi_.accessor<float, 4>();

    if (!usePolicy_) {
      batchPi_.fill_(uniformPolicy_);
    }

    for (size_t i = 0; i != s.size(); ++i) {
      auto* state = s[i];
      float val;
      if (useValue_) {
        val = valueAcc[i][0];
      } else {
        val = state->getRandomRolloutReward(state->getCurrentPlayer());
      }
      auto a2pi = getLegalPi(*dynamic_cast<const State*>(state), piAcc[i]);
      resultCallback(
          i, mcts::PiVal(state->getCurrentPlayer(), val, std::move(a2pi)));
    }
  }

  virtual void recordMove(const mcts::State* state) override {
    auto id = assembler_->getTournamentModelId();
    ++modelTrackers_[state][id];
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

 private:
  tube::Dispatcher dispatcher_;

  std::shared_ptr<tube::DataBlock> feat_;
  std::shared_ptr<tube::DataBlock> pi_;
  std::shared_ptr<tube::DataBlock> value_;

  const bool useValue_;
  const bool usePolicy_;
  const std::vector<int64_t> policySize_;
  const float uniformPolicy_;

  std::shared_ptr<tube::ChannelAssembler> assembler_;

  torch::Tensor batchFeat_;
  torch::Tensor batchPi_;
  torch::Tensor batchValue_;

  std::unordered_map<const mcts::State*,
                     std::unordered_map<std::string_view, float>>
      modelTrackers_;
  double batchTiming_ = 0.0;
};
