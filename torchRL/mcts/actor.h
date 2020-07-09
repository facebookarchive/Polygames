/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "mcts/state.h"
#include "mcts/utils.h"

#include <functional>

namespace mcts {

// this is a minimal interface class,
// should ONLY keep functions used by mcts
class Actor {
 public:
  Actor() = default;

  Actor(const Actor&) = delete;
  Actor& operator=(const Actor&) = delete;

  virtual PiVal& evaluate(const State& s, PiVal& pival) = 0;

  virtual ~Actor() {
  }

  virtual void batchResize(size_t n) {
  }
  virtual void batchPrepare(size_t index,
                            const State& s,
                            const std::vector<float>& rnnState) {
    std::terminate();
  }
  virtual void batchEvaluate(size_t n) {
  }
  virtual void batchResult(size_t index, const State& s, PiVal& pival) {
    std::terminate();
  }

  virtual void evaluate(
      const std::vector<const State*>& s,
      std::vector<PiVal*>& pival,
      const std::function<void(size_t, PiVal&)>& resultCallback) {
    for (size_t i = 0; i != s.size(); ++i) {
      resultCallback(i, evaluate(*s[i], *pival[i]));
    }
  };

  virtual void terminate() {
  }

  virtual void recordMove(const mcts::State* state) {
  }

  virtual void result(const State* state, float reward) {
  }

  virtual bool isTournamentOpponent() const {
    return false;
  }

  virtual double batchTiming() const {
    return -1.0f;
  }

  virtual std::string getModelId() const {
    return "";
  }

  virtual std::vector<int64_t> rnnStateSize() const {
    return {};
  }
  virtual int rnnSeqlen() const {
    return 0;
  }
};

}  // namespace mcts
