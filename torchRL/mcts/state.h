/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "mcts/types.h"

namespace mcts {

// this is a minimal interface class,
// should ONLY keep functions used by mcts
class State {
 public:
  bool stochasticReset() const { return _stochasticReset; }
  // State() {} //= default;

  // State(const State&) = delete;
  // State& operator=(const State&) = delete;
  virtual std::unique_ptr<State> clone() const = 0;

  virtual bool isOnePlayerGame() const { return false; }

  virtual int getCurrentPlayer() const = 0;

  virtual int getStepIdx() const = 0;

  virtual float getReward(int player) const = 0;

  virtual bool terminated() const = 0;

  virtual float getRandomRolloutReward(int player) const = 0;

  virtual bool forward(const Action& a) = 0;

  virtual uint64_t getHash() const = 0;

  virtual bool operator==(const State&) const = 0;

  virtual bool isStochastic() const {
    return _stochastic;
  }

  virtual ~State() {
  }
  int forcedDice;

 protected:
  bool _stochastic;
  bool _stochasticReset;
};
}
