/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <ctime>

#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <vector>

#include "core/actor.h"
#include "core/actor_player.h"
#include "core/state.h"
#include "mcts/node.h"
#include "mcts/storage.h"
#include "mcts/utils.h"

namespace mcts {

int computeRollouts(const std::vector<Node*>& rootNode,
                    const std::vector<const core::State*>& rootState,
                    const std::vector<std::vector<float>>& rnnState,
                    core::Actor& actor,
                    const MctsOption& option,
                    double thisMoveTime,
                    std::minstd_rand& rng);

class MctsPlayer : public core::ActorPlayer {
 public:
  MctsPlayer(const MctsOption& option)
      : option_(option)
      , rng_(option.seed) {
    reset();
  }

  std::vector<MctsResult> actMcts(const std::vector<const core::State*>& states,
                                  const std::vector<torch::Tensor>& rnnState);

  MctsResult actMcts(const core::State& state) {
    return actMcts({&state}, {}).at(0);
  }

  MctsResult actMcts(const core::State& state, const torch::Tensor& rnnState) {
    return actMcts({&state}, {rnnState}).at(0);
  }

  double rolloutsPerSecond() {
    return rolloutsPerSecond_;
  }

  MctsOption& option() {
    return option_;
  }

  const MctsOption& option() const {
    return option_;
  }

  virtual void reset() override {
    remaining_time = option_.totalTime;
  }

 private:
  MctsOption option_;
  double remaining_time;
  std::minstd_rand rng_;
  // Storage storage_;
  double rolloutsPerSecond_ = 0.0;
};
}  // namespace mcts
