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

#include "mcts/node.h"
#include "mcts/storage.h"
#include "mcts/utils.h"
#include "core/actor.h"
#include "core/actor_player.h"
#include "core/state.h"

namespace mcts {

struct PersistentTree {
  Node* root = nullptr;
  Storage* storage = nullptr;
  PersistentTree() = default;
  PersistentTree(const PersistentTree&) = delete;
  PersistentTree(PersistentTree&& n) {
    root = std::exchange(n.root, nullptr);
    storage = std::exchange(n.storage, nullptr);
  }
  PersistentTree& operator=(const PersistentTree&) = delete;
  PersistentTree& operator=(PersistentTree&& n) {
    std::swap(root, n.root);
    std::swap(storage, n.storage);
    return *this;
  }
  ~PersistentTree() {
    if (root) {
      root->freeTree();
    }
  }
};

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
      : option_(option), rng_(option.seed) {
    remaining_time = option.totalTime;
  }

  std::vector<MctsResult> actMcts(
      const std::vector<const core::State*>& states,
      const std::vector<torch::Tensor>& rnnState,
      const std::vector<PersistentTree*>& persistentTrees);

  MctsResult actMcts(const core::State& state) {
    return actMcts({&state}, {}, {}).at(0);
  }

  MctsResult actMcts(const core::State& state, const torch::Tensor& rnnState) {
    return actMcts({&state}, {rnnState}, {}).at(0);
  }

  double rolloutsPerSecond() {
    return rolloutsPerSecond_;
  }

  const MctsOption& option() {
    return option_;
  }

 private:
  MctsOption option_;
  double remaining_time;
  std::minstd_rand rng_;
  // Storage storage_;
  double rolloutsPerSecond_ = 0.0;
};
}  // namespace mcts
