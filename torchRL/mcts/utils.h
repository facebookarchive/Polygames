/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

#include "mcts/types.h"

namespace mcts {

class MctsOption {
 public:
  float totalTime = 0;
  float timeRatio = 0.07;
  // whether to use mcts simulations to decide a move
  // if true, it will run mcts rollouts and selection action using uct
  // if false, it will only use policy in MctsPlayer::actMcts
  bool useMcts = true;

  // coefficient of prior score
  float puct = 0.0;

  // TODO[qucheng]: persistentTree is not implemented
  // bool persistentTree = false;

  // first K steps in the game where we use sample instead of greedily
  // pick the best action. For example, if K = 6, then each player will
  // sample action based on mcts probability for their first 3 steps
  // in a two player game.
  int sampleBeforeStepIdx = 0;

  // num of rollout for each move
  int numRolloutPerThread = -1;

  // TODO[qucheng]: implement time based mcts
  // // in seconds
  // int maxTimeSec = -1;

  int seed = 123;

  // capacity of pre-allocated storage space
  int storageCap = 100000;

  float virtualLoss = 0.0;

  // If true, initialize unvisited node with prior values from siblings
  bool useValuePrior = true;

  // Whether to store the game state in each node, or recompute it on each
  // rollout. Not storing it saves memory and can have greater performance
  // on simple games that are quick to recompute.
  bool storeStateInNode = false;
};

class MctsStats {
 public:
  MctsStats() {
    reset();
  }

  void reset() {
    value_ = 0.0;
    numVisit_ = 0;
    virtualLoss_ = 0.0;
    sumChildV_ = 0.0;
    numChild_ = 0;
  }

  float getValue() const {
    // std::lock_guard<std::mutex> lock(mSelf_);
    return value_;
  }

  int getNumVisit() const {
    // std::lock_guard<std::mutex> lock(mSelf_);
    return numVisit_;
  }

  // Get prior child value (from the perspective of the current node).
  //
  // When a child hasn't been explored yet, we don't know its value and need to
  // use the "prior" from other children that has been explored before.
  // This is very important, otherwise the tree search could be overly-
  // optimistic and explore all actions once and waste a lot of rollouts (which
  // is bad for cases with high-branching factor).
  float getAvgChildV() const {
    if (numChild_ == 0)
      return 0.0;
    else
      return sumChildV_ / numChild_;
  }

  float getAvgValue() const {
    assert(numVisit_ > 0);
    return value_ / numVisit_;
  }

  float getVirtualLoss() const {
    return virtualLoss_;
  }

  void addVirtualLoss(float virtualLoss) {
    std::lock_guard<std::mutex> lock(mSelf_);
    virtualLoss_ += virtualLoss;
  }

  void atomicUpdate(float value, float virtualLoss) {
    std::lock_guard<std::mutex> lock(mSelf_);
    value_ += value;
    numVisit_++;
    virtualLoss_ -= virtualLoss;
  }

  // Update child value estimate with a new obtained child value
  // (from the perspective of the root node
  void atomicUpdateChildV(float childV) {
    std::lock_guard<std::mutex> lock(mSelf_);
    sumChildV_ += childV;
    numChild_++;
  }

  std::string summary() const {
    std::stringstream ss;
    ss << value_ << "/" << numVisit_ << " (" << value_ / numVisit_
       << "), vloss: " << virtualLoss_;
    return ss.str();
  }

 private:
  float value_;
  int numVisit_;
  float virtualLoss_;

  // Summation of the value prediction from a child
  float sumChildV_;
  // # child that has been explored.
  int numChild_;

  std::mutex mSelf_;
};

class MctsResult {
 public:
  MctsResult(std::minstd_rand* rng)
      : maxVisits(0)
      , sumVisits(0)
      , bestAction(InvalidAction)
      , rng_(rng) {
  }

  void add(const Action& a, int visits) {
    mctsPolicy[a] = visits;
    sumVisits += visits;
    if (visits > maxVisits) {
      maxVisits = visits;
      bestAction = a;
    }
  }

  void normalize() {
    for (auto& pair : mctsPolicy) {
      pair.second = pair.second / (float)sumVisits;
    }
  }

  // assume already normalized
  void sample() {
    float best = 0.0f;
    for (auto& pair : mctsPolicy) {
      float v = std::exp(pair.second * pair.second * 2) - 0.92f;
      v = std::uniform_real_distribution<float>(0.0f, v)(*rng_);
      if (v > best) {
        best = v;
        bestAction = pair.first;
      }
    }
  }

  void setMctsPolicy(std::unordered_map<Action, float> pi) {
    mctsPolicy = std::move(pi);
  }

  int maxVisits;
  int sumVisits;
  Action bestAction;
  std::unordered_map<Action, float> mctsPolicy;
  float rootValue;

 private:
  std::minstd_rand* rng_;
};

class PiVal {
 public:
  PiVal() {
    reset();
  }

  PiVal(int playerId, float value, std::unordered_map<Action, float>&& policy)
      : playerId(playerId)
      , value(value)
      , policy(policy) {
  }

  void reset() {
    playerId = -999;
    value = 0.0;
    policy.clear();
  }

  int playerId;
  float value;
  std::unordered_map<Action, float> policy;
};

inline void printPolicy(const std::unordered_map<Action, float>& pi) {
  for (const auto& a2p : pi) {
    std::cout << a2p.first << ":" << a2p.second << std::endl;
  }
}

}  // namespace mcts
