/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

#include "core/actor.h"
#include "mcts/types.h"

namespace mcts {

class MctsOption {
 public:
  float totalTime = 0;
  float timeRatio = 0.035;
  // coefficient of prior score
  float puct = 0.0;

  // first K steps in the game where we use sample instead of greedily
  // pick the best action. For example, if K = 6, then each player will
  // sample action based on mcts probability for their first 3 steps
  // in a two player game.
  int sampleBeforeStepIdx = 0;

  // num of rollout for each move
  int numRolloutPerThread = -1;

  int seed = 123;

  float virtualLoss = 0.0;

  // If true, initialize unvisited node with prior values from siblings
  bool useValuePrior = true;

  // Store the state in the MCTS node at multiples of this tree depth.
  int storeStateInterval = 1;

  bool randomizedRollouts = false;

  bool samplingMcts = false;

  float forcedRolloutsMultiplier = 2.0f;
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
    if (numChild_ == 0) {
      return 0.0;
    } else {
      return sumChildV_ / numChild_;
    }
  }

  float getAvgValue() const {
    assert(numVisit_ > 0);
    return value_ / numVisit_;
  }

  float getVirtualLoss() const {
    return virtualLoss_;
  }

  void addVirtualLoss(float virtualLoss) {
    // std::lock_guard<std::mutex> lock(mSelf_);
    virtualLoss_ += virtualLoss;
  }

  void atomicUpdate(float value, float virtualLoss) {
    // std::lock_guard<std::mutex> lock(mSelf_);
    value_ += value;
    numVisit_++;
    virtualLoss_ -= virtualLoss;
  }

  // Update child value estimate with a new obtained child value
  // (from the perspective of the root node
  void atomicUpdateChildV(float childV) {
    // std::lock_guard<std::mutex> lock(mSelf_);
    sumChildV_ += childV;
    numChild_++;
  }

  std::string summary() const {
    std::stringstream ss;
    ss << value_ << "/" << numVisit_ << " (" << value_ / numVisit_
       << "), vloss: " << virtualLoss_;
    return ss.str();
  }

  void subtractVisit() {
    --numVisit_;
  }
  void addVisit() {
    ++numVisit_;
  }

 private:
  float value_;
  int numVisit_;
  float virtualLoss_;

  // Summation of the value prediction from a child
  float sumChildV_;
  // # child that has been explored.
  int numChild_;

  // std::mutex mSelf_;
};

template <typename F, typename Rng>
size_t sampleDiscreteProbability(size_t nElements,
                                 float maxValue,
                                 F&& getValue,
                                 Rng& rng) {
  if (nElements == 0) {
    throw std::runtime_error("sampleDiscreteProbability was passed 0 elements");
  }
  for (size_t i = 0; i != 4; ++i) {
    size_t index = std::uniform_int_distribution<int>(0.0f, nElements - 1)(rng);
    if (std::generate_canonical<float, 20>(rng) <= getValue(index) / maxValue) {
      return index;
    }
  }
  thread_local std::vector<float> probs;
  probs.resize(nElements);
  float sum = 0.0f;
  for (size_t i = 0; i != nElements; ++i) {
    probs[i] = getValue(i);
    sum += probs[i];
  }
  float v = std::uniform_real_distribution<float>(0.0f, sum)(rng);
  return std::lower_bound(probs.begin(), std::prev(probs.end()), v) -
         probs.begin();
}

class MctsResult {
 public:
  MctsResult() = default;
  MctsResult(std::minstd_rand* rng)
      : maxVisits(-1000)
      , sumVisits(0)
      , bestAction(InvalidAction)
      , rng_(rng) {
  }

  void add(Action a, float visits) {
    if (mctsPolicy.size() <= (size_t)a) {
      if (mctsPolicy.capacity() <= (size_t)a) {
        mctsPolicy.reserve(mctsPolicy.size() * 2);
      }
      mctsPolicy.resize(a + 1);
    }
    mctsPolicy[a] = visits;
    sumVisits += visits;
    if (visits > maxVisits) {
      maxVisits = visits;
      bestAction = a;
    }
  }

  void normalize() {
    for (auto& value : mctsPolicy) {
      value = value / (float)sumVisits;
    }
  }

  // assume already normalized
  void sample() {
    auto weight = [this](float pival) {
      return std::exp(pival * pival * 2) - (1.0f - 0.5f / mctsPolicy.size());
    };
    float maxWeight = 0.0f;
    for (size_t i = 0; i != mctsPolicy.size(); ++i) {
      if (mctsPolicy[i] > maxWeight) {
        maxWeight = mctsPolicy[i];
      }
    }
    maxWeight = weight(maxWeight);
    bestAction = sampleDiscreteProbability(
        mctsPolicy.size(), maxWeight,
        [&](size_t i) { return weight(mctsPolicy[i]); }, *rng_);
  }

  void setMctsPolicy(std::vector<float> pi) {
    mctsPolicy = std::move(pi);
  }

  float maxVisits;
  float sumVisits;
  Action bestAction;
  std::vector<float> mctsPolicy;
  float rootValue = 0.0f;
  int rollouts = 0;
  torch::Tensor rnnState;

 private:
  std::minstd_rand* rng_;
};

using core::PiVal;

inline void printPolicy(const std::vector<float>& pi) {
  for (mcts::Action i = 0; i != (mcts::Action)pi.size(); ++i) {
    std::cout << i << ":" << pi[i] << std::endl;
  }
}

}  // namespace mcts
