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

#include "mcts/actor.h"
#include "mcts/node.h"
#include "mcts/player.h"
#include "mcts/storage.h"
#include "mcts/utils.h"

namespace mcts {

inline std::atomic_uint64_t rolloutCount;

void computeRollouts(const std::vector<Node*>& rootNode,
                     const std::vector<const State*>& rootState,
                     Actor& actor,
                     const MctsOption& option,
                     double thisMoveTime,
                     std::minstd_rand& rng);

class MctsPlayer : public Player {
 public:
  MctsPlayer(const MctsOption& option)
      : Player(false)
      , option_(option)
      , rng_(option.seed)
      , storage_(option.storageCap) {
    reset();
  }

  void addActor(std::shared_ptr<Actor> actor) {
    actors_.push_back(actor);
  }

  virtual void reset() override {
    remaining_time = option_.totalTime;
  }

  void newEpisode() override {
  }

  void recordMove(const State* state) override {
    actors_[0]->recordMove(state);
  }

  void result(const State* state, float reward) override {
    actors_[0]->result(state, reward);
  }

  bool isTournamentOpponent() const {
    return actors_[0]->isTournamentOpponent();
  }

  std::vector<MctsResult> actMcts(const std::vector<const State*>& states) {
    std::vector<MctsResult> result(states.size(), &rng_);

    auto begin = std::chrono::steady_clock::now();
    uint64_t beginRolloutCount = rolloutCount;

    // prior only
    if (!option_.useMcts) {
      for (size_t i = 0; i != states.size(); ++i) {
        PiVal piVal = actors_[0]->evaluate(*states[i]);
        result[i].setMctsPolicy(std::move(piVal.policy));
      }
    } else {
      std::vector<Node*> roots;
      for (auto* state : states) {
        Node* rootNode = storage_.newNode();
        rootNode->init(nullptr,
                       option_.storeStateInNode ? state->clone() : nullptr,
                       state->getHash());
        roots.push_back(rootNode);
      }

      double thisMoveTime = remaining_time * option_.timeRatio;
      bool forceSingleActor = false;
      if (option_.totalTime > 0) {
        std::cerr << "Remaining time:" << remaining_time << std::endl;
        std::cerr << "This move time:" << thisMoveTime << std::endl;
        if (remaining_time < 1 || thisMoveTime < 0.25) {
          forceSingleActor = true;
        }
      }
      auto begin = std::chrono::steady_clock::now();
      if (actors_.size() == 1 || forceSingleActor) {
        computeRollouts(
            roots, states, *actors_[0], option_, thisMoveTime, rng_);
      } else {
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < actors_.size(); ++i) {
          std::future<void> fut = std::async(std::launch::async,
                                             &computeRollouts,
                                             std::ref(roots),
                                             std::ref(states),
                                             std::ref(*actors_[i]),
                                             std::ref(option_),
                                             thisMoveTime,
                                             std::ref(rng_));
          futures.push_back(std::move(fut));
        }

        for (size_t i = 0; i < futures.size(); ++i) {
          futures[i].get();
        }
      }
      if (option_.totalTime) {
        auto end = std::chrono::steady_clock::now();
        remaining_time -=
            std::chrono::duration_cast<
                std::chrono::duration<double, std::ratio<1, 1>>>(end - begin)
                .count();
      }
      for (size_t i = 0; i != states.size(); ++i) {
        Node* rootNode = roots[i];
        assert(rootNode->getMctsStats().getVirtualLoss() == 0);
        if (option_.totalTime > 0) {
          std::cerr << "Value : " << rootNode->getMctsStats().getValue()
                    << " total rollouts : "
                    << rootNode->getMctsStats().getNumVisit() << std::endl;
          std::cerr << "Current value is (-1 to 1): "
                    << rootNode->getMctsStats().getValue() /
                           rootNode->getMctsStats().getNumVisit()
                    << std::endl;
        }
        result[i].rootValue = rootNode->getMctsStats().getAvgValue();
        for (const auto& pair : rootNode->getChildren()) {
          int visits = 0;
          for (size_t u = 0; u < pair.second.size(); u++) {
            visits += pair.second[u]->getMctsStats().getNumVisit();
          }
          result[i].add(pair.first, visits);
        }
        result[i].normalize();
        // rootNode->printTree(0, 2, -1);
        rootNode->freeTree();
      }
    }
    for (size_t i = 0; i != states.size(); ++i) {
      if (states[i]->getStepIdx() < option_.sampleBeforeStepIdx) {
        // std::cout << "sample:" << std::endl;
        result[i].sample();
      }
    }

    uint64_t n = rolloutCount - beginRolloutCount;
    double s = std::chrono::duration_cast<
                   std::chrono::duration<double, std::ratio<1, 1>>>(
                   std::chrono::steady_clock::now() - begin)
                   .count();
    rolloutsPerSecond_ = n / s;

    return result;
  }

  virtual void terminate() override {
    for (auto& v : actors_) {
      v->terminate();
    }
  }

  MctsResult actMcts(const State& state) {
    return actMcts({&state})[0];
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

  float calculateValue(const State& state) {
    PiVal result =actors_.at(0)->evaluate(state);
    return result.value;
  }

 private:
  MctsOption option_;
  double remaining_time;
  std::vector<std::shared_ptr<Actor>> actors_;
  std::minstd_rand rng_;
  Storage storage_;
  double rolloutsPerSecond_ = 0.0;
};
}  // namespace mcts
