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

inline std::atomic_uint64_t rolloutCount;

int computeRollouts(const std::vector<Node*>& rootNode,
                    const std::vector<const State*>& rootState,
                    const std::vector<std::vector<float>>& rnnState,
                    Actor& actor,
                    const MctsOption& option,
                    double thisMoveTime,
                    std::minstd_rand& rng,
                    const std::vector<std::vector<float>>& policyBias = {});

class MctsPlayer : public Player {
 public:
  MctsPlayer(const MctsOption& option)
      : Player(false)
      , option_(option)
      , rng_(option.seed)
  //, storage_(option.storageCap)
  {
    remaining_time = option.totalTime;
  }

  void addActor(std::shared_ptr<Actor> actor) {
    actors_.push_back(actor);
  }

  void newEpisode() override {
  }

  void recordMove(const State* state) override {
    actors_[0]->recordMove(state);
  }

  void result(const State* state, float reward) override {
    actors_[0]->result(state, reward);
  }

  void forget(const State* state) override {
    actors_[0]->forget(state);
  }

  bool isTournamentOpponent() const {
    return actors_[0]->isTournamentOpponent();
  }

  bool wantsTournamentResult() const {
    return actors_[0]->wantsTournamentResult();
  }

  double batchTiming() const {
    return actors_[0]->batchTiming();
  }

  std::string getModelId() const {
    return actors_[0]->getModelId();
  }

  std::vector<MctsResult> actMcts(
      const std::vector<const State*>& states,
      const std::vector<std::vector<float>>& rnnState,
      const std::vector<PersistentTree*>& persistentTrees,
      const std::vector<std::vector<float>>& policyBias = {}) {
    std::vector<MctsResult> result(states.size(), &rng_);

    auto begin = std::chrono::steady_clock::now();
    uint64_t beginRolloutCount = rolloutCount;

    std::vector<Node*> roots;
    // prior only
    if (!option_.useMcts) {
      for (size_t i = 0; i != states.size(); ++i) {
        PiVal piVal;
        actors_[0]->evaluate(*states[i], piVal);
        result[i].setMctsPolicy(std::move(piVal.policy));
      }
    } else {
      if (!persistentTrees.empty()) {
        for (size_t i = 0; i != states.size(); ++i) {
          auto& t = *persistentTrees[i];
          auto& s = *states[i];
          if (!t.root) {
            if (!s.getMoves().empty()) {
              throw std::runtime_error("Refusing to create new persistent tree "
                                       "with moves already made");
            }
            t.storage = Storage::getStorage();
            t.root = t.storage->newNode();
            t.root->init(nullptr);
            roots.push_back(t.root);
          } else {
            // printf("new rollout root %p\n", t.root);
            Node* n = t.root;
            for (auto x : s.getMoves()) {
              n = n->getChild(x);
              // printf("tree child %d\n", x);
              if (!n) {
                t.root->printTree(0, 2, -1);
                throw std::runtime_error("Child not found in persistent tree!");
              }
            }
            roots.push_back(n);
          }
        }
      } else {
        Storage* storage = Storage::getStorage();
        for (auto* state : states) {
          Node* rootNode = storage->newNode();
          rootNode->init(nullptr);
          roots.push_back(rootNode);

          if (state->terminated()) {
            throw std::runtime_error(
                "Attempt to run MCTS from terminated state");
          }
        }
      }

      double thisMoveTime = remaining_time * option_.timeRatio;
      if (option_.totalTime) {
        std::cerr << "Remaining time:" << remaining_time << std::endl;
        std::cerr << "This move time:" << thisMoveTime << std::endl;
      }
      auto begin = std::chrono::steady_clock::now();
      int rollouts = 0;
      if (actors_.size() == 1) {
        rollouts = computeRollouts(roots,
                                   states,
                                   rnnState,
                                   *actors_[0],
                                   option_,
                                   thisMoveTime,
                                   rng_,
                                   policyBias);
      } else {
        std::vector<std::future<int>> futures;
        for (size_t i = 0; i < actors_.size(); ++i) {
          std::future<int> fut = std::async(std::launch::async,
                                            &computeRollouts,
                                            std::ref(roots),
                                            std::ref(states),
                                            std::ref(rnnState),
                                            std::ref(*actors_[i]),
                                            std::ref(option_),
                                            thisMoveTime,
                                            std::ref(rng_),
                                            std::vector<std::vector<float>>());
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
        result[i].rollouts = rollouts;
        result[i].rootValue = rootNode->getMctsStats().getAvgValue();
        if (option_.moveSelectUseMctsValue) {
          for (auto& v : rootNode->getChildren()) {
            result[i].add(v.first, 1 + v.second->getMctsStats().getAvgValue());
          }
        } else {
          for (auto& v : rootNode->getChildren()) {
            int visits = v.second->getMctsStats().getNumVisit();
            if (visits > 1) {
              result[i].add(v.first, visits);
            }
          }
          if (result[i].bestAction == InvalidAction) {
            for (auto& v : rootNode->getChildren()) {
              int visits = v.second->getMctsStats().getNumVisit();
              result[i].add(v.first, visits);
            }
          }
        }
        //        for (auto& v1 : rootNode->getChildren()) {
        //          for (auto& v : v1) {
        //            int visits = v.second->getMctsStats().getNumVisit();
        //            result[i].add(v.first, visits);
        //          }
        //        }
        result[i].normalize();
        // rootNode->printTree(0, 2, -1);
      }
    }
    for (size_t i = 0; i != states.size(); ++i) {
      if (result[i].bestAction == InvalidAction) {
        throw std::runtime_error("MCTS could not find any valid actions");
      }
      if (states[i]->getStepIdx() < option_.sampleBeforeStepIdx) {
        // std::cout << "sample:" << std::endl;
        result[i].sample();
      }
    }

    if (option_.useMcts) {
      for (size_t i = 0; i != states.size(); ++i) {
        auto* n = roots[i]->getChild(result[i].bestAction);
        if (n && !n->getPiVal().rnnState.empty()) {
          result[i].rnnState = n->getPiVal().rnnState;
        }
        if (persistentTrees.empty()) {
          roots[i]->freeTree();
        }
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
    return actMcts({&state}, {}, {}).at(0);
  }

  MctsResult actMcts(const State& state, const std::vector<float>& rnnState) {
    return actMcts({&state}, {rnnState}, {}).at(0);
  }

  double rolloutsPerSecond() {
    return rolloutsPerSecond_;
  }

  const MctsOption& option() {
    return option_;
  }

  float calculateValue(const State& state) {
    PiVal result;
    actors_.at(0)->evaluate(state, result);
    return result.value;
  }

  std::vector<std::vector<float>> nextRnnState(
      const std::vector<const State*>& state,
      const std::vector<std::vector<float>>& rnnState) {
    std::vector<std::vector<float>> r;
    PiVal result;
    actors_.at(0)->batchResize(state.size());
    for (size_t i = 0; i != state.size(); ++i) {
      actors_.at(0)->batchPrepare(i, *state[i], rnnState.at(i));
    }
    actors_.at(0)->batchEvaluate(state.size());
    for (size_t i = 0; i != state.size(); ++i) {
      actors_.at(0)->batchResult(i, *state[i], result);
      r.push_back(std::move(result.rnnState));
    }
    return r;
  }

  std::vector<int64_t> rnnStateSize() const {
    return actors_[0]->rnnStateSize();
  }

  virtual int rnnSeqlen() const override {
    return actors_[0]->rnnSeqlen();
  }

 private:
  MctsOption option_;
  double remaining_time;
  std::vector<std::shared_ptr<Actor>> actors_;
  std::minstd_rand rng_;
  // Storage storage_;
  double rolloutsPerSecond_ = 0.0;
};
}  // namespace mcts
