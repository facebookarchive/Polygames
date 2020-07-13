/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mcts/mcts.h"
#include <chrono>

using namespace mcts;

Action pickBestAction(int rootPlayerId,
                      const Node* const node,
                      float puct,
                      bool useValuePrior,
                      std::minstd_rand& rng) {
  float bestScore = -1e10;
  // We need to flip here because at opponent's step, we need to find
  // opponent's best action which minimizes our value.  Careful not to
  // flip the exploration term.
  int flip = (node->getPiVal().playerId == rootPlayerId) ? 1 : -1;
  // std::cout << "flip is " << flip << std::endl;
  Action bestAction = InvalidAction;
  const auto& pi = node->getPiVal().policy;
  for (const auto& pair : pi) {
    // std::cout << "*";
    const auto& childList = node->getChild(pair.first);

    float q = 0;
    int childNumVisit = 0;
    float vloss = 0;
    float value = 0;

    for (size_t i = 0; i < childList.size(); ++i) {
      Node* child = childList[i];
      const MctsStats& mctsStats = child->getMctsStats();
      childNumVisit += mctsStats.getNumVisit();
      vloss += mctsStats.getVirtualLoss();
      value += mctsStats.getValue();
    }
    if (childNumVisit != 0) {
      q = (value * flip - vloss) / (childNumVisit + vloss);
    } else {
      // When there is no child nodes under this action, replace the q value
      // with prior.
      // This prior is estimated from the values of other explored child.
      // q = 0 if this is the first child to be explroed. In this case, all q =
      // 0 and
      // we start with the child with highest policy probability.
      if (useValuePrior) {
        q = node->getMctsStats().getAvgChildV() * flip;
      }
    }

    auto parentNumVisit = node->getMctsStats().getNumVisit();
    float priorScore = (float)pair.second / (1 + childNumVisit) *
                       (float)std::sqrt(parentNumVisit);
    float score = priorScore * puct + q;
    if (score > bestScore) {
      bestScore = score;
      bestAction = pair.first;
    }
  }
  // std::cout << "best score is " << bestScore << std::endl;
  // assert(false);
  return bestAction;
}

void mcts::computeRollouts(const std::vector<Node*>& rootNode,
                           const std::vector<const State*>& rootState,
                           Actor& actor,
                           const MctsOption& option,
                           double max_time,
                           std::minstd_rand& rng) {
  int numRollout = 0;
  // int rootPlayerId = rootNode->getPiVal().playerId;

  struct RolloutState {
    Node* root = nullptr;
    Node* node = nullptr;
    std::unique_ptr<State> state;
    const State* rootState = nullptr;
  };

  std::vector<RolloutState> states(rootNode.size());

  std::vector<const State*> batch;
  batch.reserve(states.size());

  double elapsedTime = 0;
  auto begin = std::chrono::steady_clock::now();

  while ((option.totalTime ? elapsedTime < max_time
                           : numRollout < option.numRolloutPerThread) ||
         numRollout < 2) {
    // std::cout << " new rollout
    // ===================================================" << std::endl;
    for (size_t i = 0; i != rootNode.size(); ++i) {
      auto& st = states[i];
      st.root = rootNode[i];
      st.node = st.root;
      if (!option.storeStateInNode) {
        // std::cout << " clone " << std::endl;
        st.state = rootState[i]->clone();
      }
      st.rootState = rootState[i];
    }

    // 1.Selection

    for (auto& st : states) {
      int depth = 0;
      while (true) {
        ++depth;
        st.node->acquire();
        st.node->getMctsStats().addVirtualLoss(option.virtualLoss);
        if (!st.node->isVisited()) {
          // std::cout << " not visited" << std::endl;
          break;
        }

        // If we have policy network then value prior collected at
        // different nodes are more accurate. Otherwise it could harm the
        // exploration.
        Action bestAction = pickBestAction(st.root->getPiVal().playerId,
                                           st.node,
                                           option.puct,
                                           option.useValuePrior,
                                           rng);
        // this is a terminal state that has been visited
        if (bestAction == InvalidAction) {
          // std::cout << " invalid action " << std::endl;
          break;
        }
        bool stochasticFather = st.state->isStochastic();
        if (!option.storeStateInNode) {
          // std::cout << " forward" << std::endl;
          st.state->forward(bestAction);
          // std::cout << " forward done" << std::endl;
        }

        uint64_t hash = 0;
        if (st.state != nullptr) {
          hash = st.state->getHash();
        }

        // std::cout << " goac" << std::endl;
        Node* childNode =
            st.node->getOrAddChild(bestAction,
                                   option.storeStateInNode,
                                   stochasticFather,
                                   // st.rootState->isStochastic(),
                                   hash);
        st.node->release();
        st.node = childNode;
        // std::cout << " endloop " << std::endl;
      }
    }

    // 2. Expansion

    batch.clear();
    size_t e = states.size();
    for (size_t i = 0; i != e; ++i) {
      auto& st = states[i];
      if (st.state->terminated()) {
        PiVal piVal;
        piVal.value = st.state->getReward(st.state->getCurrentPlayer());
        piVal.playerId = st.state->getCurrentPlayer();

        st.node->settle(st.root->getPiVal().playerId, piVal);
        st.node->release();
        std::swap(st, states[e - 1]);
        --i;
        --e;
      } else {
        batch.push_back(&*st.state);
      }
    }

    if (!batch.empty()) {
      actor.evaluate(batch, [&](size_t index, PiVal piVal) {
        auto& st = states[index];
        st.node->settle(st.root->getPiVal().playerId, piVal);
        st.node->release();
      });
    }

    // 3.Backpropgation

    for (size_t i = 0; i != states.size(); ++i) {
      auto& st = states[i];
      float value = st.node->getPiVal().value;
      int flip = (st.root->getPiVal().playerId == st.node->getPiVal().playerId)
                     ? 1
                     : -1;
      value = value * flip;
      // We need to flip here because at opponent's node, we have opponent's
      // value. We need to sum up our value.
      while (st.node != nullptr) {
        MctsStats& mctsStats = st.node->getMctsStats();
        mctsStats.atomicUpdate(value, option.virtualLoss);
        st.node = st.node->getParent();
      }
    }

    rolloutCount += states.size();

    ++numRollout;
    auto end = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1>>>(end - begin)
            .count();
  }
}
