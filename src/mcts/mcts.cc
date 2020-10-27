/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mcts/mcts.h"
#include "common/async.h"
#include "common/thread_id.h"
#include "common/threads.h"
#include "core/state.h"

#include <chrono>

namespace mcts {

int forcedRollouts(float piValue, int numVisits, const MctsOption& option) {
  return (int)std::sqrt(option.forcedRolloutsMultiplier * piValue * numVisits);
}

// TODO: eliminate duplicate code shared with pickBestAction below
float puctValue(int rootPlayerId,
                float puct,
                const Node* node,
                mcts::Action action) {
  const auto& pi = node->legalPolicy_;
  const Node* child = node->getChild(action);
  auto childNumVisit = child->getMctsStats().getNumVisit();
  float piValue = pi[action];
  auto parentNumVisit = node->getMctsStats().getNumVisit();
  float priorScore =
      (float)piValue / (1 + childNumVisit) * (float)std::sqrt(parentNumVisit);
  int flip = (node->getPiVal().playerId == rootPlayerId) ? 1 : -1;
  float value = child->getMctsStats().getValue();
  float vloss = child->getMctsStats().getVirtualLoss();
  float q = (value * flip - vloss) / (childNumVisit + vloss);
  float score = priorScore * puct + q;
  return score;
}

template <bool sample>
Action pickBestAction(int rootPlayerId,
                      const Node* const node,
                      const MctsOption& option,
                      std::minstd_rand& rng,
                      int maxNumRollouts) {
  const auto& pi = node->legalPolicy_;
  if (pi.empty()) {
    return InvalidAction;
  }

  float puct = option.puct;
  bool useValuePrior = option.useValuePrior;
  // We need to flip here because at opponent's step, we need to find
  // opponent's best action which minimizes our value.  Careful not to
  // flip the exploration term.
  int flip = (node->getPiVal().playerId == rootPlayerId) ? 1 : -1;
  float priorValue = node->getMctsStats().getAvgChildV() * flip;

  auto getScore = [&](Action actionIndex, const Node* child) {
    float q = 0;
    int childNumVisit = 0;
    float vloss = 0;
    float value = 0;

    float piValue = pi[actionIndex];
    auto parentNumVisit = node->getMctsStats().getNumVisit();

    if (child) {
      const MctsStats& mctsStats = child->getMctsStats();
      childNumVisit += mctsStats.getNumVisit();
      vloss += mctsStats.getVirtualLoss();
      value += mctsStats.getValue();
    }
    if (childNumVisit != 0) {
      q = (value * flip - vloss) / (childNumVisit + vloss);
    } else {
      // When there are no child nodes under this action, replace the q value
      // with prior.
      // This prior is estimated from the values of other explored child.
      // q = 0 if this is the first child to be explroed. In this case, all q =
      // 0 and we start with the child with highest policy probability.
      if (useValuePrior) {
        q = priorValue;
      }
    }

    float priorScore =
        (float)piValue / (1 + childNumVisit) * (float)std::sqrt(parentNumVisit);
    return priorScore * puct + q;
  };

  if (option.forcedRolloutsMultiplier && !node->getParent()) {
    // Forced rollouts; this only happens at the root node
    int maxForcedRollouts = forcedRollouts(1.0f, maxNumRollouts, option);
    for (auto& v : node->getChildren()) {
      Action actionIndex = v.first;
      const Node* child = v.second;
      int childNumVisit = child->getMctsStats().getNumVisit();
      if (childNumVisit < maxForcedRollouts &&
          childNumVisit <
              forcedRollouts(pi[actionIndex], maxNumRollouts, option)) {
        return actionIndex;
      }
    }
  }

  if (sample) {
    return sampleDiscreteProbability(pi.size(), 1.0f,
                                     [&](size_t index) {
                                       float score = getScore(
                                           index, node->getChild(index));
                                       return std::exp(score * 4);
                                     },
                                     rng);
  } else {
    float bestScore = -std::numeric_limits<float>::infinity();
    Action bestAction = InvalidAction;

    for (mcts::Action actionIndex = 0; actionIndex != (mcts::Action)pi.size();
         ++actionIndex) {
      const Node* child = node->getChild(actionIndex);

      float score = getScore(actionIndex, child);
      if (score > bestScore) {
        bestScore = score;
        bestAction = actionIndex;
      }
    }
    return bestAction;
  }
}

namespace {

std::atomic_uint64_t rolloutCount;

std::chrono::steady_clock::time_point starttime;
bool started = false;

}  // namespace

int computeRolloutsImpl(const std::vector<Node*>& rootNode,
                        const std::vector<const core::State*>& rootState,
                        const std::vector<torch::Tensor>& rnnState,
                        core::Actor& actor,
                        const MctsOption& option,
                        double max_time,
                        std::minstd_rand& rng) {

  double elapsedTime = 0;
  auto begin = std::chrono::steady_clock::now();

  struct RolloutState {
    Node* root = nullptr;
    Node* node = nullptr;
    std::unique_ptr<core::State> state;
    bool terminated = false;
    Storage* storage = nullptr;
    torch::Tensor rnnState;

    Node* forcedParent = nullptr;
    Action forcedAction = InvalidAction;
  };

  std::vector<RolloutState> states(rootNode.size());

  async::Task task(threads::threads);

  size_t stride =
      (states.size() + threads::threads.size() - 1) / threads::threads.size();

  std::vector<async::Thread*> reservedThreads(states.size());
  for (size_t i = 0; i < states.size(); i += stride) {
    reservedThreads[i] = &threads::threads.getThread();
  }

  int numRollout = 0;
  std::vector<async::Handle> functionHandles(states.size());

  int rollouts = option.totalTime ? 0 : option.numRolloutPerThread;

  bool keepGoing = false;

  for (size_t i = 0; i < states.size(); i += stride) {
    rng.discard(1);
    size_t n = std::min(states.size() - i, stride);

    auto f = [&, ii = i, n, rng]() mutable {
      size_t i = ii;

      for (size_t s = 0; s != n; ++s, ++i) {

        auto& st = states[i];
        Node* root = rootNode[i];
        st.root = root;

        if (!st.storage) {
          st.storage = Storage::getStorage();
        }
        Storage* storage = st.storage;

        if (numRollout != 0) {
          Node* node = st.node;
          if (!st.terminated) {
            auto& state = *st.state;
            actor.batchResult(i, state, node->piVal_);
            core::getLegalPi(
                state, node->piVal_.logitPolicy, node->legalPolicy_);
            core::softmax_(node->legalPolicy_);
            node->piVal_.logitPolicy.reset();
          }

          node->settle(st.root->getPiVal().playerId);

          float value = node->getPiVal().value;
          int flip = st.root->getPiVal().playerId == node->getPiVal().playerId
                         ? 1
                         : -1;
          value = value * flip;
          // We need to flip here because at opponent's node, we have opponent's
          // value. We need to sum up our value.
          while (node != nullptr) {
            MctsStats& mctsStats = node->getMctsStats();
            mctsStats.atomicUpdate(value, 0.0f);
            node = node->getParent();
          }
        }

        if (!keepGoing) {
          continue;
        }

        Node* node = root;
        std::unique_ptr<core::State> localState = std::move(st.state);
        const core::State* src =
            st.forcedParent ? &*st.forcedParent->localState() : rootState[i];
        if (!src) {
          throw std::runtime_error("src state is null");
        }
        if (!localState) {
          localState = src->clone();
        } else {
          localState->copy(*src);
        }

        const torch::Tensor* rsp = nullptr;
        if (!rnnState.empty()) {
          rsp = &rnnState[i];
        }

        // 1. Selection

        thread_local std::vector<Action> queuedActions;
        queuedActions.clear();

        const core::State* checkpointState = nullptr;

        auto flushActions = [&]() {
          if (checkpointState) {
            localState->copy(*checkpointState);
          }
          if (!queuedActions.empty()) {
            for (Action a : queuedActions) {
              localState->forward(a);
            }
            queuedActions.clear();
          }
        };

        Node* parent = nullptr;
        Action action = InvalidAction;

        bool save = false;

        if (st.forcedParent) {
          parent = st.forcedParent;
          action = st.forcedAction;
          st.forcedParent = nullptr;

          node = parent->newChild(storage->newNode(), action);

          auto& state = *localState;

          if ((size_t)action >= state.GetLegalActions().size()) {
            throw std::runtime_error("forced rollout bad action :((");
          }

        } else if (node->isVisited()) {
          while (true) {
            rsp = &node->piVal_.rnnState;

            Action bestAction =
                (option.samplingMcts
                     ? pickBestAction<true>
                     : pickBestAction<false>)(root->getPiVal().playerId, node,
                                              option, rng, rollouts);
            // this is a terminal state that has been visited
            if (bestAction == InvalidAction) {
              flushActions();
              break;
            }

            Node* childNode = node->getChild(bestAction);
            if (childNode) {
              node = childNode;
              if (node->hasState()) {
                checkpointState = &node->getState();
                queuedActions.clear();
              } else {
                queuedActions.push_back(bestAction);
              }
              continue;
            }
            save = queuedActions.size() >= (size_t)option.storeStateInterval;
            flushActions();

            childNode = node->newChild(storage->newNode(), bestAction);

            action = bestAction;
            parent = node;
            node = childNode;
            break;
          }

          auto& state = *localState;

          if (node->isVisited() && !state.terminated()) {
            if (state.GetLegalActions().empty()) {
              throw std::runtime_error(
                  "MCTS error - no legal actions in unterminated game state");
            }
            throw std::runtime_error("MCTS error - rollout ended on unvisited "
                                     "node with unterminated game state");
          }
        }

        auto& state = *localState;

        auto saveState = [&](Node* saveNode) {
          if (saveNode->localState() &&
              saveNode->localState()->typeId() == state.typeId()) {
            core::State* dst = &*saveNode->localState();
            dst->copy(state);
            saveNode->setState(dst);
          } else {
            saveNode->localState() = localState->clone();
            saveNode->setState(&*saveNode->localState());
          }
        };

        if (parent) {
          // Force visits to any children that share this policy output
          // location.
          const _Action& a = state.GetLegalActions().at(action);
          for (auto& x : state.GetLegalActions()) {
            if (x.GetIndex() != action && x.GetX() == a.GetX() &&
                x.GetY() == a.GetY() && x.GetZ() == a.GetZ()) {
              if (!parent->getChild(x.GetIndex())) {
                st.forcedParent = parent;
                st.forcedAction = x.GetIndex();

                if (!parent->hasState()) {
                  saveState(parent);
                }
                break;
              }
            }
          }

          localState->forward(action);

          if (save) {
            saveState(node);
          }
        }

        // 2. Expansion

        if (state.terminated()) {
          PiVal& piVal = node->piVal_;
          piVal.value = state.getReward(state.getCurrentPlayer()) * 2.0f;
          piVal.playerId = state.getCurrentPlayer();

          st.terminated = true;
        } else {
          st.terminated = false;
        }

        st.node = node;
        st.state = std::move(localState);
        actor.batchPrepare(i, state, rsp ? *rsp : torch::Tensor());
      }
    };

    functionHandles[i] = task.getHandle(*reservedThreads[i], std::move(f));
    functionHandles[i].setPriority(common::getThreadId());
  }

  actor.batchResize(states.size());

  if (option.randomizedRollouts && rollouts >= 4) {
    float mean = std::uniform_int_distribution<int>(0, 3)(rng) != 0
                     ? rollouts / 8.0f
                     : rollouts * 2.0f;
    std::normal_distribution<float> r(mean, rollouts / 4.0f);
    int max = rollouts * 4;
    do {
      rollouts = r(rng);
    } while (rollouts < 1 || rollouts > max);
  }

  while (true) {
    keepGoing = (option.totalTime ? elapsedTime < max_time
                                  : numRollout < option.numRolloutPerThread) ||
                numRollout < 2;

    for (size_t i = 0; i < states.size(); i += stride) {
      task.enqueue(functionHandles[i]);
    }

    task.wait();
    if (!keepGoing) {
      break;
    }
    actor.batchEvaluate(states.size());

    rolloutCount += states.size();

    ++numRollout;
    auto end = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1>>>(end - begin)
            .count();
  }

  for (const Node* root : rootNode) {
    mcts::Action bestAction = -1;
    int best = 0;
    for (auto& v : root->getChildren()) {
      const Node* child = v.second;
      if (child->getMctsStats().getNumVisit() > best) {
        best = child->getMctsStats().getNumVisit();
        bestAction = v.first;
      }
    }
    if (bestAction != -1) {
      float bestPuct =
          puctValue(root->getPiVal().playerId, option.puct, root, bestAction);
      for (auto& v : root->getChildren()) {
        if (v.first == bestAction) {
          continue;
        }
        Node* child = v.second;
        int forced =
            forcedRollouts(root->legalPolicy_[v.first], rollouts, option);
        for (; forced && child->getMctsStats().getNumVisit(); --forced) {
          child->getMctsStats().subtractVisit();
          float pv =
              puctValue(root->getPiVal().playerId, option.puct, root, v.first);
          if (pv > bestPuct) {
            child->getMctsStats().addVisit();
            break;
          }
        }
      }
    }
  }

  return rollouts;
}

int computeRollouts(const std::vector<Node*>& rootNode,
                    const std::vector<const core::State*>& rootState,
                    const std::vector<torch::Tensor>& rnnState,
                    core::Actor& actor,
                    const MctsOption& option,
                    double max_time,
                    std::minstd_rand& rng) {

  return computeRolloutsImpl(
      rootNode, rootState, rnnState, actor, option, max_time, rng);
}

std::vector<MctsResult> MctsPlayer::actMcts(
    const std::vector<const core::State*>& states,
    const std::vector<torch::Tensor>& rnnState) {
  std::vector<MctsResult> result(states.size(), &rng_);

  auto begin = std::chrono::steady_clock::now();
  uint64_t beginRolloutCount = rolloutCount;

  if (!started) {
    started = true;
    starttime = begin;
  }

  std::vector<Node*> roots;
  Storage* storage = Storage::getStorage();
  for (auto* state : states) {
    Node* rootNode = storage->newNode();
    rootNode->init(nullptr);
    roots.push_back(rootNode);

    if (state->terminated()) {
      throw std::runtime_error("Attempt to run MCTS from terminated state");
    }
  }

  double thisMoveTime = remaining_time * option_.timeRatio;
  if (option_.totalTime) {
    std::cerr << "Remaining time:" << remaining_time << std::endl;
    std::cerr << "This move time:" << thisMoveTime << std::endl;
  }
  int rollouts = computeRollouts(
      roots, states, rnnState, *actor_, option_, thisMoveTime, rng_);
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
    result[i].normalize();
  }

  for (size_t i = 0; i != states.size(); ++i) {
    if (result[i].bestAction == InvalidAction) {
      throw std::runtime_error(
          "MCTS could not find any valid actions at state " +
          states[i]->history());
    }
    if (states[i]->getStepIdx() < option_.sampleBeforeStepIdx) {
      // std::cout << "sample:" << std::endl;
      result[i].sample();
    }
  }

  if (option_.useMcts) {
    for (size_t i = 0; i != states.size(); ++i) {
      auto* n = roots[i]->getChild(result[i].bestAction);
      if (n && n->getPiVal().rnnState.defined()) {
        result[i].rnnState = n->getPiVal().rnnState;
      }
      roots[i]->freeTree();
    }
  }

  uint64_t n = rolloutCount - beginRolloutCount;
  double s = std::chrono::duration_cast<
                 std::chrono::duration<double, std::ratio<1, 1>>>(
                 std::chrono::steady_clock::now() - begin)
                 .count();
  rolloutsPerSecond_ = n / s;

  printf("rollouts per second: %g\n", rolloutsPerSecond_);

  double sx = std::chrono::duration_cast<
                  std::chrono::duration<double, std::ratio<1, 1>>>(
                  std::chrono::steady_clock::now() - starttime)
                  .count();
  printf("total rollouts per second: %g\n", rolloutCount / sx);

  return result;
}

}  // namespace mcts
