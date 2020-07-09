/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mcts/mcts.h"
#include <chrono>

#include "async.h"

namespace tube {
inline thread_local int threadId;
}

namespace mcts {

std::mutex freeStoragesMutex;
std::list<Storage*> freeStorages;

int forcedRollouts(float piValue, int numVisits, const MctsOption& option) {
  return (int)std::sqrt(option.forcedRolloutsMultiplier * piValue * numVisits);
}

float puctValue(int rootPlayerId,
                float puct,
                const Node* node,
                mcts::Action action) {
  const auto& pi = node->getPiVal().policy;
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
                      int maxnumrollouts) {
  float bestScore = -1e10;

  float puct = option.puct;
  bool useValuePrior = option.useValuePrior;

  // We need to flip here because at opponent's step, we need to find
  // opponent's best action which minimizes our value.  Careful not to
  // flip the exploration term.
  int flip = (node->getPiVal().playerId == rootPlayerId) ? 1 : -1;
  Action bestAction = InvalidAction;
  const auto& pi = node->getPiVal().policy;
  //  if (pi.empty()) return InvalidAction;
  //  return std::uniform_int_distribution<size_t>(0, pi.size() - 1)(rng);
  float priorValue = node->getMctsStats().getAvgChildV() * flip;
  for (mcts::Action actionIndex = 0; actionIndex != pi.size(); ++actionIndex) {
    const Node* child = node->getChild(actionIndex);

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
      if (option.forcedRolloutsMultiplier && !node->getParent() &&
          childNumVisit < forcedRollouts(piValue, maxnumrollouts, option)) {
        return actionIndex;
      }
      q = (value * flip - vloss) / (childNumVisit + vloss);
    } else {
      // When there is no child nodes under this action, replace the q value
      // with prior.
      // This prior is estimated from the values of other explored child.
      // q = 0 if this is the first child to be explroed. In this case, all q =
      // 0 and
      // we start with the child with highest policy probability.
      if (useValuePrior) {
        q = priorValue;
      }
    }

    float priorScore =
        (float)piValue / (1 + childNumVisit) * (float)std::sqrt(parentNumVisit);
    float score = priorScore * puct + q;
    if (sample) {
      score =
          std::uniform_real_distribution<float>(0.0f, std::exp(score * 4))(rng);
    }
    // score = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    if (score > bestScore) {
      bestScore = score;
      bestAction = actionIndex;
    }
  }
  return bestAction;
}

namespace {
AsyncThreads threads(10);

struct Timer {
  std::chrono::steady_clock::time_point start;
  Timer() {
    reset();
  }
  void reset() {
    start = std::chrono::steady_clock::now();
  }
  double elapsed() {
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::ratio<1, 1000>>>(
               std::chrono::steady_clock::now() - start)
        .count();
  }
  double elapsed_reset() {
    auto now = std::chrono::steady_clock::now();
    auto t = now - start;
    start = now;
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::ratio<1, 1000>>>(t)
        .count();
  }
};

}  // namespace

template <bool storeStateInNode>
int computeRolloutsImpl(const std::vector<Node*>& rootNode,
                        const std::vector<const State*>& rootState,
                        const std::vector<std::vector<float>>& rnnState,
                        Actor& actor,
                        const MctsOption& option,
                        double max_time,
                        std::minstd_rand& rng,
                        const std::vector<std::vector<float>>& policyBias) {

  double elapsedTime = 0;
  auto begin = std::chrono::steady_clock::now();

  struct RolloutState {
    Node* root = nullptr;
    Node* node = nullptr;
    std::unique_ptr<State> state;
    bool terminated = false;
    Storage* storage = nullptr;
    std::vector<float> rnnState;
  };

  std::vector<RolloutState> states(rootNode.size());

  AsyncTask task(threads);

  size_t stride =
      (states.size() + threads.threads.size() - 1) / threads.threads.size();

  std::vector<AsyncThreads::Thread*> reservedThreads(states.size());
  for (size_t i = 0; i < states.size(); i += stride) {
    reservedThreads[i] = &threads.getThread();
  }

  int numRollout = 0;

  double enqueueTime = 0.0f;
  double waitTime = 0.0f;
  double otherTime = 0.0f;
  double setupTime = 0.0f;
  double evaluateTime = 0.0f;
  double backpropTime = 0.0f;

  std::atomic<double> sumftimes = 0.0f;
  std::atomic_int nftimes = 0;

  Timer timer;

  std::vector<AsyncThreads::Thread::Handle> functionHandles(states.size());

  int rollouts = option.totalTime ? 0 : option.numRolloutPerThread;

  bool keepGoing = false;

  for (size_t i = 0; i < states.size(); i += stride) {
    rng.discard(1);
    size_t n = std::min(states.size() - i, stride);

    auto f = [&, ii = i, n, rng]() mutable {
      Timer ftimer;

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
            auto& state = storeStateInNode ? node->getState() : *st.state;
            actor.batchResult(i, state, node->piVal_);
            // node->piVal_.value += state.getReward(node->piVal_.playerId);
            if (node->getParent() == nullptr && !policyBias.empty() &&
                !policyBias.at(i).empty()) {
              auto& bias = policyBias[i];
              if (node->piVal_.policy.size() != bias.size()) {
                throw std::runtime_error(
                    "policyBias size mismatch, got " +
                    std::to_string(bias.size()) + ", expected " +
                    std::to_string(node->piVal_.policy.size()));
              }
              for (size_t i = 0; i != bias.size(); ++i) {
                node->piVal_.policy[i] += bias[i];
              }
            }
          }

          node->settle(st.root->getPiVal().playerId);
          node->release();

          float value = node->getPiVal().value;
          int flip = st.root->getPiVal().playerId == node->getPiVal().playerId
                         ? 1
                         : -1;
          value = value * flip;
          // We need to flip here because at opponent's node, we have opponent's
          // value. We need to sum up our value.
          while (node != nullptr) {
            MctsStats& mctsStats = node->getMctsStats();
            mctsStats.atomicUpdate(value, option.virtualLoss);
            node = node->getParent();
          }
        }

        if (!keepGoing) {
          continue;
        }

        Node* node = root;
        std::unique_ptr<State> localState = std::move(st.state);
        if (!storeStateInNode) {
          if (!localState) {
            localState = rootState[i]->clone();
          } else {
            localState->copy(*rootState[i]);
          }
        }

        const std::vector<float>* rsp = nullptr;
        if (!rnnState.empty()) {
          rsp = &rnnState[i];
        }

        // 1. Selection

        thread_local std::vector<Action> queuedActions;
        if (!storeStateInNode) {
          queuedActions.clear();
        }

        const State* checkpointState = nullptr;

        auto flushActions = [&]() {
          if (checkpointState) {
            localState->copy(*checkpointState);
            // printf("restored from checkpoint\n");
          }
          if (!queuedActions.empty()) {
            for (Action a : queuedActions) {
              localState->forward(a);
            }
            // printf("forwarded %d actions\n", queuedActions.size());
            // printf("ff -> %s\n", localState->history().c_str());
            queuedActions.clear();
          }
        };

        while (true) {
          node->acquire();
          node->getMctsStats().addVirtualLoss(option.virtualLoss);
          if (!node->isVisited()) {
            // printf("not visited\n");
            break;
          }

          rsp = &node->piVal_.rnnState;

          // If we have policy network then value prior collected at
          // different nodes are more accurate. Otherwise it could harm the
          // exploration.
          Action bestAction =
              (option.samplingMcts
                   ? pickBestAction<true>
                   : pickBestAction<false>)(root->getPiVal().playerId,
                                            node,
                                            option,
                                            rng,
                                            rollouts);
          // this is a terminal state that has been visited
          if (bestAction == InvalidAction) {
            flushActions();
            break;
          }
          auto& state = storeStateInNode ? node->getState() : *localState;
          bool save = false;
          if (!storeStateInNode) {
            Node* childNode = node->getChild(bestAction);
            if (childNode) {
              node = childNode;
              if (node->hasState()) {
                checkpointState = &node->getState();
                queuedActions.clear();
              } else {
                queuedActions.push_back(bestAction);
              }
              node->release();
              node = childNode;
              // printf("skip\n");
              continue;
            }
            save = queuedActions.size() >= (size_t)option.storeStateInterval;
            flushActions();
            localState->forward(bestAction);
            // printf("forward -> %s\n", localState->history().c_str());
          }

          Node* childNode = node->newChild(storage->newNode(), bestAction);
          if (save) {
            if (childNode->localState() &&
                childNode->localState()->typeId() == state.typeId()) {
              State* dst = &*childNode->localState();
              dst->copy(state);
              childNode->setState(dst);
            } else {
              childNode->localState() = localState->clone();
              childNode->setState(&*childNode->localState());
            }
          }
          node->release();
          node = childNode;
        }

        // 2. Expansion

        auto& state = storeStateInNode ? node->getState() : *localState;
        if (state.terminated()) {
          PiVal& piVal = node->piVal_;
          piVal.policy.clear();
          piVal.value = state.getReward(state.getCurrentPlayer());
          piVal.playerId = state.getCurrentPlayer();

          st.terminated = true;
        } else {
          st.terminated = false;
        }

        st.node = node;
        st.state = std::move(localState);
        actor.batchPrepare(i, state, rsp ? *rsp : std::vector<float>{});
      }

      double e = ftimer.elapsed();
      double s;
      do {
        s = sumftimes;
      } while (!sumftimes.compare_exchange_weak(s, s + e));
      ++nftimes;
    };

    functionHandles[i] = task.getHandle(*reservedThreads[i], std::move(f));
    functionHandles[i].setPriority(tube::threadId);
  }

  actor.batchResize(states.size());

  if (option.randomizedRollouts && rollouts > 1) {
    float mean = std::uniform_int_distribution<int>(0, 1)(rng) != 0
                     ? rollouts / 8
                     : rollouts * 2;
    std::normal_distribution<float> r(mean, rollouts / 4);
    int max = rollouts * 4;
    do {
      rollouts = r(rng);
    } while (rollouts < 1 || rollouts > max);
  }

  while (true) {

    keepGoing = ((((max_time > 0) || (numRollout < rollouts)) &&
                  ((elapsedTime < max_time) || (max_time <= 0))) ||
                 numRollout < 2);

    otherTime += timer.elapsed_reset();

    for (size_t i = 0; i < states.size(); i += stride) {
      task.enqueue(functionHandles[i]);
    }

    enqueueTime += timer.elapsed_reset();
    task.wait();
    waitTime += timer.elapsed_reset();

    if (!keepGoing) {
      break;
    }

    setupTime += timer.elapsed_reset();

    actor.batchEvaluate(states.size());

    evaluateTime += timer.elapsed_reset();

    backpropTime += timer.elapsed_reset();

    rolloutCount += states.size();

    ++numRollout;
    auto end = std::chrono::steady_clock::now();
    elapsedTime =
        std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1>>>(end - begin)
            .count();
  }

  double t3 = std::chrono::duration_cast<
                  std::chrono::duration<double, std::ratio<1, 1000>>>(
                  std::chrono::steady_clock::now() - begin)
                  .count();

  // printf("mcts x%d done in %fms  - enqueue %f wait %f other %f  setup %f
  // evaluate %f backprop %f  ftimes %f x %d avg %f\n", states.size(), t3,
  // enqueueTime, waitTime, otherTime, setupTime, evaluateTime, backpropTime,
  // sumftimes.load(), nftimes.load(), sumftimes.load() / nftimes.load());

  // std::terminate();

  for (const Node* root : rootNode) {
    const Node* bestChild = nullptr;
    mcts::Action bestAction = -1;
    int best = 0;
    for (auto& v : root->getChildren()) {
      const Node* child = v.second;
      if (child->getMctsStats().getNumVisit() > best) {
        best = child->getMctsStats().getNumVisit();
        bestAction = v.first;
        bestChild = child;
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
            forcedRollouts(root->getPiVal().policy[v.first], rollouts, option);
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
                    const std::vector<const State*>& rootState,
                    const std::vector<std::vector<float>>& rnnState,
                    Actor& actor,
                    const MctsOption& option,
                    double max_time,
                    std::minstd_rand& rng,
                    const std::vector<std::vector<float>>& policyBias) {

  if (option.storeStateInNode) {
    return computeRolloutsImpl<true>(rootNode,
                                     rootState,
                                     rnnState,
                                     actor,
                                     option,
                                     max_time,
                                     rng,
                                     policyBias);
  } else {
    return computeRolloutsImpl<false>(rootNode,
                                      rootState,
                                      rnnState,
                                      actor,
                                      option,
                                      max_time,
                                      rng,
                                      policyBias);
  }
}

}  // namespace mcts
