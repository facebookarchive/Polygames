/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "game.h"
#include "common/thread_id.h"
#include "common/threads.h"
#include "forward_player.h"
#include "utils.h"

#include <fmt/printf.h>

namespace core {

struct BatchExecutor {

  struct MoveHistory {
    int turn = 0;
    uint64_t move = 0;
    float value = 0.0f;
    torch::Tensor shortFeat;
    bool featurized = false;
  };

  struct Sequence {
    std::vector<torch::Tensor> feat;
    std::vector<torch::Tensor> v;
    std::vector<torch::Tensor> pi;
    std::vector<torch::Tensor> piMask;
    std::vector<torch::Tensor> actionPi;
    std::vector<torch::Tensor> predV;
    torch::Tensor rnnInitialState;
    std::vector<torch::Tensor> rnnStateMask;
    std::vector<torch::Tensor> predictPi;
    std::vector<torch::Tensor> predictPiMask;
  };

  struct GameState {
    std::unique_ptr<State> state;
    std::vector<std::unique_ptr<State>> playerState;
    std::vector<size_t> players;
    std::vector<size_t> playersReverseMap;
    std::vector<std::vector<torch::Tensor>> feat;
    std::vector<std::vector<torch::Tensor>> pi;
    std::vector<std::vector<torch::Tensor>> piMask;
    std::vector<std::vector<torch::Tensor>> rnnStates;
    std::vector<std::vector<torch::Tensor>> actionPi;
    std::vector<std::vector<torch::Tensor>> predV;
    std::vector<std::vector<float>> reward;
    size_t stepindex;
    std::chrono::steady_clock::time_point start;
    std::vector<int> resignCounter;
    int drawCounter = 0;
    bool canResign = false;
    int resigned = -1;
    bool drawn = false;
    std::chrono::steady_clock::time_point prevMoveTime =
        std::chrono::steady_clock::now();
    std::vector<size_t> playerOrder;
    std::vector<MoveHistory> history;
    bool justRewound = false;
    bool justRewoundToNegativeValue = false;
    int rewindCount = 0;
    std::vector<torch::Tensor> rnnState;
    std::vector<torch::Tensor> rnnState2;

    std::vector<int> allowRandomMoves;
    bool validTournamentGame = false;

    std::vector<size_t> startMoves;

    int randMoveCount = 0;
  };

  Game* game = nullptr;
  std::vector<Player*> players_;
  std::unique_ptr<State> basestate;
  std::minstd_rand rng{std::random_device{}()};
  std::vector<Sequence> seqs;
  std::list<GameState> states;
  std::list<GameState> freeGameList;
  int64_t startedGameCount = 0;
  int64_t completedGameCount = 0;
  float runningAverageGameSteps = 0.0f;
  ActorPlayer* devPlayer = nullptr;
  std::vector<ActorPlayer*> actorPlayers;
  std::vector<mcts::MctsPlayer*> mctsPlayers;
  std::vector<ForwardPlayer*> forwardPlayers;
  std::vector<float> result_;
  std::vector<std::vector<const State*>> actStates;
  std::vector<std::vector<const State*>> actPlayerStates;
  std::vector<std::vector<GameState*>> actGameStates;
  std::vector<const State*> playerActStates;
  bool alignPlayers = true;
  std::vector<std::pair<size_t, size_t>> statePlayerSize;
  std::vector<size_t> remapPlayerIdx;
  async::Task task;
  std::vector<torch::Tensor> actRnnState;
  const mcts::MctsOption* mctsOption = nullptr;
  std::vector<mcts::MctsResult> mctsResult;
  mutable std::mutex recordMoveMutex;

  int randint(int n) {
    return std::uniform_int_distribution<int>(0, n - 1)(rng);
  }

  std::unique_ptr<State> cloneState(const std::unique_ptr<State>& state) const {
    return state->clone();
  }

  void doRandomMoves(GameState& gst, int n) {
    auto o = cloneState(gst.state);
    std::vector<size_t> moves;
    for (; n > 0; --n) {
      if (gst.state->terminated()) {
        break;
      }
      size_t n = randint(gst.state->GetLegalActions().size());
      moves.push_back(n);
      gst.state->forward(n);
    }
    if (gst.state->terminated()) {
      gst.state = std::move(o);
    } else {
      for (auto m : moves) {
        for (auto& x : gst.playerState) {
          if (x) {
            x->forward(m);
          }
        }
      }
      // fmt::printf("Did %d random moves: '%s'\n", gst.state->getStepIdx(),
      // gst.state->history());
    }
    gst.startMoves = std::move(moves);
  };

  std::list<GameState>::iterator addGame(std::list<GameState>::iterator at) {
    if (!freeGameList.empty()) {
      GameState gst = std::move(freeGameList.front());
      freeGameList.pop_front();
      return states.insert(at, std::move(gst));
    }
    ++startedGameCount;
    GameState gst;
    for (size_t i = 0; i != players_.size(); ++i) {
      gst.players.push_back(i);
    }
    std::shuffle(gst.players.begin(), gst.players.end(), rng);
    gst.playersReverseMap.resize(players_.size());
    for (size_t i = 0; i != players_.size(); ++i) {
      gst.playersReverseMap[gst.players[i]] = i;
    }
    gst.state = cloneState(basestate);
    unsigned long seed = rng();
    gst.state->newGame(seed);
    gst.playerState.resize(players_.size());
    for (size_t i = 0; i != players_.size(); ++i) {
      std::unique_ptr<State> s = nullptr;
      int index = gst.players[i];
      if (&*game->playerGame_[index] != game) {
        s = cloneState(game->playerGame_[index]->state_);
        s->newGame(seed);
      }
      gst.playerState[i] = std::move(s);
    }
    gst.feat.resize(players_.size());
    gst.pi.resize(players_.size());
    gst.piMask.resize(players_.size());
    gst.reward.resize(players_.size());
    gst.rnnState.resize(players_.size());
    gst.rnnState2.resize(players_.size());
    gst.rnnStates.resize(players_.size());
    gst.actionPi.resize(players_.size());
    gst.predV.resize(players_.size());
    gst.stepindex = 0;
    gst.start = std::chrono::steady_clock::now();
    gst.resignCounter.resize(players_.size());
    gst.canResign = !game->evalMode && players_.size() == 2 && randint(3) != 0;
    gst.validTournamentGame = true;
    gst.allowRandomMoves.resize(players_.size());
    for (auto& v : gst.allowRandomMoves) {
      v = randint(4) == 0;
    }
    if (randint(250) == 0) {
      switch (randint(2)) {
      case 0:
        doRandomMoves(gst, randint(std::max((int)runningAverageGameSteps, 1)));
        break;
      case 1:
        doRandomMoves(
            gst, randint(std::max((int)runningAverageGameSteps / 10, 1)));
        break;
      case 2:
        doRandomMoves(
            gst, randint(std::max((int)runningAverageGameSteps / 5, 1)));
        break;
      }
      gst.validTournamentGame = false;
    }
    return states.insert(at, std::move(gst));
  }

  bool rewind(GameState* s, int player, bool rewindToNegativeValue) const {
    if (s->history.size() <= 2) {
      // fmt::printf("refusing to rewind with history size %d\n",
      // s->history.size());
      return false;
    }
    float flip = rewindToNegativeValue ? -1 : 1;
    size_t index = 0;
    for (index = s->history.size(); index;) {
      --index;
      auto& h = s->history[index];
      if (h.turn == player && h.value * flip > 0) {
        break;
      }
    }
    if (index <= 2) {
      // fmt::printf("refusing to rewind to index %d\n", index);
      return false;
    }
    if (!s->rnnStates.empty() || !s->rnnState.empty() ||
        !s->rnnState2.empty()) {
      bool rnn = false;
      for (auto& v : actorPlayers) {
        if (v->rnnSeqlen()) {
          rnn = true;
        }
      }
      if (rnn) {
        fmt::printf("Cannot currently rewind with rnn states, sorry :(\n");
        return false;
      }
    }
    fmt::printf("rewinding from %d to index %d\n", s->history.size(), index);
    s->justRewound = true;
    s->justRewoundToNegativeValue = rewindToNegativeValue;

    auto& gst = *s;
    gst.state = cloneState(basestate);

    for (size_t i = 0; i != gst.playerState.size(); ++i) {
      auto& x = gst.playerState[i];
      if (x) {
        int player = gst.players.at(i);
        x = cloneState(game->playerGame_.at(player)->state_);
      }
    }

    for (auto m : gst.startMoves) {
      gst.state->forward(m);
      for (auto& x : gst.playerState) {
        if (x) {
          x->forward(m);
        }
      }
    }

    for (auto& v : gst.feat) {
      v.clear();
    }
    for (auto& v : gst.pi) {
      v.clear();
    }
    for (auto& v : gst.piMask) {
      v.clear();
    }
    for (auto& v : gst.reward) {
      v.clear();
    }
    for (auto& v : gst.actionPi) {
      v.clear();
    }
    for (auto& v : gst.predV) {
      v.clear();
    }
    for (auto& v : gst.resignCounter) {
      v = 0;
    }
    gst.drawCounter = 0;
    gst.resigned = -1;
    gst.drawn = false;

    gst.history.resize(index);
    for (auto& v : gst.history) {
      v.featurized = false;
      gst.state->forward(v.move);
      for (auto& x : gst.playerState) {
        if (x) {
          x->forward(v.move);
        }
      }
    }
    return true;
  }

  using stateCallback = void (BatchExecutor::*)(GameState*,
                                                int currentPlayerIndex,
                                                size_t index) const;

  void actPrepareRnn(GameState* gameState,
                     int currentPlayerIndex,
                     size_t index) const {
    size_t slot = gameState->playersReverseMap.at(currentPlayerIndex);

    if (!gameState->rnnState.at(slot).defined()) {
      auto shape = actorPlayers.at(currentPlayerIndex)->rnnStateSize();
      gameState->rnnState[slot] = torch::zeros(shape);
    }

    const_cast<torch::Tensor&>(actRnnState[index]) =
        std::move(gameState->rnnState[slot]);
    gameState->rnnState[slot].reset();

    if (&*game->playerGame_.at(currentPlayerIndex) == game) {
      gameState->rnnStates.at(slot).push_back(actRnnState[index].cpu());
    }
  }

  void actPrepareForward(GameState* gameState,
                         int currentPlayerIndex,
                         size_t index) const {
    auto* player = forwardPlayers.at(currentPlayerIndex);
    if (actRnnState.empty()) {
      player->batchPrepare(index, *gameState->state, {});
    } else {
      player->batchPrepare(index, *gameState->state, actRnnState.at(index));
    }
  }

  void actResult(GameState* gameState,
                 int currentPlayerIndex,
                 size_t index) const {
    State* state = &*gameState->state;
    size_t slot = gameState->playersReverseMap.at(currentPlayerIndex);

    if (gameState->rnnState.at(slot).defined()) {
      throw std::runtime_error("rnnState is not empty error");
    }
    mcts::MctsPlayer* mctsPlayer = mctsPlayers[currentPlayerIndex];
    ForwardPlayer* forwardPlayer = forwardPlayers[currentPlayerIndex];
    PiVal pival;
    int bestAction = -1;
    float value = 0.0f;

    thread_local std::mt19937_64 tlrng(std::random_device{}());

    if (forwardPlayer) {
      forwardPlayer->batchResult(index, *state, pival);

      gameState->rnnState[slot] = std::move(pival.rnnState);

      thread_local std::vector<float> x;
      getLegalPi(*state, pival.logitPolicy, x);
      softmax_(x);

      bestAction = std::discrete_distribution(x.begin(), x.end())(tlrng);

      int oa = state->overrideAction();
      if (oa != -1) {
        bestAction = oa;
      }

      value = pival.value;

    } else if (mctsPlayer) {
      gameState->rnnState[slot] = std::move(mctsResult.at(index).rnnState);
      bestAction = mctsResult.at(index).bestAction;
      value = mctsResult.at(index).rootValue;
    } else {
      throw std::runtime_error("unknown player");
    }

    if (gameState->canResign) {
      if (value < -0.95f) {
        if (++gameState->resignCounter.at(slot) >= 7) {
          gameState->resigned = int(slot);
        }
      } else {
        gameState->resignCounter.at(slot) = 0;
      }
      int opponent =
          (gameState->playersReverseMap.at(currentPlayerIndex) + 1) % 2;
      if (value > 0.95f) {
        ++gameState->resignCounter.at(opponent);
      } else {
        gameState->resignCounter.at(opponent) = 0;
      }

      // Automatic draw disabled for now; this is not the best way to handle
      // this
      // TODO: enable this only for models with logit value outputs, since they
      // have
      //       a logit specific for draw - this would work best with MCTS
      //       modification that allows backpropagating the draw probability
      //      if (gameState->stepindex >= 40 && value >
      //      -0.05 && value < 0.05) {
      //        ++gameState->drawCounter;
      //        if (gameState->drawCounter >= 7) {
      //          gameState->drawn = true;
      //        }
      //      } else {
      //        gameState->drawCounter = 0;
      //      }
    }
    bool saveForTraining = true;
    // TODO: improve this randomizedRollouts check, 1.5 is a magic number that
    //       needs to be synchronized with mcts.cc
    if (mctsOption && mctsOption->randomizedRollouts &&
        mctsResult.at(index).rollouts <
            mctsOption->numRolloutPerThread * 1.5f) {
      saveForTraining = false;
    }
    if (saveForTraining) {
      torch::Tensor feat = getFeatureInTensor(*state);
      gameState->feat.at(slot).push_back(feat);
      if (forwardPlayer) {
        auto actionPolicy = torch::zeros_like(pival.logitPolicy);
        auto action = state->GetLegalActions().at(bestAction);
        actionPolicy[action.GetX()][action.GetY()][action.GetZ()] = 1;
        gameState->actionPi.at(slot).push_back(actionPolicy);
        gameState->pi.at(slot).push_back(pival.logitPolicy);
        gameState->piMask.at(slot).push_back(getPolicyMaskInTensor(*state));
      } else {
        auto [policy, policyMask] =
            getPolicyInTensor(*state, mctsResult.at(index).mctsPolicy);
        gameState->pi.at(slot).push_back(policy);
        gameState->piMask.at(slot).push_back(policyMask);
      }
      torch::Tensor predV = torch::zeros({1}, torch::kFloat32);
      predV[0] = value;
      gameState->predV.at(slot).push_back(predV);

      gameState->reward[slot].push_back(state->getReward(slot));
    }

    gameState->history.emplace_back();
    auto& h = gameState->history.back();
    h.turn = slot;
    h.move = bestAction;
    h.value = value;
    h.featurized = saveForTraining;
    h.shortFeat = getRawFeatureInTensor(*state);

    if (gameState->rewindCount == 0) {
      std::lock_guard l(recordMoveMutex);
      actorPlayers[currentPlayerIndex]->recordMove(state);
    }

    state->forward(bestAction);

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<
                         std::chrono::duration<double, std::ratio<1, 1>>>(
                         now - gameState->prevMoveTime)
                         .count();
    gameState->prevMoveTime = now;

    // fmt::printf("Thread %d: move took %gs\n", common::getThreadId(),
    // elapsed);

    {
      std::unique_lock<std::mutex> lkStats(game->mutexStats_);
      auto& stats_s = game->stats_["Move Duration (seconds)"];
      std::get<0>(stats_s) += 1;
      std::get<1>(stats_s) += elapsed;
      std::get<2>(stats_s) += elapsed * elapsed;
    }

    if (gameState->justRewound) {
      float flip = gameState->justRewoundToNegativeValue ? -1.0f : 1.0f;
      if (h.value * flip < 0.0f) {
        // fmt::printf("rewound turned negative, rewinding more!\n");
        rewind(gameState, slot, gameState->justRewoundToNegativeValue);
      } else {
        gameState->justRewound = false;
      }
    }

    // fmt::printf("game in progress: %s\n", state->history());
  }

  std::vector<stateCallback> stateCallbacks;

  std::vector<async::Handle> taskHandles;

  void prepareStateCallbacks() {
    stateCallbacks.clear();
    taskHandles.clear();
    size_t offset = 0;
    for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
      size_t currentPlayerIndex = statePlayerSize[pi].first;
      size_t currentPlayerStates = statePlayerSize[pi].second;
      for (size_t i = 0; i != currentPlayerStates; ++i) {
        GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);

        auto f = [this, gameState, currentPlayerIndex, index = offset + i]() {
          for (auto& cb : stateCallbacks) {
            (this->*cb)(gameState, currentPlayerIndex, index);
          }
        };

        auto& thread = threads::threads.getThread();
        taskHandles.push_back(task.getHandle(thread, std::move(f)));
        taskHandles.back().setPriority(common::getThreadId());
      }
      offset += currentPlayerStates;
    }
  }

  void pushStateCallback(stateCallback cb) {
    stateCallbacks.push_back(cb);
  }

  void runStateCallbacks() {
    if (stateCallbacks.empty()) {
      return;
    }
    for (auto& v : taskHandles) {
      task.enqueue(v);
    }
    task.wait();
  }

  void actForPlayer(size_t playerIndex) {
    // Merge all identical players so they get batched together
    auto& states = actStates[playerIndex];
    if (!states.empty()) {
      statePlayerSize.clear();
      statePlayerSize.emplace_back(playerIndex, states.size());
      for (size_t i = 0; i != players_.size(); ++i) {
        if (i != playerIndex && remapPlayerIdx[i] == playerIndex) {
          auto& nstates = actStates[i];
          if (!nstates.empty()) {
            states.insert(states.end(), nstates.begin(), nstates.end());
            statePlayerSize.emplace_back(i, nstates.size());
            nstates.clear();
          }
        }
      }
      playerActStates.resize(states.size());
      size_t offset = 0;
      for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
        size_t currentPlayerIndex = statePlayerSize[pi].first;
        size_t currentPlayerStates = statePlayerSize[pi].second;
        for (size_t i = 0; i != currentPlayerStates; ++i) {
          GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);
          int slot = gameState->playersReverseMap.at(currentPlayerIndex);
          if (gameState->playerState.at(slot)) {
            playerActStates.at(offset + i) = &*gameState->playerState[slot];
          } else {
            playerActStates.at(offset + i) = states.at(offset + i);
          }
        }
        offset += currentPlayerStates;
      }

      prepareStateCallbacks();

      if (actorPlayers.at(playerIndex)->rnnSeqlen()) {
        actRnnState.resize(states.size());
        pushStateCallback(&BatchExecutor::actPrepareRnn);
      } else {
        actRnnState.clear();
      }

      mctsResult.clear();

      if (forwardPlayers.at(playerIndex)) {
        forwardPlayers[playerIndex]->batchResize(states.size());
        pushStateCallback(&BatchExecutor::actPrepareForward);
        runStateCallbacks();
        forwardPlayers[playerIndex]->batchEvaluate(states.size());
      } else {

        runStateCallbacks();

        mctsResult =
            mctsPlayers.at(playerIndex)->actMcts(playerActStates, actRnnState);
      }

      stateCallbacks.clear();

      // allowRandomMoves support
      // TODO: move into a state callback
      if (mctsPlayers.at(playerIndex)) {
        offset = 0;
        for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
          size_t currentPlayerIndex = statePlayerSize[pi].first;
          size_t currentPlayerStates = statePlayerSize[pi].second;
          for (size_t i = 0; i != currentPlayerStates; ++i) {
            State* state = (State*)states.at(offset + i);
            GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);

            size_t slot = gameState->playersReverseMap.at(currentPlayerIndex);

            if (gameState->allowRandomMoves.at(slot)) {
              float x = 4.0f / std::pow(state->getStepIdx() + 10, 2.0f);
              if (std::uniform_real_distribution<float>(0, 1.0f)(rng) < x) {
                mctsResult.at(offset + i).bestAction =
                    randint(state->GetLegalActions().size());
                // fmt::printf("at state '%s' - performing random move %s\n",
                // state->history(),
                // state->actionDescription(state->GetLegalActions().at(mctsResult.at(offset
                // + i).bestAction)));
                gameState->validTournamentGame = false;
              }
            }
          }
          offset += currentPlayerStates;
        }
      }

      // Support for running the opponent player in a different game
      // implementation. This is rarely needed or used, but can be useful to
      // train against a model that was trained on a different game
      // implementation but with the same action space
      // TODO: move into a state callback
      if (&*game->playerGame_.at(playerIndex) != game) {
        if (devPlayer->rnnSeqlen()) {
          actRnnState.clear();

          offset = 0;
          for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
            size_t currentPlayerIndex = statePlayerSize[pi].first;
            size_t currentPlayerStates = statePlayerSize[pi].second;
            for (size_t i = 0; i != currentPlayerStates; ++i) {
              GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);

              size_t slot = gameState->playersReverseMap.at(currentPlayerIndex);

              if (!gameState->rnnState2.at(slot).defined()) {
                gameState->rnnState2[slot] =
                    torch::zeros(devPlayer->rnnStateSize());
              }

              actRnnState.push_back(std::move(gameState->rnnState2[slot]));

              gameState->rnnStates.at(slot).push_back(actRnnState.back().cpu());
            }
            offset += currentPlayerStates;
          }

          std::vector<torch::Tensor> nextRnnState =
              devPlayer->nextRnnState(states, actRnnState);

          offset = 0;
          for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
            size_t currentPlayerIndex = statePlayerSize[pi].first;
            size_t currentPlayerStates = statePlayerSize[pi].second;
            for (size_t i = 0; i != currentPlayerStates; ++i) {
              GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);

              size_t slot = gameState->playersReverseMap.at(currentPlayerIndex);

              gameState->rnnState2[slot] =
                  std::move(nextRnnState.at(offset + i));
            }
            offset += currentPlayerStates;
          }
        }

        offset = 0;
        for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
          size_t currentPlayerStates = statePlayerSize[pi].second;
          for (size_t i = 0; i != currentPlayerStates; ++i) {
            State* state = (State*)playerActStates.at(offset + i);

            auto& res = mctsResult.at(offset + i);

            state->forward(res.bestAction);
          }
          offset += currentPlayerStates;
        }

      } else {
        offset = 0;
        for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
          size_t currentPlayerIndex = statePlayerSize[pi].first;
          size_t currentPlayerStates = statePlayerSize[pi].second;
          for (size_t i = 0; i != currentPlayerStates; ++i) {
            GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);
            auto& res = mctsResult.at(offset + i);

            for (auto& x : gameState->playerState) {
              if (x) {
                x->forward(res.bestAction);
              }
            }
          }
          offset += currentPlayerStates;
        }
      }

      mctsOption = mctsPlayers[playerIndex]
                       ? &mctsPlayers[playerIndex]->option()
                       : nullptr;

      if (mctsPlayers.at(playerIndex)) {
        double rps = mctsPlayers.at(playerIndex)->rolloutsPerSecond();
        std::unique_lock<std::mutex> lkStats(game->mutexStats_);
        auto& stats_s = game->stats_["Rollouts per second"];
        std::get<0>(stats_s) += 1;
        std::get<1>(stats_s) += rps;
        std::get<2>(stats_s) += rps * rps;
      }

      pushStateCallback(&BatchExecutor::actResult);

      runStateCallbacks();

      states.clear();
    }
  }

  void run() {

    task = async::Task(threads::threads);

    for (auto& v : game->players_) {
      players_.push_back(&*v);
    }

    result_.resize(players_.size());

    actStates.resize(players_.size());
    actPlayerStates.resize(players_.size());
    actGameStates.resize(players_.size());
    remapPlayerIdx.resize(players_.size());

    basestate = std::move(game->state_);

    seqs.resize(players_.size());

    size_t ngames = size_t(std::max(game->perThreadBatchSize, 1));

    if (game->perThreadBatchSize < 1) {
      int bs = 102400;
      int n = 0;
      for (auto& v : players_) {
        auto actorPlayer = dynamic_cast<ActorPlayer*>(v);
        if (actorPlayer) {
          int v = actorPlayer->findBatchSize(*basestate);
          if (v > 0) {
            bs = std::min(bs, v);
            ++n;
          }
        }
      }
      if (n) {
        fmt::printf("Using batch size of %d\n", bs);
        ngames = bs;
      }
    }

    while (states.size() < ngames &&
           (game->numEpisode < 0 || startedGameCount < game->numEpisode)) {
      addGame(states.end());
    }

    devPlayer = nullptr;
    for (auto& v : players_) {
      auto actorPlayer = dynamic_cast<ActorPlayer*>(v);
      if (!actorPlayer) {
        throw std::runtime_error(
            "Cannot use perThreadBatchSize without ActorPlayer");
      }
      if (actorPlayer->getName() == "dev") {
        devPlayer = actorPlayer;
      }
      actorPlayers.push_back(std::move(actorPlayer));
      mctsPlayers.push_back(dynamic_cast<mcts::MctsPlayer*>(v));
      forwardPlayers.push_back(dynamic_cast<ForwardPlayer*>(v));
    }
    if (!devPlayer) {
      throw std::runtime_error("dev player not found");
    }

    // If two players are the same (pointer comparison), then they can act
    // together.
    for (size_t i = 0; i != players_.size(); ++i) {
      remapPlayerIdx[i] = i;
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (i != i2 && players_[i] == players_[i2]) {
          remapPlayerIdx[i] = i2;
        }
      }
    }

    while (!states.empty() && !game->terminate_) {

      for (auto& v : actStates) {
        v.clear();
      }
      for (auto& v : actPlayerStates) {
        v.clear();
      }
      for (auto& v : actGameStates) {
        v.clear();
      }

      for (auto i = states.begin(); i != states.end();) {
        auto* state = &*i->state;
        bool completed = state->terminated() || i->resigned != -1 || i->drawn;
        if (completed) {
          const auto end = std::chrono::steady_clock::now();
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::seconds>(end - i->start)
                  .count();
          const size_t stepindex = i->stepindex;
          if (i->rewindCount == 0) {
            std::unique_lock<std::mutex> lkStats(game->mutexStats_);
            auto& stats_steps = game->stats_["Game Duration (steps)"];
            std::get<0>(stats_steps) += 1;
            std::get<1>(stats_steps) += stepindex;
            std::get<2>(stats_steps) += stepindex * stepindex;
            auto& stats_s = game->stats_["Game Duration (seconds)"];
            std::get<0>(stats_s) += 1;
            std::get<1>(stats_s) += elapsed;
            std::get<2>(stats_s) += elapsed * elapsed;
          }
          if (i->drawn) {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(i->players.at(idx)) = 0;
            }
          }
          if (i->resigned != -1) {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(i->players.at(idx)) = int(idx) == i->resigned ? -1 : 1;
            }
            // fmt::printf("player %d (%s) resigned : %s\n", i->resigned,
            //            players_.at(i->players.at(i->resigned))->getName(),
            //            state->history());
          } else {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(i->players.at(idx)) = state->getReward(idx);
            }
            // fmt::printf("game ended normally: %s\n",
            // state->history().c_str());
            if (randint(256) == 0) {
              fmt::printf(
                  "game ended normally: %s\n", state->history().c_str());
            }
          }

          runningAverageGameSteps =
              runningAverageGameSteps * 0.99f + state->getStepIdx() * 0.01f;
        }

        bool doRewind = false;
        int rewindPlayer = 0;
        bool rewindToNegativeValue = false;

        bool isForward = dynamic_cast<ForwardPlayer*>(devPlayer) != nullptr;

        int seqlen = devPlayer->rnnSeqlen();

        if ((isForward && seqlen > 0) || completed) {
          for (size_t slot = 0; slot != players_.size(); ++slot) {
            size_t dstp = i->players.at(slot);

            if (completed) {
#ifdef OPENBW_UI
              fmt::printf("Result for %s: %g\n", players_[dstp]->getName(),
                          result_[dstp]);
#endif
            } else {
              if ((int)i->pi[slot].size() < seqlen * 16 + 1 ||
                  i->history.empty() || i->history.back().turn != (int)slot) {
                continue;
              }
              result_[dstp] = i->history.back().value;

              i->pi[slot].pop_back();
              i->piMask[slot].pop_back();
              i->actionPi[slot].pop_back();
              i->predV[slot].pop_back();
              i->feat[slot].pop_back();
              i->rnnStates[slot].pop_back();
              i->reward[slot].pop_back();
            }

            auto addseq = [&](const std::vector<torch::Tensor>& src,
                              std::vector<torch::Tensor>& dst,
                              tube::EpisodicTrajectory& traj) {
              for (auto& x : src) {
                dst.push_back(x);
                if ((int)dst.size() > seqlen) {
                  throw std::runtime_error("addseq bad seqlen");
                }
                if ((int)dst.size() == seqlen) {
                  traj.pushBack(torch::stack(dst));
                  dst.clear();
                }
              }
            };

            std::vector<float> dReward;
            dReward.resize(i->feat[slot].size());
            if (isForward) {
              float gae = 0.0f;
              float gamma = 0.997;
              float gaeLambda = 0.95;
              float reward = result_[slot];
              // reward = 0;
              for (size_t n = dReward.size(); n;) {
                --n;
                float predv = i->predV[slot].at(n).item<float>();
                float npredv = 0.0f;
                if (n == dReward.size() - 1) {
                  npredv = result_[dstp];
                  // npredv = 0;
                } else {
                  npredv = i->predV[slot].at(n + 1).item<float>();
                }
                float delta = reward + gamma * npredv - predv;
                gae = delta + gamma * gaeLambda * gae;

                dReward.at(n) = gae + predv;

                reward = i->reward[slot].at(n);
              }
            } else {
              float reward = result_[dstp];
              for (size_t n = dReward.size(); n;) {
                --n;
                dReward.at(n) = reward;

                // reward *= 0.99;
                // reward += i->reward[slot].at(n);
              }
            }

            std::vector<torch::Tensor> rewards;
            for (size_t j = 0; j != i->feat[slot].size(); ++j) {
              if (devPlayer->vOutputs() == 3) {
                torch::Tensor reward = torch::zeros({3}, torch::kFloat32);
                reward[0] = result_[dstp] > 0;
                reward[1] = result_[dstp] < 0;
                reward[2] = result_[dstp] == 0;
                rewards.push_back(std::move(reward));
              } else {
                torch::Tensor reward = torch::zeros({1}, torch::kFloat32);
                reward[0] = dReward.at(j);
                rewards.push_back(std::move(reward));
              }
            }

            std::vector<float> piReward;
            piReward.resize(i->pi[slot].size());
            auto& seq = seqs[dstp];
            if (actorPlayers[dstp]->getModelId() == "dev" &&
                i->feat[slot].size() > 0) {
              if (seqlen) {
                for (size_t n = 0; n != i->feat[slot].size(); ++n) {
                  if ((seq.feat.size() + n) % seqlen == seqlen - 1) {
                    game->rnnInitialState_[dstp].pushBack(
                        i->rnnStates[slot].at(n));
                  }
                }
                addseq(i->feat[slot], seq.feat, game->feature_[dstp]);
                addseq(i->pi[slot], seq.pi, game->pi_[dstp]);
                addseq(i->piMask[slot], seq.piMask, game->piMask_[dstp]);
                if (isForward) {
                  addseq(
                      i->actionPi[slot], seq.actionPi, game->actionPi_[dstp]);
                }
                addseq(i->predV[slot], seq.predV, game->predV_[dstp]);
                std::vector<torch::Tensor> rnnStateMask;
                rnnStateMask.resize(i->feat[slot].size());
                for (auto& v : rnnStateMask) {
                  v = torch::ones({1});
                }
                rnnStateMask.at(0).zero_();
                addseq(
                    rnnStateMask, seq.rnnStateMask, game->rnnStateMask_[dstp]);
              } else {
                for (auto& v : i->feat[slot]) {
                  game->feature_[dstp].pushBack(v);
                }
                for (auto& v : i->pi[slot]) {
                  game->pi_[dstp].pushBack(v);
                }
                for (auto& v : i->piMask[slot]) {
                  game->piMask_[dstp].pushBack(v);
                }
                for (auto& v : i->actionPi[slot]) {
                  game->actionPi_[dstp].pushBack(v);
                }
                for (auto& v : i->predV[slot]) {
                  game->predV_[dstp].pushBack(v);
                }
              }

              if (game->predictEndState || game->predictNStates) {
                int n = (game->predictEndState ? 2 : 0) + game->predictNStates;
                auto size = state->GetRawFeatureSize();
                size.insert(size.begin(), n);
                auto finalsize = size;
                finalsize[1] *= finalsize[0];
                finalsize.erase(finalsize.begin());
                for (size_t m = 0; m != i->history.size(); ++m) {
                  if (!i->history[m].featurized ||
                      i->history[m].turn != (int)slot) {
                    continue;
                  }
                  auto tensor = torch::zeros(size);
                  auto mask = torch::zeros(size);
                  size_t offset = 0;
                  if (game->predictEndState) {
                    if (state->terminated()) {
                      tensor[0].copy_(i->history.back().shortFeat);
                      mask[0].fill_(1.0f);
                    } else {
                      tensor[1].copy_(i->history.back().shortFeat);
                      mask[1].fill_(1.0f);
                    }
                    offset += 2;
                  }
                  for (int j = 0; j != game->predictNStates; ++j, ++offset) {
                    size_t index = m + 1 + j;
                    if (index < i->history.size()) {
                      tensor[offset].copy_(i->history[m].shortFeat);
                      mask[offset].fill_(1.0f);
                    }
                  }

                  tensor = tensor.view(finalsize);
                  mask = mask.view(finalsize);

                  if (seqlen) {
                    addseq({tensor}, seq.predictPi, game->predictPi_[dstp]);
                    addseq(
                        {mask}, seq.predictPiMask, game->predictPiMask_[dstp]);
                  } else {
                    game->predictPi_[dstp].pushBack(tensor);
                    game->predictPiMask_[dstp].pushBack(mask);
                  }
                }
              }

              // fmt::printf("result[%d] (%s) is %g\n", p,
              // players_[p]->getName(), result_[p]);

              if (seqlen) {
                addseq(rewards, seq.v, game->v_[dstp]);
              } else {
                for (auto& reward : rewards) {
                  game->v_[dstp].pushBack(std::move(reward));
                }
              }
            }

            i->pi[slot].clear();
            i->piMask[slot].clear();
            i->actionPi[slot].clear();
            i->predV[slot].clear();
            i->feat[slot].clear();
            i->rnnStates[slot].clear();
            i->reward[slot].clear();
            for (auto& v : i->history) {
              if (v.turn == (int)slot) {
                v.featurized = false;
              }
            }

            if (completed) {
              if (actorPlayers[dstp]->getModelId() == "dev") {
                if (result_[dstp] != 0) {
                  doRewind = true;
                  rewindPlayer = slot;
                  rewindToNegativeValue = result_[dstp] > 0;
                }
              }

              if (i->rewindCount == 0 && i->validTournamentGame) {
                actorPlayers[dstp]->result(state, result_[dstp]);
              } else {
                actorPlayers[dstp]->forget(state);
              }
            }
          }
          game->sendTrajectory();

          if (doRewind) {
            for (size_t slot = 0; slot != players_.size(); ++slot) {
              size_t dstp = i->players.at(slot);
              if (actorPlayers[dstp]->wantsTournamentResult()) {
                doRewind = false;
                break;
              }
            }
          }
        }

        if (completed) {
          ++completedGameCount;
          if (doRewind && i->rewindCount < game->maxRewinds &&
              rewind(&*i, rewindPlayer, rewindToNegativeValue)) {
            ++i->rewindCount;
          } else {
            i = states.erase(i);
            if (game->numEpisode < 0 || startedGameCount < game->numEpisode) {
              i = addGame(i);
            }
          }
        } else {
          i->stepindex++;
          int slot = state->getCurrentPlayer();
          auto playerIdx = i->players.at(slot);
          actStates.at(playerIdx).push_back(state);
          actPlayerStates.at(playerIdx).push_back(&*i->playerState[slot]);
          actGameStates.at(playerIdx).push_back(&*i);
          ++i;
        }
      }

      if (alignPlayers) {
        size_t bestPlayerIdx = 0;
        size_t bestPlayerIdxSize = 0;
        for (size_t playerIdx = 0; playerIdx != actStates.size(); ++playerIdx) {
          auto& states = actStates[playerIdx];
          if (states.size() > bestPlayerIdxSize) {
            bestPlayerIdxSize = states.size();
            bestPlayerIdx = playerIdx;
          }
        }
        actForPlayer(bestPlayerIdx);
      } else {
        for (size_t playerIdx = 0; playerIdx != actStates.size(); ++playerIdx) {
          actForPlayer(playerIdx);
        }
      }
    }
  }
};

void Game::mainLoop() {
  threads::setCurrentThreadName("game thread " +
                                std::to_string(common::getThreadId()));
  threads::init(0);
  if (players_.size() != (isOnePlayerGame() ? 1 : 2)) {
    std::cout << "Error: wrong number of players: " << players_.size()
              << std::endl;
    assert(false);
  }
  if (!evalMode) {
    reset();

    for (auto& v : playerGame_) {
      if (&*v != this && v->state_) {
        v->state_->reset();
      }
    }

    BatchExecutor batchExecutor;
    batchExecutor.game = this;

    batchExecutor.run();

  } else {

    // Warm up JIT/model. This can take several seconds, so do it before we
    // start time counting.
    for (auto& v : players_) {
      auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(v);
      if (mctsPlayer && mctsPlayer->option().totalTime) {
        std::cout << "Warming up model.\n";
        auto opt = mctsPlayer->option();
        mctsPlayer->option().totalTime = 0;
        mctsPlayer->option().numRolloutPerThread = 20;
        mctsPlayer->option().randomizedRollouts = false;
        mctsPlayer->reset();
        mctsPlayer->actMcts(*state_);
        mctsPlayer->actMcts(*state_);
        mctsPlayer->actMcts(*state_);
        mctsPlayer->actMcts(*state_);

        mctsPlayer->option() = opt;
        mctsPlayer->reset();
      }
    }

    int64_t gameCount = 0;
#ifdef DEBUG_GAME
    std::thread::id thread_id = std::this_thread::get_id();
#endif
    while ((numEpisode < 0 || gameCount < numEpisode) && !terminate_) {
      if (terminate_) {
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ": terminating, "
                  << "game " << this << ", " << gameCount << " / " << numEpisode
                  << " games played" << std::endl;
#endif
        break;
      }
#ifdef DEBUG_GAME
      std::cout << "Thread " << thread_id << ", game " << this
                << ": not terminating - run another iteration. " << std::endl;
#endif
      bool aHuman = std::any_of(players_.begin(), players_.end(),
                                [](const std::shared_ptr<Player>& player) {
                                  return player->isHuman();
                                });
      if (aHuman && state_->stochasticReset()) {
        std::string line;
        std::cout << "Random outcome ?" << std::endl;
        std::cin >> line;
        state_->forcedDice = std::stoul(line, nullptr, 0);
      }
      reset();
      int stepindex = 0;
      auto start = std::chrono::steady_clock::now();
      while (!state_->terminated()) {
        stepindex += 1;
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ", game " << this << ": step "
                  << stepindex << std::endl;
#endif
        step();
        if (isInSingleMoveMode_) {
          std::cout << lastMctsValue_ << "\n";
          state_->printLastAction();
          std::exit(0);
        }
        if (printMoves_) {
          std::cout << "MCTS value: " << lastMctsValue_ << "\n";
          std::cout << "Made move: " << state_->lastMoveString() << std::endl;
        }
      }
      auto end = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
      {
        std::unique_lock<std::mutex> lkStats(mutexStats_);
        auto& stats_steps = stats_["Game Duration (steps)"];
        std::get<0>(stats_steps) += 1;
        std::get<1>(stats_steps) += stepindex;
        std::get<2>(stats_steps) += stepindex * stepindex;
        auto& stats_s = stats_["Game Duration (seconds)"];
        std::get<0>(stats_s) += 1;
        std::get<1>(stats_s) += elapsed;
        std::get<2>(stats_s) += elapsed * elapsed;
      }
#ifdef DEBUG_GAME
      std::cout << "Thread " << thread_id << ", game " << this << ": game "
                << gameCount << " / " << numEpisode << " ended; " << stepindex
                << " steps, " << (static_cast<float>(stepindex) / elapsed)
                << " steps per second" << std::endl;
#endif
      if (!lastAction_.empty() && aHuman) {
        std::cout << "\n#Last Action: " << lastAction_ << "\n\n";
        state_->printCurrentBoard();
      }
      if (std::any_of(players_.begin(), players_.end(),
                      [](const std::shared_ptr<Player>& player) {
                        return player->isTP();
                      })) {
        state_->errPrintCurrentBoard();
      }

      result_[0] = state_->getReward(0);
      if (players_.size() > 1) {
        result_[1] = state_->getReward(1);
      }

      ++gameCount;
    }
#ifdef DEBUG_GAME
    std::cout << "Thread " << thread_id << ", game " << this
              << ": exiting main loop" << std::endl;
#endif
  }
}

std::optional<int> Game::parseSpecialAction(const std::string& str) {
  if (str == "-1" || str == "undo" || str == "u") {
    std::cout << "Undoing the last move\n";
    state_->undoLastMoveForPlayer(state_->getCurrentPlayer());
    return -1;
  } else if (str == "exit") {
    std::exit(0);
  } else if (str == "m" || str == "manual") {
    bool resume = false;
    auto playerString = [&](int index) {
      std::string str;
      auto& player = players_.at(index);
      if (std::dynamic_pointer_cast<mcts::MctsPlayer>(player)) {
        str += "MctsPlayer";
      } else if (std::dynamic_pointer_cast<HumanPlayer>(player)) {
        str += "HumanPlayer";
      } else {
        str += typeid(player).name();
      }
      return str;
    };
    auto specialAction = [&](const std::string& str) -> std::optional<int> {
      if (str == "singlemovemode" || str == "sm") {
        isInSingleMoveMode_ = true;
        return -1;
      } else if (str.substr(0, 3) == "set") {
        state_->setStateFromStr(str.substr(4));
        return -1;
      } else if (str == "r" || str == "reset") {
        state_->reset();
        for (auto& v : players_) {
          v->reset();
        }
        return -1;
      } else if (str == "u" || str == "undo") {
        state_->undoLastMove();
        return -1;
      } else if (str == "c" || str == "continue") {
        resume = true;
        return -1;
      } else if (str == "swap") {
        std::next_permutation(players_.begin(), players_.end());
        for (size_t i = 0; i != players_.size(); ++i) {
          std::cout << "Player " << i << " is now " << playerString(i) << "\n";
        }
        return -1;
      } else if (str == "printmoves") {
        printMoves_ = true;
        return -1;
      } else if (str == "printvalue") {
        auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(
            players_.at(state_->getCurrentPlayer()));
        if (!mctsPlayer) {
          for (auto& v : players_) {
            mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(v);
            if (mctsPlayer) {
              break;
            }
          }
        }
        if (mctsPlayer) {
          std::cout << "NN Value: " << mctsPlayer->calculateValue(*state_)
                    << "\n";
        } else {
          std::cout << "NN Value: 0\n";
        }
      }
      return std::nullopt;
    };
    std::cout
        << "\nEntering moves manually. Enter 'r' or 'reset' to reset the "
           "board, 'u' or 'undo' to undo the last move, 'c' or 'continue' to "
           "continue play, or 'swap' to swap the turn order of the players\n\n";
    while (!state_->terminated()) {
      int index = -1;
      while (index == -1) {
        std::cout << "Enter a move for player " << state_->getCurrentPlayer()
                  << " (" << playerString(state_->getCurrentPlayer()) << ")\n";
        index = state_->humanInputAction(specialAction);
        if (resume) {
          return -1;
        }
      }

      if (!lastAction_.empty()) {
        std::cout << "\nLast Action: " << lastAction_ << "\n\n";
      }
      std::cout << " applying action... " << std::endl;
      auto action = state_->GetLegalActions().at(index);
      lastAction_ = state_->actionDescription(action);
      if (!state_->isStochastic()) {
        state_->forward(action.GetIndex());
      } else {
        // auto backup_state = state_->clone();
        std::string line;
        std::cout << "Random outcome ?" << std::endl;
        std::cin >> line;
        state_->forcedDice = std::stoul(line, nullptr, 0);
        state_->forward(action.GetIndex());
      }
    }
    return -1;
  }
  return std::nullopt;
}

/* virtual */ tube::EnvThread::Stats Game::get_stats() {
  std::unique_lock<std::mutex> lkStats(mutexStats_);
  return stats_;
}

void Game::step() {
  auto playerIdx = state_->getCurrentPlayer();
  auto& player = players_.at(playerIdx);
  // std::cout << "board" << std::endl;
  // state_->printCurrentBoard();
  if (player->isTP()) {
    // auto TPplayer = std::dynamic_pointer_cast<TPPlayer>(player);
    assert(!state_->isStochastic());
    auto index = state_->TPInputAction();
    auto action = state_->GetLegalActions().at(index);
    lastAction_ = state_->actionDescription(action);
    state_->forward(index);
  } else if (player->isHuman()) {
    if (!hasPrintedHumanHelp_) {
      std::cout << "\nEnter a move for the human player. Enter 'u' or 'undo' "
                   "to undo your previous move, 'm' or 'manual' to enter moves "
                   "manually for all players.\n\n";
      hasPrintedHumanHelp_ = true;
    }
    auto humanPlayer = std::dynamic_pointer_cast<HumanPlayer>(player);
    if (!lastAction_.empty()) {
      std::cout << "\nLast Action: " << lastAction_ << "\n\n";
    }

    std::cout << "History: " << state_->history() << "\n";

    int index = state_->humanInputAction(
        std::bind(&Game::parseSpecialAction, this, std::placeholders::_1));
    if (index == -1) {
      return step();
    }
    std::cout << " applying action... " << std::endl;
    auto action = state_->GetLegalActions().at(index);
    lastAction_ = state_->actionDescription(action);
    if (!state_->isStochastic()) {
      state_->forward(action.GetIndex());
    } else {
      // auto backup_state = state_->clone();
      std::string line;
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
      state_->forward(action.GetIndex());
    }
  } else {
    auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(player);
    auto rnnShape = mctsPlayer->rnnStateSize();
    if (!rnnShape.empty()) {
      if (rnnState_.size() <= (size_t)playerIdx) {
        rnnState_.resize(playerIdx + 1);
      }
      if (!rnnState_.at(playerIdx).defined()) {
        rnnState_[playerIdx] = torch::zeros(rnnShape);
      }
    }
    mcts::MctsResult result;
    if (!rnnShape.empty()) {
      result = mctsPlayer->actMcts(*state_, rnnState_.at(playerIdx));
      rnnState_.at(playerIdx) = std::move(result.rnnState);
    } else {
      result = mctsPlayer->actMcts(*state_);
    }
    lastMctsValue_ = result.rootValue;

    // store feature for training
    if (!evalMode) {
      torch::Tensor feat = getFeatureInTensor(*state_);
      auto [policy, policyMask] = getPolicyInTensor(*state_, result.mctsPolicy);
      feature_[playerIdx].pushBack(std::move(feat));
      pi_[playerIdx].pushBack(std::move(policy));
      piMask_[playerIdx].pushBack(std::move(policyMask));
    }

    // std::cout << ">>>>actual act" << std::endl;
    _Action action = state_->GetLegalActions().at(result.bestAction);
    lastAction_ = state_->actionDescription(action);
    bool noHuman = std::none_of(players_.begin(), players_.end(),
                                [](const std::shared_ptr<Player>& player) {
                                  return player->isHuman();
                                });
    if (!state_->isStochastic()) {
      if (!noHuman) {
        std::cout << "Performing action "
                  << state_->actionDescription(
                         state_->GetLegalActions().at(result.bestAction))
                  << "\n";
      }
    } else if (!noHuman) {
      std::string line;
      std::cout << "Performing action "
                << state_->actionDescription(
                       state_->GetLegalActions().at(result.bestAction))
                << "\n";
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
    }
    state_->forward(result.bestAction);
  }
}

void Game::sendTrajectory() {
  for (int i = 0; i < (int)players_.size(); ++i) {
    assert(v_[i].len() == pi_[i].len() && pi_[i].len() == feature_[i].len());
    assert(pi_[i].len() == piMask_[i].len());
    int errcode;
    while (prepareForSend(i)) {
      // ignore error codes from the dispatcher
      errcode = dispatchers_[i].dispatchNoReply();
      switch (errcode) {
      case tube::Dispatcher::DISPATCH_ERR_DC_TERM:
#ifdef DEBUG_GAME
        std::cout << "game " << this << ", sendTrajectory: "
                  << "attempt to dispatch through"
                  << " a terminated data channel " << std::endl;
#endif
        break;
      case tube::Dispatcher::DISPATCH_ERR_NO_SLOT:
#ifdef DEBUG_GAME
        std::cout << "game " << this << ": sendTrajectory: "
                  << "no slots available to dispatch" << std::endl;
#endif
        break;
      case tube::Dispatcher::DISPATCH_NOERR:
        break;
      }
    }
    assert(v_[i].len() == 0);
    assert(pi_[i].len() == 0);
    assert(piMask_[i].len() == 0);
    assert(feature_[i].len() == 0);
  }
}

bool Game::prepareForSend(int playerId) {
  int len = feature_[playerId].len();
#define check(n)                                                               \
  if (!n.empty() && n[playerId].len() != len)                                  \
  throw std::runtime_error("len mismatch in " #n)
  check(pi_);
  check(piMask_);
  check(actionPi_);
  check(v_);
  check(predV_);
  check(rnnInitialState_);
  check(rnnStateMask_);
#undef check
  if (feature_[playerId].prepareForSend()) {
    bool b = pi_[playerId].prepareForSend();
    b &= piMask_[playerId].prepareForSend();
    if (!actionPi_.empty()) {
      b &= actionPi_[playerId].prepareForSend();
    }
    b &= v_[playerId].prepareForSend();
    b &= predV_[playerId].prepareForSend();
    if (predictEndState + predictNStates) {
      b &= predictPi_[playerId].prepareForSend();
      b &= predictPiMask_[playerId].prepareForSend();
    }
    if (!rnnInitialState_.empty()) {
      b &= rnnInitialState_[playerId].prepareForSend();
      b &= rnnStateMask_[playerId].prepareForSend();
    }
    if (!b) {
      throw std::runtime_error("prepareForSend mismatch 1");
    }
    return true;
  }
  bool b = pi_[playerId].prepareForSend();
  b |= piMask_[playerId].prepareForSend();
  b |= v_[playerId].prepareForSend();
  b |= predV_[playerId].prepareForSend();
  if (!actionPi_.empty()) {
    b |= actionPi_[playerId].prepareForSend();
  }
  if (predictEndState + predictNStates) {
    b |= predictPi_[playerId].prepareForSend();
    b |= predictPiMask_[playerId].prepareForSend();
  }
  if (!rnnInitialState_.empty()) {
    b |= rnnInitialState_[playerId].prepareForSend();
    b |= rnnStateMask_[playerId].prepareForSend();
  }
  if (b) {
    throw std::runtime_error("prepareForSend mismatch 2");
  }
  return false;
}

}  // namespace core
