/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "game.h"

#include <fmt/printf.h>

void Game::mainLoop() {
  if (players_.size() != (isOnePlayerGame() ? 1 : 2)) {
    std::cout << "Error: wrong number of players: " << players_.size()
              << std::endl;
    assert(false);
  }
  if (perThreadBatchSize > 0) {
    bool aHuman = std::any_of(players_.begin(), players_.end(),
                              [](const std::shared_ptr<mcts::Player>& player) {
                                return player->isHuman();
                              });
    if (aHuman && state_->stochasticReset()) {
      std::string line;
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
    }
    reset();

    auto basestate = std::move(state_);

    struct GameState {
      std::unique_ptr<State> state;
      std::vector<std::vector<torch::Tensor>> feat;
      std::vector<std::vector<torch::Tensor>> pi;
      std::vector<std::vector<torch::Tensor>> piMask;
      size_t stepindex;
      std::chrono::system_clock::time_point start;
      std::vector<int> resignCounter;
      bool canResign = false;
      int resigned = -1;
      std::chrono::steady_clock::time_point prevMoveTime =
          std::chrono::steady_clock::now();
    };

    std::list<GameState> states;

    size_t ngames = size_t(perThreadBatchSize);

    int64_t startedGameCount = 0;
    int64_t completedGameCount = 0;

    std::minstd_rand rng(std::random_device{}());
    auto randint = [&](int n) {
      return std::uniform_int_distribution<int>(0, n - 1)(rng);
    };

    auto cloneState = [&](auto& state) {
      auto x = state->clone();
      auto n = dynamic_cast<State*>(x.get());
      std::unique_ptr<State> r(n);
      if (n) {
        x.release();
      }
      return r;
    };

    auto addGame = [&](auto at) {
      ++startedGameCount;
      GameState gst;
      gst.state = cloneState(basestate);
      gst.feat.resize(players_.size());
      gst.pi.resize(players_.size());
      gst.piMask.resize(players_.size());
      gst.stepindex = 0;
      gst.start = std::chrono::system_clock::now();
      gst.resignCounter.resize(players_.size());
      gst.canResign = !evalMode && players_.size() == 2 && randint(3) != 0;
      return states.insert(at, std::move(gst));
    };

    while (states.size() < ngames &&
           (numEpisode < 0 || startedGameCount < numEpisode)) {
      addGame(states.end());
    }

    std::vector<std::shared_ptr<mcts::MctsPlayer>> mctsPlayers;
    for (auto& v : players_) {
      auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(v);
      if (!mctsPlayer) {
        throw std::runtime_error(
            "Cannot use perThreadBatchSize without MctsPlayer");
      }
      mctsPlayers.push_back(std::move(mctsPlayer));
    }

    std::vector<std::vector<const mcts::State*>> actStates(players_.size());
    std::vector<std::vector<GameState*>> actGameStates(players_.size());

    bool alignPlayers = false;

    // If two players are the same (pointer comparison), then they can act
    // together.
    std::vector<size_t> remapPlayerIdx(players_.size());
    for (size_t i = 0; i != players_.size(); ++i) {
      remapPlayerIdx[i] = i;
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (i != i2 && players_[i] == players_[i2]) {
          remapPlayerIdx[i] = i2;
        }
      }
    }

    std::vector<std::pair<size_t, size_t>> statePlayerSize;

    while (!states.empty() && !terminate_) {

      for (auto& v : actStates) {
        v.clear();
      }
      for (auto& v : actGameStates) {
        v.clear();
      }

      for (auto i = states.begin(); i != states.end();) {
        auto* state = &*i->state;
        if (state->terminated() || i->resigned != -1) {
          const auto end = std::chrono::system_clock::now();
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::seconds>(end - i->start)
                  .count();
          const size_t stepindex = i->stepindex;
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
          if (i->resigned != -1) {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(idx) = int(idx) == i->resigned ? -1 : 1;
            }
            // fmt::printf("player %d (%s) resigned : %s\n", i->resigned,
            //            players_.at(i->resigned)->getName(),
            //            state->history());
          } else {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_[idx] = state->getReward(idx);
            }
            // fmt::printf("game ended normally: %s\n",
            // state->history().c_str());
          }

          for (size_t p = 0; p != players_.size(); ++p) {
            // fmt::printf(
            //    "Result for %s: %g\n", players_[p]->getName(), result_[p]);
            for (auto& v : i->feat[p]) {
              feature_[p].pushBack(v);
            }
            for (auto& v : i->pi[p]) {
              pi_[p].pushBack(v);
            }
            for (auto& v : i->piMask[p]) {
              piMask_[p].pushBack(v);
            }

            // fmt::printf("result[%d] (%s) is %g\n", p, players_[p]->getName(),
            // result_[p]);
            while (v_[p].len() < pi_[p].len()) {
              torch::Tensor reward = torch::zeros({1}, torch::kFloat32);
              reward[0] = result_[p];
              v_[p].pushBack(std::move(reward));
            }

            players_[p]->result(state, result_[p]);
          }
          sendTrajectory();

          ++completedGameCount;
          i = states.erase(i);
          if (numEpisode < 0 || startedGameCount < numEpisode) {
            i = addGame(i);
          }
        } else {
          i->stepindex++;
          auto playerIdx = state->getCurrentPlayer();
          actStates.at(playerIdx).push_back(state);
          actGameStates.at(playerIdx).push_back(&*i);
          ++i;
        }
      }

      auto actForPlayer = [&](size_t playerIndex) {
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
          auto result = mctsPlayers.at(playerIndex)->actMcts(states);

          {
            double rps = mctsPlayers.at(playerIndex)->rolloutsPerSecond();
            std::unique_lock<std::mutex> lkStats(mutexStats_);
            auto& stats_s = stats_["Rollouts per second"];
            std::get<0>(stats_s) += 1;
            std::get<1>(stats_s) += rps;
            std::get<2>(stats_s) += rps * rps;
          }

          // Distribute results to the correct player/state and store
          // features for training.
          size_t offset = 0;
          for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
            size_t currentPlayerIndex = statePlayerSize[pi].first;
            size_t currentPlayerStates = statePlayerSize[pi].second;
            for (size_t i = 0; i != currentPlayerStates; ++i) {
              State* state = (State*)states.at(offset + i);
              torch::Tensor feat = getFeatureInTensor(*state);
              auto [policy, policyMask] =
                  getPolicyInTensor(*state, result.at(offset + i).mctsPolicy);

              auto& gameState = actGameStates.at(currentPlayerIndex).at(i);
              if (gameState->canResign) {
                float value = result.at(offset + i).rootValue;
                if (value < -0.95f) {
                  if (++gameState->resignCounter.at(currentPlayerIndex) >= 7) {
                    gameState->resigned = int(currentPlayerIndex);
                  }
                } else {
                  gameState->resignCounter.at(currentPlayerIndex) = 0;
                }
                int opponent = (currentPlayerIndex + 1) % 2;
                if (value > 0.95f) {
                  ++gameState->resignCounter.at(opponent);
                } else {
                  gameState->resignCounter.at(opponent) = 0;
                }
              }
              gameState->feat.at(currentPlayerIndex).push_back(feat);
              gameState->pi.at(currentPlayerIndex).push_back(policy);
              gameState->piMask.at(currentPlayerIndex).push_back(policyMask);

              state->forward(result.at(offset + i).bestAction);

              players_[currentPlayerIndex]->recordMove(state);

              auto now = std::chrono::steady_clock::now();
              double elapsed =
                  std::chrono::duration_cast<
                      std::chrono::duration<double, std::ratio<1, 1>>>(
                      now - gameState->prevMoveTime)
                      .count();
              gameState->prevMoveTime = now;

              {
                std::unique_lock<std::mutex> lkStats(mutexStats_);
                auto& stats_s = stats_["Move Duration (seconds)"];
                std::get<0>(stats_s) += 1;
                std::get<1>(stats_s) += elapsed;
                std::get<2>(stats_s) += elapsed * elapsed;
              }

              // fmt::printf("game in progress: %s\n", state->history());
            }
            offset += currentPlayerStates;
          }

          states.clear();
        }
      };

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
// std::unique_lock<std::mutex> lk(terminateMutex_);
#ifdef DEBUG_GAME
      std::cout << "Thread " << thread_id << ", game " << this
                << ": not terminating - run another iteration. " << std::endl;
#endif
      bool aHuman =
          std::any_of(players_.begin(), players_.end(),
                      [](const std::shared_ptr<mcts::Player>& player) {
                        return player->isHuman();
                      });
      if (aHuman && state_->stochasticReset()) {
        std::string line;
        std::cout << "Random outcome ?" << std::endl;
        std::cin >> line;
        state_->forcedDice = std::stoul(line, nullptr, 0);
      }
      auto randint = [&](int n) {
        return std::uniform_int_distribution<int>(0, n - 1)(rng_);
      };
      reset();
      int stepindex = 0;
      auto start = std::chrono::system_clock::now();
      resignCounter_.resize(players_.size());
      canResign_ = !evalMode && players_.size() == 2 && randint(6) != 0;
      resigned_ = -1;
      while (!state_->terminated() && resigned_ == -1) {
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
      }
      auto end = std::chrono::system_clock::now();
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
                      [](const std::shared_ptr<mcts::Player>& player) {
                        return player->isTP();
                      })) {
        state_->errPrintCurrentBoard();
      } /*else {
        state_->printCurrentBoard();
      }*/

      if (resigned_ != -1) {
        for (size_t idx = 0; idx != players_.size(); ++idx) {
          result_.at(idx) = int(idx) == resigned_ ? 1 : -1;
        }
      } else {
        result_[0] = state_->getReward(0);
        if (players_.size() > 1) {
          result_[1] = state_->getReward(1);
        }
      }

      if (!evalMode) {
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ", game " << this
                  << ": sending trajectory... " << std::endl;
#endif
        setReward(*state_, resigned_);
        sendTrajectory();
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ", game " << this
                  << ": trajectory sent... " << std::endl;
#endif
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
      auto action = *(state_->GetLegalActions().at(index).get());
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
    auto action = *(state_->GetLegalActions().at(index).get());
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
    auto action = *(state_->GetLegalActions().at(index).get());
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
    mcts::MctsResult result = mctsPlayer->actMcts(*state_);
    lastMctsValue_ = result.rootValue;

    if (canResign_ && result.rootValue < -0.95f) {
      if (++resignCounter_.at(playerIdx) >= 2) {
        resigned_ = playerIdx;
      }
    }

    // store feature for training
    if (!evalMode) {
      torch::Tensor feat = getFeatureInTensor(*state_);
      auto [policy, policyMask] = getPolicyInTensor(*state_, result.mctsPolicy);
      feature_[playerIdx].pushBack(std::move(feat));
      pi_[playerIdx].pushBack(std::move(policy));
      piMask_[playerIdx].pushBack(std::move(policyMask));
    }

    // std::cout << ">>>>actual act" << std::endl;
    _Action action = *(state_->GetLegalActions().at(result.bestAction));
    lastAction_ = state_->actionDescription(action);
    bool noHuman =
        std::none_of(players_.begin(), players_.end(),
                     [](const std::shared_ptr<mcts::Player>& player) {
                       return player->isHuman();
                     });
    if (!state_->isStochastic()) {
      if (!noHuman) {
        std::cout << "Performing action "
                  << state_->performActionDescription(
                         *state_->GetLegalActions().at(result.bestAction))
                  << "\n";
      }
    } else if (!noHuman) {
      std::string line;
      std::cout << "Performing action "
                << state_->performActionDescription(
                       *state_->GetLegalActions().at(result.bestAction))
                << "\n";
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
    }
    state_->forward(result.bestAction);
  }
}

void Game::setReward(const State& state, int resigned) {
  for (int i = 0; i < (int)players_.size(); ++i) {
    assert(v_[i].len() <= pi_[i].len() && pi_[i].len() == feature_[i].len());
    while (v_[i].len() < pi_[i].len()) {
      torch::Tensor reward = torch::zeros({1}, torch::kFloat32);
      reward[0] = resigned == -1 ? state.getReward(i) : i == resigned ? -1 : 1;
      v_[i].pushBack(std::move(reward));
    }
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
  if (feature_[playerId].prepareForSend()) {
    bool sendPi = pi_[playerId].prepareForSend();
    bool sendPiMask = piMask_[playerId].prepareForSend();
    bool sendV = v_[playerId].prepareForSend();
    assert(sendPi && sendV && sendPiMask);
    return true;
  }
  bool sendPi = pi_[playerId].prepareForSend();
  bool sendPiMask = piMask_[playerId].prepareForSend();
  bool sendV = v_[playerId].prepareForSend();
  assert((!sendPi) && (!sendV) && (!sendPiMask));
  return false;
}
