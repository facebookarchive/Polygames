/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "actor.h"
#include "mcts.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <random>

using namespace mcts;
std::atomic<int> seed(0);

class TicTacToeState : public State {
 public:
  TicTacToeState()
      : State()
      , rng_(seed++) {
    board.resize(9);
    std::fill(board.begin(), board.end(), 0);
    currentPlayer = 1;
  }

  int getCurrentPlayer() const override {
    return currentPlayer;
  }

  uint64_t getHash() const override {
    return 0;
  }

  float getReward(int player) const override {
    int r = winner;
    if (r == 0)
      r = checkWinner();
    return r * player;
  };

  // bool isStochastic() const override {
  //   return false;
  // };

  std::vector<int> getLegalActions() const {
    std::vector<int> actions;
    for (int i = 0; i < 9; i++) {
      if (board[i] == 0) {
        actions.push_back(i);
      }
    }
    return actions;
  }

  float getRandomRolloutReward(int player) const override {
    int numRandomRollout = 100;
    int totalReward = 0;
    for (int i = 0; i < numRandomRollout; ++i) {
      TicTacToeState state;
      // std::cout << "start random rollout" << std::endl;
      // state.printState();
      state.board = board;
      state.currentPlayer = currentPlayer;
      state.moveIdx = moveIdx;
      while (!state.terminated()) {
        // state.printState();
        auto actions = state.getLegalActions();
        int idx = state.rng_() % actions.size();
        // std::cout << idx << "," << actions[idx] << ";;  ";
        state.forward(actions[idx]);
        // std::cout << "player " << player << ", action: "  << actions[idx] <<
        // std::endl;
        // state.printState();
      }
      // std::cout << "+++++++end of random rollout +++++++, winner: "
      //           << checkWinner() << std::endl;
      // state.winner = state.checkWinner();
      totalReward += state.checkWinner() * player;
    }
    return totalReward / (float)numRandomRollout;
  }

  bool operator==(const State&) const override {
    return false;
  }

  int getStepIdx() const override {
    return moveIdx;
  }

  std::unique_ptr<State> clone() const override {
    auto other = std::make_unique<TicTacToeState>();
    other->moveIdx = moveIdx;
    other->board = board;
    other->currentPlayer = currentPlayer;
    return other;
    // return std::make_unique<TicTacToeState>(other);
  }

  bool forward(const Action& a) override {
    assert(a >= 0 && a <= 8);
    if (board[a] != 0) {
      winner = -currentPlayer;
    }
    board[a] = currentPlayer;
    currentPlayer = -currentPlayer;
    moveIdx += 1;
    return true;
  }

  bool terminated() const override {
    return winner != 0 || checkWinner() != 0 || moveIdx == 9;
  }

  int at(int i, int j) const {
    // std::cout << i << j << std::endl;
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return board[i * 3 + j];
  }

  void checkSum(int sum, int* winner) const {
    if (sum == 3)
      *winner = 1;
    if (sum == -3)
      *winner = -1;
  }

  int checkWinner() const {
    int w = 0;
    int sum = 0;
    for (int i = 0; i < 3; i++) {
      sum = 0;
      for (int j = 0; j < 3; j++) {
        sum += at(i, j);
        checkSum(sum, &w);
      }
      sum = 0;
      for (int j = 0; j < 3; j++) {
        sum += at(j, i);
        checkSum(sum, &w);
      }
    }
    sum = 0;
    for (int i = 0; i < 3; i++) {
      sum += at(i, i);
      checkSum(sum, &w);
    }
    sum = 0;
    for (int i = 0; i < 3; i++) {
      sum += at(i, 2 - i);
      checkSum(sum, &w);
    }
    return w;
  }

  void printState() {
    std::cout << "PRINT STATE===" << std::endl;
    std::cout << "current player is " << currentPlayer << std::endl;
    for (int i = 0; i < 9; ++i) {
      std::cout << std::setw(2);
      std::cout << board[i] << " ";
      if (i % 3 == 2) {
        std::cout << std::endl;
      }
    }
    // std::cout << std::endl;
  }

  std::vector<int> board;
  int currentPlayer = 1;
  int winner = 0;
  int moveIdx = 0;
  std::mt19937 rng_;
};

class TestActor : public Actor {
 public:
  TestActor() {
  }

  PiVal evaluate(const State& s) override {
    const auto& state = dynamic_cast<const TicTacToeState*>(&s);
    const auto& actions = state->getLegalActions();
    std::unordered_map<Action, float> pi;

    for (size_t i = 0; i < actions.size(); ++i) {
      pi[actions[i]] = 1.0 / actions.size();
    }
    auto player = state->getCurrentPlayer();
    float value = state->getRandomRolloutReward(state->getCurrentPlayer());
    PiVal piVal(player, value, std::move(pi));
    return piVal;
  }
};

int main(int argc, char* argv[]) {
  // args are thread, rollouts
  assert(argc == 3);
  TicTacToeState state;
  MctsOption option;
  // option.numThread = 2;
  option.numRolloutPerThread = std::stoi(std::string(argv[2]));
  option.puct = 1.0;
  option.virtualLoss = 1.0;
  std::vector<std::unique_ptr<MctsPlayer>> players;

  for (int i = 0; i < 2; ++i) {
    players.push_back(std::make_unique<MctsPlayer>(option));
    for (int j = 0; j < std::stoi(std::string(argv[1])); ++j) {
      players.at(i)->addActor(std::make_shared<TestActor>());
    }
  }

  int i = 0;
  while (!state.terminated()) {
    int playerIdx = state.getCurrentPlayer() == 1 ? 0 : 1;
    MctsResult result = players.at(playerIdx)->actMcts(state);
    std::cout << "best action is " << result.bestAction << std::endl;
    state.forward(result.bestAction);
    state.printState();
    std::cout << "-----------" << std::endl;
    ++i;
    // if (i > 1) {
    //   break;
    // }
  }
  std::cout << "winner is " << state.checkWinner() << std::endl;
  assert(state.checkWinner() == 0);
  return 0;
}
