/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../core/state.h"

class ActionForConnectFour : public _Action {
 public:
  ActionForConnectFour(int x, size_t index)
      : _Action() {
    _loc[0] = x;
    _loc[1] = 0;
    _loc[2] = 0;
    _hash = (uint32_t)x;
    _i = (int)index;
  }
};

class StateForConnectFour : public State {
 public:
  StateForConnectFour(int seed)
      : State(seed) {
  }

  virtual void Initialize() override {
    _moves.clear();
    _hash = 2166136261u;
    _status = GameStatus::player0Turn;
    _featSize[0] = 3;
    _featSize[1] = boardHeight;
    _featSize[2] = boardWidth;
    _actionSize[0] = boardWidth;
    _actionSize[1] = 1;
    _actionSize[2] = 1;
    _features.clear();
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
    std::fill(_features.begin(), _features.end(), 1.0f);
    board.clear();
    board.resize(boardWidth * boardHeight, 0);
    height.clear();
    height.resize(boardWidth, 0);
    featurize();
    findActions();
    fillFullFeatures();
  }

  virtual std::unique_ptr<mcts::State> clone_() const override {
    return std::make_unique<StateForConnectFour>(*this);
  }

  virtual void printCurrentBoard() const override {
    std::cout << "printing board" << std::endl << std::flush;
    for (int r = boardHeight - 1; r >= 0; --r) {
      std::cout << "|";
      for (int c = 0; c < boardWidth; ++c) {
        auto val = board[r * boardWidth + c];
        if (val == 0) {
          std::cout << " ";
        } else if (val == 1) {
          std::cout << "X";
        } else if (val == 2) {
          std::cout << "O";
        } else {
          assert(false);
        }
        std::cout << "|";
      }
      std::cout << std::endl;
    }
  }

  void featurize() {
    int player = 1 + getCurrentPlayer();
    int otherPlayer = player == 1 ? 2 : 1;
    for (int i = 0; i != (int)board.size(); ++i) {
      int v = board[i];
      _features[i] = v == player;
      _features[board.size() + i] = v == otherPlayer;
    }
  }

  void findActions() {
    _legalActions.clear();
    for (int i = 0; i != boardWidth; ++i) {
      if (height[i] != boardHeight) {
        _legalActions.push_back(
            std::make_shared<ActionForConnectFour>(i, _legalActions.size()));
      }
    }
  }

  virtual void ApplyAction(const _Action& action) override {
    int x = action.GetX();
    int y = height.at(x);
    ++height[x];
    int player = 1 + getCurrentPlayer();
    size_t index = x + y * boardWidth;
    board.at(index) = player;
    _hash ^= index;
    _hash *= 16777619u;
    auto count = [&](int dx, int dy) {
      int nx = x + dx;
      int ny = y + dy;
      int r = 0;
      int stride = dx + dy * boardWidth;
      size_t nIndex = index + stride;
      while (nx >= 0 && nx < boardWidth && ny >= 0 && ny < boardHeight &&
             board.at(nIndex) == player) {
        ++r;
        nIndex += stride;
        nx += dx;
        ny += dy;
      }
      return r;
    };
    bool won = count(-1, 0) + count(1, 0) >= 3;
    won |= count(0, -1) + count(0, 1) >= 3;
    won |= count(-1, -1) + count(1, 1) >= 3;
    won |= count(1, -1) + count(-1, 1) >= 3;
    if (won) {
      _status = player == 1 ? GameStatus::player0Win : GameStatus::player1Win;
    } else {
      featurize();
      findActions();
      if (_legalActions.empty()) {
        _status = GameStatus::tie;
      } else {
        _status =
            player == 1 ? GameStatus::player1Turn : GameStatus::player0Turn;
      }
    }
    fillFullFeatures();
  }

  virtual void DoGoodAction() override {
    return DoRandomAction();
  }

  const int boardWidth = 7;
  const int boardHeight = 6;
  std::vector<char> board;
  std::vector<char> height;
};
