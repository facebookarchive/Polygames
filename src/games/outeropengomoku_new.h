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

class StateForOOGomoku : public core::State {
 public:
  StateForOOGomoku(int seed)
      : State(seed) {
  }

  virtual void Initialize() override {
    _moves.clear();
    _hash = 2166136261u;
    _status = GameStatus::player0Turn;
    _featSize[0] = 3;
    _featSize[1] = boardHeight;
    _featSize[2] = boardWidth;
    _actionSize[0] = 1;
    _actionSize[1] = boardWidth;
    _actionSize[2] = boardHeight;
    _features.clear();
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
    std::fill(_features.begin(), _features.end(), 1.0f);
    board.clear();
    board.resize(boardWidth * boardHeight, 0);
    FirstMove = 1;
    featurize();
    findActions();
    fillFullFeatures();
  }

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<StateForOOGomoku>(*this);
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
    clearActions();
    if (FirstMove) {
      FirstMove = 0;
      for (int i = 0; i < boardWidth; ++i) {
        for (int j = 0; j < boardHeight; ++j) {
          if ((i == 0 || i == 1 || i == 13 || i == 14 || j == 0 || j == 1 ||
               j == 13 || j == 14))
            addAction(0, i, j);
        }
      }
    } else {
      for (int i = 0; i < boardWidth; ++i) {
        for (int j = 0; j < boardHeight; ++j) {
          auto pos = i + j * boardHeight;
          if (board[pos] == 0)
            addAction(0, i, j);
        }
      }
    }
  }

  virtual void ApplyAction(const _Action& action) override {
    int x = action.GetY();
    int y = action.GetZ();
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
    bool won = count(-1, 0) + count(1, 0) >= 4;
    won |= count(0, -1) + count(0, 1) >= 4;
    won |= count(-1, -1) + count(1, 1) >= 4;
    won |= count(1, -1) + count(-1, 1) >= 4;
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

  static const int boardWidth = 15;
  static const int boardHeight = 15;
  bool FirstMove;
  std::vector<char> board;
};
