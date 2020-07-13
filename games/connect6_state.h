/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author 1. CHEN,SHIH-YU leo03164@gmail.com
// Author 2. CHIU,HSIEN-TUNG yumjelly@gmail.com
#pragma once

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "../core/state.h"
#include "connect6.h"

#include <fmt/printf.h>

namespace Connect6{
  
const int StateForConnect6NumActions = 19 * 19 * 3;

class ActionForConnect6 : public ::_Action {
 public:
  ActionForConnect6(int x, int y)
      : _Action() {
    _loc[0] = 0;
    _loc[1] = x;
    _loc[2] = y;
    _hash = x + y * 19;
  }  // step is 2 or 3.
};

template<int version = 2>
class StateForConnect6 : public ::State, C6Board {
 public:
  int twice;
  int firhand;

  StateForConnect6(int seed)
      : State(seed) {}
  
  virtual void Initialize() override {
    // printf("Initialize\n");
    // People implementing classes should not have much to do in _moves; just
    // _moves.clear().
    _moves.clear();

    const int StateForConnect6X = version == 2 ? 2 + 1 : 2 * 6 + 1;
    const int StateForConnect6Y = 19;
    const int StateForConnect6Z = 19;

    _featSize[0] = StateForConnect6X;
    _featSize[1] = StateForConnect6Y;
    _featSize[2] = StateForConnect6Z;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).
    _actionSize[0] = 1;
    _actionSize[1] = 19;
    _actionSize[2] = 19;

    // _hash is an unsigned int, it has to be *unique*.
    _hash = 0;
    _status = GameStatus::player1Turn;

    _features.resize(StateForConnect6X * StateForConnect6Y * StateForConnect6Z);

    twice = 0;
    firhand = 1;

    init();
    findFeatures();
    findActions();
    fillFullFeatures();
  }

  virtual std::unique_ptr<mcts::State> clone_() const override {
    return std::make_unique<StateForConnect6>(*this);
  }

  void findActions() {
    // printf("findActions\n");
    C6Move moves[C6MaxLegalMoves];
    int nb = legalMoves(moves);

    _legalActions.clear();
    for (int i = 0; i < nb; i++) {
      int x = moves[i].x;
      int y = moves[i].y;

      _legalActions.push_back(std::make_shared<ActionForConnect6>(x, y));
      _legalActions[i]->SetIndex(i);
    }
  }

  void findFeatures() {

    // printf("findFeatures\n");
    if ((_status == GameStatus::player0Win) ||
        (_status == GameStatus::player1Win) || (_status == GameStatus::tie)) {
      return;
    }

    if (version == 2) {
      std::fill(_features.begin() + 2 * C6Dx * C6Dy, _features.end(), twice || firhand ? 1.0f : 0.0f);
    } else {
      std::vector<float> old(_features);
      for (int i = 0; i < C6Dx * C6Dy * 2; i++)
        _features[i] = 0;
      for (int i = 0; i < C6Dx * C6Dy; i++)
        if (board[i % C6Dx][i / C6Dy] == C6Black)
          _features[i] = 1;
      for (int i = 0; i < C6Dx * C6Dy; i++)
        if (board[i % C6Dx][i / C6Dy] == C6White)
          _features[C6Dx * C6Dy + i] = 1;

      std::copy(old.begin(), old.begin() + 3610, _features.begin() + 722);

      // 4332-4693
      std::fill(_features.begin() + 4332, _features.end(), getCurrentPlayer());
    }
  }

  virtual void ApplyAction(const ::_Action& action) override {
    // printf("ApplyAction\n");

    C6Move m;
    // print(stdout);
    if (_status == GameStatus::player0Turn) {  // C6White
      m.color = C6White;
      m.x = action.GetY();
      m.y = action.GetZ();

      play(m);

      if (version == 2) {
        _features[m.x * C6Dy + m.y + C6Dx * C6Dy * 0] = 1.0f;
      }

      bool hasWon = won(m);
      if (hasWon) {
        _status = GameStatus::player0Win;
      } else {
        findActions();
        if (nb == 0) {
          _status = GameStatus::tie;
        } else {
          if (twice == 0) {
            twice = 1;
          } else if (twice == 1) {
            twice = 0;
            _status = GameStatus::player1Turn;
          }
        }
      }
    } else if (_status == GameStatus::player1Turn) {
      // C6Black
      m.color = C6Black;
      m.x = action.GetY();
      m.y = action.GetZ();

      play(m);

      if (version == 2) {
        _features[m.x * C6Dy + m.y + C6Dx * C6Dy * 1] = 1.0f;
      }

      bool hasWon = won(m);
      if (hasWon) {
        _status = GameStatus::player1Win;
      } else {
        findActions();
        if (nb == 0) {
          _status = GameStatus::tie;
        } else {
          if (firhand) {
            _status = GameStatus::player0Turn;
            firhand = 0;
          } else {
            if (twice == 0)
              twice = 1;
            else if (twice == 1) {
              twice = 0;
              _status = GameStatus::player0Turn;
            }
          }
        }
      }

    }
    findFeatures();
    _hash = hash;
    fillFullFeatures();
  }

  virtual void DoGoodAction() override {
    return DoRandomAction();
  }

  std::string stateDescription() const override {
    std::string s;
    s += fmt::sprintf("   ");
    for (int k = 65; k < 84; k++)
      s += fmt::sprintf("%c ", k);
    s += fmt::sprintf("\n");
    for (int i = 0; i < C6Dx; i++) {
      if (C6Dx - i < 10)
        s += fmt::sprintf("%d  ", C6Dx - i);
      else
        s += fmt::sprintf("%d ", C6Dx - i);
      for (int j = 0; j < C6Dy; j++) {
        if (board[C6Dx - 1 - i][j] == C6Black)
          s += "X ";
        else if (board[C6Dx - 1 - i][j] == C6White)
          s += "O ";
        else
          s += ". ";
      }
      s += "\n";
    }
    return s;
  }

  std::string actionDescription(const _Action &action) const override {
    return std::string(1, 'A' + action.GetZ()) + std::to_string(action.GetY() + 1);
  }

  int parseAction(const std::string &str) override {
    if (str.size() < 2) {
      return -1;
    }
    int z = str[0] - 'A';
    if (z < 0 || z >= 19) {
      z = str[0] - 'a';
      if (z < 0 || z >= 19) {
        return -1;
      }
    }
    int y = std::atoi(str.data() + 1) - 1;
    if (y < 0 || y >= 19) {
      return -1;
    }
    for (auto& a : _legalActions) {
      if (a->GetZ() == z && a->GetY() == y) {
        return a->GetIndex();
      }
    }
    return -1;
  }

  int humanInputAction(
      std::function<std::optional<int>(std::string)> specialAction) {
    std::cout << "Current board:" << std::endl << stateDescription() << std::endl;
    std::string str;
    int index = -1;
    while (index < 0) {
      std::cout << "Input action: ";
      std::getline(std::cin, str);
      index = parseAction(str);
      if (index < 0) {
        if (auto r = specialAction(str); r)
          return *r;
        std::cout << "invalid input, try again." << std::endl;
      }
    }
    return index;
  }

};

}
