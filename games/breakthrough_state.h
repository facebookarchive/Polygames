/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "time.h"
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "breakthrough.h"
//#include "game.h"
#include "../core/state.h"

const int StateForBreakthroughNumActions = 64 * 3;
const int StateForBreakthroughX = 2;
const int StateForBreakthroughY = 8;
const int StateForBreakthroughZ = 8;
const int BTMaxLegalMoves = 48;

class ActionForBreakthrough : public _Action {
 public:
  // each action has a position (_x[0], _x[1], _x[2])
  // here for Breakthrough, there is (0, 0, 0) and (1, 0, 0),
  // corresponding to steps 2 and 3 respectively.
  ActionForBreakthrough(int x, int y, int direction)
      : _Action() {
    _loc[0] = direction;
    _loc[1] = x;
    _loc[2] = y;
    _hash = (x + y * 8) * 3 + direction;
  }
};

template<bool fixedPolicy = true>
class StateForBreakthrough : public State, BTBoard {
 public:
  StateForBreakthrough(int seed)
      : State(seed) {
    Initialize();
  }

  virtual ~StateForBreakthrough() {
  }

  int parseAction(const std::string& str) {
  int x = int(str[0]) - 48;
  int y = int(str[1]) - 48;
  int z = int(str[2]) - 48;
  try {
    for (unsigned k=0; k<_legalActions.size(); k++)
      if (_legalActions[k]->GetX() == x and _legalActions[k]->GetY() == y and _legalActions[k]->GetZ() == z)
        return k;
  }
  catch (...) {
    std::cout << "failed to parse action" << std::endl;
  }
  return -1;
}

  virtual void Initialize() override {
    // People implementing classes should not have much to do in _moves; just
    // _moves.clear().
    _moves.clear();
    // std::cout << "OTGBreakthrough initialize" << std::endl;

    // the features are just one number between 0 and 1 (the distance,
    // normalized).
    _featSize[0] = StateForBreakthroughX;
    _featSize[1] = StateForBreakthroughY;
    _featSize[2] = StateForBreakthroughZ;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).
    _actionSize[0] = 3;
    _actionSize[1] = 8;
    _actionSize[2] = 8;

    // _hash is an unsigned int, it has to be *unique*.
    _hash = 0;
    _status = GameStatus::player0Turn;
    // std::cout << "restart!" << std::endl;
    // _features is a vector representing the current state. It can
    // (must...) be large for complex games; here just one number
    // between 0 and 1. trivial case in dimension 1.
    _features.resize(StateForBreakthroughX * StateForBreakthroughY *
                     StateForBreakthroughZ);
    /*
        // _features[:_hash] = 1
        for (int i = 0; i < DISTANCE; i++) {
          _features[i] = (float(_hash) > float(i)) ? 1. : 0.;
        }
    */
    init();
    findFeatures();
    findActions(White);
    fillFullFeatures();
  }

  virtual std::unique_ptr<mcts::State> clone_() const override {
    return std::make_unique<StateForBreakthrough>(*this);
  }

  void findActions(int color) {
    BTMove moves[BTMaxLegalMoves];
    int nb = legalBTMoves(color, moves);

    _legalActions.clear();
    for (int i = 0; i < nb; i++) {
      int x = moves[i].x;
      int y = moves[i].y;
      int dir = 2;
      if (moves[i].x1 == x - 1)
        dir = 0;
      else if (moves[i].x1 == x)
        dir = 1;
      _legalActions.push_back(
          std::make_shared<ActionForBreakthrough>(x, y, dir));
      _legalActions[i]->SetIndex(i);
    }
  }

  void findFeatures() {
    if ((_status == GameStatus::player0Win) ||
        (_status == GameStatus::player1Win))
      return;
    // init
    const size_t numFeats =
        StateForBreakthroughX * StateForBreakthroughY * StateForBreakthroughZ;
    std::fill(_features.begin(), _features.begin() + numFeats, 0.);
    for (int i = 0; i < 64; i++) {
      auto value = fixedPolicy ? board[i / 8][i % 8] : board[i % 8][i / 8];
      if (value == Black)
        _features[i] = 1;
      else if (value == White)
        _features[64 + i] = 1;
    }
  }
  // The action just decreases the distance and swaps the turn to play.
  virtual void ApplyAction(const _Action& action) override {

    BTMove m;
    // print(stdout);
    if (_status == GameStatus::player0Turn) {  // White
      m.color = White;
      m.x = action.GetY();
      m.y = action.GetZ();
      if (action.GetX() == 0) {
        m.x1 = action.GetY() - 1;
        m.y1 = action.GetZ() - 1;
      } else if (action.GetX() == 1) {
        m.x1 = action.GetY();
        m.y1 = action.GetZ() - 1;
      } else if (action.GetX() == 2) {
        m.x1 = action.GetY() + 1;
        m.y1 = action.GetZ() - 1;
      }
      play(m);
      findActions(Black);
      if (won(White))
        _status = GameStatus::player0Win;
      else
        _status = GameStatus::player1Turn;
    } else {
      // Black
      m.color = Black;
      m.x = action.GetY();
      m.y = action.GetZ();
      if (action.GetX() == 0) {
        m.x1 = action.GetY() - 1;
        m.y1 = action.GetZ() + 1;
      } else if (action.GetX() == 1) {
        m.x1 = action.GetY();
        m.y1 = action.GetZ() + 1;
      } else if (action.GetX() == 2) {
        m.x1 = action.GetY() + 1;
        m.y1 = action.GetZ() + 1;
      }
      play(m);
      findActions(White);
      if (won(Black))
        _status = GameStatus::player1Win;
      else
        _status = GameStatus::player0Turn;
    }
    findFeatures();
    _hash = hash;
    fillFullFeatures();
  }

  // For this trivial example we just compare to random play. Ok, this is not
  // really a good action.
  // By the way we need a good default DoGoodAction, e.g. one-ply at least.
  // FIXME
  virtual void DoGoodAction() override {

    int i = rand() % _legalActions.size();
    _Action a = *(_legalActions[i].get());
    ApplyAction(a);
  }

  std::string stateDescription() const override {
    std::string s (" 0 1 2 3 4 5 6 7\n");
    for (int i = 0; i < BTDy; i++) {
      s += std::to_string (i);
      for (int j = 0; j < BTDx; j++)
        if (board[j][i] == Empty)
          s += " +";
        else if (board[j][i] == Black)
          s += " @";
        else
          s += " O";
      s += " \n";
    }
    s += " \n";
    return s;
  }
};
