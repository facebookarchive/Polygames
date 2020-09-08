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

#include "../core/game.h"
#include "tristan_nogo.h"

const int StateForTristannogoNumActions = Dx * Dy;
const int StateForTristannogoX = 5;
const int StateForTristannogoY = Dx;
const int StateForTristannogoZ = Dy;
const int NogoMaxLegalMoves = Dx * Dy;

class ActionForTristannogo : public _Action {
 public:
  // each action has a position (_x[0], _x[1], _x[2])
  // here for Tristannogo, there is (0, 0, 0) and (1, 0, 0),
  // corresponding to steps 2 and 3 respectively.
  ActionForTristannogo(int x, int y)
      : _Action() {
    _loc[0] = 0;
    _loc[1] = x;
    _loc[2] = y;
    _hash = (x + y * Dx);
  }
};

class StateForTristannogo : public core::State, NogoBoard {
 public:
  StateForTristannogo(int seed)
      : State(seed) {
    Initialize();
  }

  virtual ~StateForTristannogo() {
  }

  virtual void Initialize() override {
    // People implementing classes should not have much to do in _moves; just
    // _moves.clear().
    _moves.clear();
    // std::cout << "OTGTristannogo initialize" << std::endl;

    // the features are just one number between 0 and 1 (the distance,
    // normalized).
    _featSize[0] = StateForTristannogoX;
    _featSize[1] = StateForTristannogoY;
    _featSize[2] = StateForTristannogoZ;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).
    _actionSize[0] = 1;
    _actionSize[1] = Dx;
    _actionSize[2] = Dy;

    // _hash is an unsigned int, it has to be *unique*.
    _hash = 0;
    _status = GameStatus::player0Turn;
    // std::cout << "restart!" << std::endl;
    // _features is a vector representing the current state. It can
    // (must...) be large for complex games; here just one number
    // between 0 and 1. trivial case in dimension 1.
    _features.resize(StateForTristannogoX * StateForTristannogoY *
                     StateForTristannogoZ);
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

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<StateForTristannogo>(*this);
  }

  void findActions(int color) {
    NogoMove moves[NogoMaxLegalMoves];
    int nb = legalNogoMoves(color, moves);

    _legalActions.clear();
    for (int i = 0; i < nb; i++) {
      int x = moves[i].inter % Dx;
      int y = moves[i].inter / Dx;
      _legalActions.push_back(std::make_shared<ActionForTristannogo>(x, y));
      _legalActions[i]->SetIndex(i);
    }
  }

  void findFeatures() {
    if ((_status == GameStatus::player0Win) ||
        (_status == GameStatus::player1Win))
      return;
    if (_status == GameStatus::player0Turn) {  // Black
      for (int i = 0; i < 5 * Dx * Dy; i++)
        _features[i] = 0;
      for (int i = 0; i < Dx * Dy; i++)
        if (board[interMove[i]] == Black)
          _features[i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
        if (board[interMove[i]] == White)
          _features[Dx * Dy + i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
	if (legal(interMove[i], Black))
	    _features[3 * Dx * Dy + i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
	if (legal(interMove[i], White))
	    _features[4 * Dx * Dy + i] = 1;
    } else {
      assert(_status == GameStatus::player1Turn);  // White
      for (int i = 0; i < 5 * Dx * Dy; i++)
        _features[i] = 0;
      for (int i = 0; i < Dx * Dy; i++)
        if (board[interMove[i]] == Black)
          _features[i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
        if (board[interMove[i]] == White)
          _features[Dx * Dy + i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
	_features[2 * Dx * Dy + i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
	if (legal(interMove[i], Black))
	    _features[3 * Dx * Dy + i] = 1;
      for (int i = 0; i < Dx * Dy; i++)
	if (legal(interMove[i], White))
	    _features[4 * Dx * Dy + i] = 1;
    }
  }
  // The action just decreases the distance and swaps the turn to play.
  virtual void ApplyAction(const _Action& action) override {

    NogoMove m;
    // print(stdout);
    if (_status == GameStatus::player0Turn) {  // Black
      m.color = Black;
      m.inter = action.GetY() + Dx * action.GetZ();
      play(m);
      findActions(White);
      if (won(Black))
        _status = GameStatus::player0Win;
      else
        _status = GameStatus::player1Turn;
    } else {  // White
      m.color = White;
      m.inter = action.GetY() + Dx * action.GetZ();
      play(m);
      findActions(Black);
      if (won(White))
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
    std::string s ("  0 1 2 3 4 5 6 7 8\n");
    for (int i = 0; i < Dy; i++) {
      s += std::to_string (i);
      for (int j = 0; j < Dx; j++) {
        if (board[interMove[i * Dy + j]] == Black)
           s += " @";
	else if (board[interMove[i * Dy + j]] == White)
	  s += " O";
        else if (board[interMove[i * Dy + j]] == Empty)
          s += " +";
      }
      s += " \n";
    }
    s += " \n";
    return s;
  }
};
