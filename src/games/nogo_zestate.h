/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "game.h"

typedef unsigned short Coord;

#include "nogo_game.h"
#include "nogo_state.h"
#include "time.h"
#include <iostream>
#include <random>
#include <string>
#include <vector>
//#include "base/common.h"

//#include "breakthrough.h"

class StateForNogo : public State {
 public:
  StateForNogo()
      : State() {
    Initialize();
    _history = 0;
    _outFeatures = false;
  }

  virtual ~StateForNogo() {
  }

  virtual void Initialize() override {
    // People implementing classes should not have much to do in _moves; just
    // _moves.clear().
    _moves.clear();

    _featSize[0] = 3;
    _featSize[1] = 9;
    _featSize[2] = 9;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).
    _actionSize[0] = 1;
    _actionSize[1] = 9;
    _actionSize[2] = 9;

    // _hash is an unsigned int, it has to be *unique*.
    _hash = 0;
    _nogoGame.Reset();
    _status = GameStatus::player0Turn;
    // std::cout << "restart!" << std::endl;
    // _features is a vector representing the current state. It can
    // (must...) be large for complex games; here just one number
    // between 0 and 1. trivial case in dimension 1.
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
    for (int i = 0; i < _features.size(); i++)
      _features[i] = 0.;
    clearActions();
    for (int i = 0; i < 9; i++)
      for (int j = 0; j < 9; j++) {
        addAction(0, i, j);
      }
    fillFullFeatures();
  }

  virtual std::unique_ptr<tree_search::State> clone() const override {
    auto newState = std::make_unique<StateForNogo>();
    *newState = *this;
    return newState;
  }

  // The action just decreases the distance and swaps the turn to play.
  virtual void ApplyAction(const _Action& action) override {
    assert(_status != GameStatus::player0Win);
    assert(_status != GameStatus::player1Win);
    NoGoAction nogoAction(_nogoGame.GetTurnPlayer(), action.GetHash());
    if (_nogoGame.GetTurnPlayer() == PLAYER_0) {
      _features[action.GetHash()] = 1.;
      for (int i = 0; i < 81; i++)
        _features[81 * 2 + i] = 1.;
    } else {
      _features[9 * 9 + action.GetHash()] = 1.;
      for (int i = 0; i < 81; i++)
        _features[81 * 2 + i] = 0.;
    }
    if (!_nogoGame.IsLegalAction(nogoAction)) {
      // if (true) {
      std::cerr << " before move" << std::endl;
      _nogoGame.ShowState();
      std::cerr << " the proposed action " << nogoAction.ToString()
                << " is legal ? " << _nogoGame.IsLegalAction(nogoAction)
                << std::endl;
      _nogoGame.PlayAction(nogoAction);
      std::cerr << " after move" << std::endl;
      _nogoGame.ShowState(); /*assert(false);*/
    } else {
      _nogoGame.PlayAction(nogoAction);
    }
    // let us remove the nogoAction from legal actions
    clearActions();
    auto legal_actions = _nogoGame.GetLegalActions();
    int index = 0;
    for (const auto& action : legal_actions) {
      addAction(0, action.GetPosition() % 9, action.GetPosition() / 9);
    }
    //   _nogoGame.ShowState();
    // std::cerr << " number of legal actions : " << _NewlegalActions.size() <<
    // std::endl;
    // first channel: black stones.
    // second channel: white stones.
    // third channel: 0 if player  black to play, 1 otherwise.
    // assert(false);
    if (_NewlegalActions.size() == 0) {
      // if (_nogoGame.IsTerminalState()) {
      assert(_nogoGame.IsTerminalState());
      if (_status == GameStatus::player0Turn)
        _status = GameStatus::player0Win;
      else
        _status = GameStatus::player1Win;
      //     _nogoGame.ShowState();

      assert((_status == GameStatus::player0Win) ==
             (_nogoGame.GetWinPlayer() == PLAYER_0));
      assert((_status == GameStatus::player1Win) ==
             (_nogoGame.GetWinPlayer() == PLAYER_1));
    } else {
      if (_status == GameStatus::player0Turn)
        _status = GameStatus::player1Turn;
      else
        _status = GameStatus::player0Turn;
    }
    assert(_status == GameStatus::player1Win ||
           _status == GameStatus::player0Win || _NewlegalActions.size() > 0);
    //      std::cerr << " play 0 wins:" << (_status == GameStatus::player0Win)
    //      << std::endl;
    //     std::cerr << " play 1 wins:" << (_status == GameStatus::player1Win)
    //     << std::endl;
    fillFullFeatures();
  }

  // For this trivial example we just compare to random play. Ok, this is not
  // really a good action.
  // By the way we need a good default DoGoodAction, e.g. one-ply at least.
  // FIXME
  virtual void DoGoodAction() override {

    int i = rand() % _NewlegalActions.size();
    _Action a = *(_NewlegalActions[i].get());
    ApplyAction(a);
  }

 private:
  NoGoGame _nogoGame;
};
