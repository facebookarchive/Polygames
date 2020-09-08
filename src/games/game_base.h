/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CZF_GAME_GAME_BASE_H_
#define CZF_GAME_GAME_BASE_H_

#include "game_action.h"
#include "game_player.h"
#include "game_state.h"
#include <string>
#include <vector>

template <typename StateType, typename ActionType> class GameBase {
 public:
  typedef ActionType Action;
  typedef StateType State;
  typedef std::vector<Action> History;

 protected:
  State state_;
  History history_;
  PLAYER turn_player_;
  bool is_terminal_;
  PLAYER win_player_;

 public:
  GameBase() {
    Reset();
  }
  ~GameBase() {
    ;
  }
  void Reset() {
    state_.Reset();
    history_.clear();
    turn_player_ = PLAYER_0;
    is_terminal_ = false;
    win_player_ = PLAYER_NULL;
  }
  PLAYER GetTurnPlayer() const {
    return turn_player_;
  }
  int GetGameLength() const {
    return history_.size();
  }
  void SetTurnPlayer(PLAYER turn_player) {
    turn_player_ = turn_player;
  }
  void GetHistory(History& history) const {
    history = history_;
  }
  Action GetLastAction() {
    if (history_.size() > 0)
      return history_.back();
    else
      return Action();
  }
  PLAYER GetWinPlayer() {
    return win_player_;
  }

  // pure virtual functions

  // need to save win player if game is terminal
  virtual bool PlayAction(Action action) = 0;
  virtual bool IsTerminalState() = 0;
  virtual bool IsLegalAction(Action action) = 0;
  virtual std::vector<Action> GetLegalActions() = 0;
  virtual std::vector<bool> GetIsLegalAction() = 0;
};

#endif  // CZF_GAME_GAME_BASE_H_
