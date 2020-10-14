/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "game_base.h"
#include "nogo_action.h"
#include "nogo_state.h"
#include <fstream>
#include <iostream>

class NoGoGame : public GameBase<NoGoState, NoGoAction> {
 public:
  NoGoGame();
  ~NoGoGame();

  NoGoGame& operator=(const NoGoGame& rhs);
  bool IsTerminalState() override;
  bool IsLegalAction(Action action) override;
  std::vector<Action> GetLegalActions() override;

  std::vector<bool> GetIsLegalAction() override;
  bool PlayAction(Action action) override;

  NoGoBitBoard GetIllegalBitBoard();
  NoGoState GetNoGoState();
  PLAYER GetPositionPlayer(Position position);
  bool IsLegalAction(PLAYER player, Position position);
  bool IsLegalAction(Position position);

  void ShowState();
  std::string ShowBoard();
  std::string GetGtpResultString();
  std::string ToSgfFilePrefix(std::string player0,
                              std::string player1,
                              std::string sEventName);
  std::string ToMoveString(bool with_semicolon);
  std::string ToMoveString(bool with_semicolon,
                           bool with_comments,
                           std::vector<std::string>& comments);
  std::string ToSgfFileString(std::string player0,
                              std::string player1,
                              std::string sEventName,
                              bool with_semicolon);
  std::string ToSgfFileString(std::string player0,
                              std::string player1,
                              std::string sEventName,
                              bool with_semicolon,
                              bool with_comments,
                              std::vector<std::string>& comments);
};
#include "nogo_game.cc"
