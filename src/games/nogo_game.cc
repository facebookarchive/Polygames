/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "nogo_game.h"

NoGoGame::NoGoGame() {
  Reset();
}
NoGoGame::~NoGoGame() {
  ;
}

bool NoGoGame::PlayAction(Action action) {
  if (is_terminal_)
    return false;
  if (!state_.IsLegalAction(action)) {
    is_terminal_ = true;
    win_player_ = !turn_player_;
    return false;
  }
  history_.push_back(action);
  turn_player_ = !action.GetPlayer();
  state_.PlayAction(action);
  return true;
}

NoGoGame& NoGoGame::operator=(const NoGoGame& rhs) {
  history_ = rhs.history_;
  turn_player_ = rhs.turn_player_;
  state_ = rhs.state_;
  is_terminal_ = rhs.is_terminal_;
  win_player_ = rhs.win_player_;
  return *this;
}

NoGoState NoGoGame::GetNoGoState() {
  return state_;
}

std::vector<NoGoGame::Action> NoGoGame::GetLegalActions() {
  std::vector<Action> legal_actions;
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (state_.IsLegalAction(turn_player_, i)) {
      legal_actions.push_back(Action(turn_player_, i));
    }
  }
  return legal_actions;
}

std::vector<bool> NoGoGame::GetIsLegalAction() {
  std::vector<bool> is_legal(kNOGO_GRIDS_NUM, false);
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (IsLegalAction(Action(turn_player_, i))) {
      is_legal[i] = true;
    }
  }
  return is_legal;
}

bool NoGoGame::IsTerminalState() {
  if (is_terminal_)
    return true;
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (state_.IsLegalAction(turn_player_, i))
      return false;
  }
  win_player_ = !turn_player_;
  is_terminal_ = true;
  return true;
}

NoGoBitBoard NoGoGame::GetIllegalBitBoard() {
  NoGoBitBoard bitBoard;
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (!state_.IsLegalAction(turn_player_, i)) {
      bitBoard.AddPosition(i);
    }
  }
  return bitBoard;
}

bool NoGoGame::IsLegalAction(Action action) {
  return state_.IsLegalAction(action);
}

bool NoGoGame::IsLegalAction(PLAYER player, Position position) {
  return state_.IsLegalAction(player, position);
}

bool NoGoGame::IsLegalAction(Position position) {
  return state_.IsLegalAction(turn_player_, position);
}

PLAYER NoGoGame::GetPositionPlayer(Position position) {
  return state_.GetPlayer(position);
}

void NoGoGame::ShowState() {
  if (GetTurnPlayer() == PLAYER_0)
    std::cerr << "PLAYER 0\n";
  if (GetTurnPlayer() == PLAYER_1)
    std::cerr << "PLAYER 1\n";
  state_.ShowBoard();
}

std::string NoGoGame::ShowBoard() {
  return state_.ToString();
}

std::string NoGoGame::GetGtpResultString() {
  if (!IsTerminalState()) {
    return "0";
  }
  if (GetTurnPlayer() == PLAYER_0) {
    return "W+R";
  } else {
    return "B+R";
  }
}

std::string NoGoGame::ToSgfFilePrefix(std::string player0,
                                      std::string player1,
                                      std::string event_name = "") {
  std::ostringstream oss;
  oss << "(;FF[4]CA[UTF-8]SZ[" << kNOGO_BOARD_SIZE << "]"
      << "KM[0]"
      << "EV[" << event_name << "]"
      << "PB[" << player0 << "]"
      << "PW[" << player1 << "]"
      << "RE[" << GetGtpResultString() << "]";
  return oss.str();
}

std::string NoGoGame::ToMoveString(bool with_semicolon = false) {
  std::ostringstream oss;
  for (size_t i = 0; i < history_.size(); i++) {
    if (with_semicolon)
      oss << ";";
    oss << history_[i].ToSgfString(with_semicolon);
  }
  return oss.str();
}

std::string NoGoGame::ToMoveString(bool with_semicolon,
                                   bool with_comments,
                                   std::vector<std::string>& comments) {
  std::ostringstream oss;
  for (size_t i = 0; i < history_.size(); i++) {
    if (with_semicolon)
      oss << ";";
    oss << history_[i].ToSgfString(with_semicolon);
    if (with_comments)
      oss << "C[" << comments[i] << "]";
  }
  return oss.str();
}

std::string NoGoGame::ToSgfFileString(std::string player0,
                                      std::string player1,
                                      std::string event_name,
                                      bool with_semicolon = false) {
  std::ostringstream oss;
  oss << ToSgfFilePrefix(player0, player1, event_name)
      << ToMoveString(with_semicolon) << ")";
  return oss.str();
}

std::string NoGoGame::ToSgfFileString(std::string player0,
                                      std::string player1,
                                      std::string event_name,
                                      bool with_semicolon,
                                      bool with_comments,
                                      std::vector<std::string>& comments) {
  std::ostringstream oss;
  oss << ToSgfFilePrefix(player0, player1, event_name)
      << ToMoveString(with_semicolon, with_comments, comments) << ")";
  return oss.str();
}
