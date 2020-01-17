/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "game_state.h"
#include "nogo_action.h"
#include "nogo_bitboard.h"
#include <vector>

class NoGoState : public GameState {
 private:
  NoGoBitBoard bm_board_[2];
  std::vector<Position> neighbor_list_[kNOGO_GRIDS_NUM];

  Position parent_[kNOGO_GRIDS_NUM];
  NoGoBitBoard liberty_[kNOGO_GRIDS_NUM];
  NoGoBitBoard illegal_[2];
  NoGoBitBoard warning_[2];  // might be suicide
  NoGoBitBoard liberty_is_one_;

 public:
  NoGoState();
  void Reset();
  bool PlayAction(NoGoAction action);  // return success or not
  NoGoState& operator=(const NoGoState& rhs);
  void Rotate(SYMMETRYTYPE type);

  PLAYER GetPlayer(Position position) const;
  bool IsLegalAction(NoGoAction action);
  bool IsLegalAction(PLAYER player, Position position);
  void ShowBoard() const;
  void ShowLegalMove(PLAYER turn_player);
  std::string ToString() const;
  void PrintNeighborNum() const;
  void PrintLiberty();
  void PrintLibertyIsOne(bool check_again = false);
  void PrintParent();

 private:
  Position FindParent(Position p);
  void InitNeighborList();
};
#include "nogo_state.cc"
