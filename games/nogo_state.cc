/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "nogo_state.h"
#include <iostream>

NoGoState::NoGoState() {
  InitNeighborList();
  Reset();
}

void NoGoState::Reset() {
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    parent_[i] = i;
    liberty_[i].Reset();
  }

  bm_board_[PLAYER_0].Reset();
  illegal_[PLAYER_0].Reset();
  warning_[PLAYER_0].Reset();

  bm_board_[PLAYER_1].Reset();
  illegal_[PLAYER_1].Reset();
  warning_[PLAYER_1].Reset();
  liberty_is_one_.Reset();
}

bool NoGoState::PlayAction(NoGoAction action) {
  PLAYER player = action.GetPlayer();
  Position position = action.GetPosition();
  if (!IsLegalAction(player, position)) {
    return false;
  }
  illegal_[PLAYER_0].AddPosition(position);
  illegal_[PLAYER_1].AddPosition(position);
  bm_board_[player].AddPosition(position);

  Position parent_position = position;
  NoGoBitBoard parent_new_liberty;

  for (size_t i = 0; i < neighbor_list_[position].size(); i++) {
    Position neighbor = neighbor_list_[position][i];

    if (bm_board_[player].GetPosition(neighbor)) {
      Position parent_of_neighbor = FindParent(neighbor);
      parent_new_liberty |= liberty_[parent_of_neighbor];
      if (parent_position < parent_of_neighbor)  // unite
      {
        parent_[parent_of_neighbor] = parent_position;
      } else {
        parent_[parent_position] = parent_of_neighbor;
        parent_position = parent_of_neighbor;
      }

    } else if (bm_board_[!player].GetPosition(neighbor)) {
      Position parent_of_neighbor = FindParent(neighbor);
      liberty_[parent_of_neighbor].DeletePosition(position);
      if (liberty_[parent_of_neighbor].CheckIsOne()) {
        liberty_is_one_.AddPosition(parent_of_neighbor);
        illegal_[player] |= liberty_[parent_of_neighbor];
        warning_[!player] |= liberty_[parent_of_neighbor];
      }
    } else {
      warning_[!player].AddPosition(neighbor);
      parent_new_liberty.AddPosition(neighbor);
    }
  }
  parent_new_liberty.DeletePosition(position);
  liberty_[parent_position] = parent_new_liberty;

  if (parent_new_liberty.CheckIsOne()) {
    liberty_is_one_.AddPosition(parent_position);
    illegal_[!player] |= parent_new_liberty;
    warning_[player] |= parent_new_liberty;
  } else {
    liberty_is_one_.DeletePosition(parent_position);
  }
  return true;
}

NoGoState& NoGoState::operator=(const NoGoState& rhs) {
  for (int i = 0; i < 2; i++)  // two players
  {
    bm_board_[i] = rhs.bm_board_[i];
    illegal_[i] = rhs.illegal_[i];
    warning_[i] = rhs.warning_[i];
  }
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    parent_[i] = rhs.parent_[i];
    liberty_[i] = rhs.liberty_[i];
  }
  liberty_is_one_ = rhs.liberty_is_one_;
  return *this;
}

void NoGoState::Rotate(SYMMETRYTYPE type) {
  std::vector<NoGoAction> action_list;
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    PLAYER player = GetPlayer(i);
    if (player == PLAYER_NULL)
      continue;
    Point point((Position)i);
    point.ToSymmetryOf(type);
    action_list.push_back(NoGoAction(player, point.GetPosition()));
  }
  Reset();
  for (size_t i = 0; i < action_list.size(); i++) {
    PlayAction(action_list[i]);
  }
}

PLAYER NoGoState::GetPlayer(Position position) const {
  if (bm_board_[PLAYER_0].GetPosition(position))
    return PLAYER_0;
  if (bm_board_[PLAYER_1].GetPosition(position))
    return PLAYER_1;
  return PLAYER_NULL;
}

bool NoGoState::IsLegalAction(PLAYER player, Position position) {
  if (player == PLAYER_NULL)
    return false;
  if (illegal_[player].GetPosition(position))
    return false;
  else if (!warning_[player].GetPosition(position))
    return true;
  warning_[player].DeletePosition(position);

  // start check warning (is the action a suicide action)
  for (size_t i = 0; i < neighbor_list_[position].size(); i++) {
    Position neighbor = neighbor_list_[position][i];
    if (bm_board_[player].GetPosition(neighbor)) {
      if (!liberty_is_one_.GetPosition(FindParent(neighbor)))
        return true;
    } else if (!bm_board_[!player].GetPosition(neighbor))  // have liberty
    {
      return true;
    }
  }
  illegal_[player].AddPosition(position);
  return false;
}

bool NoGoState::IsLegalAction(NoGoAction action) {
  return IsLegalAction(action.GetPlayer(), action.GetPosition());
}

void NoGoState::ShowBoard() const {
  std::cerr << ToString() << std::endl;
}

void NoGoState::ShowLegalMove(PLAYER turn_player) {
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    PLAYER player = GetPlayer(i);
    if (player == PLAYER_0)
      std::cerr << '@';
    if (player == PLAYER_1)
      std::cerr << 'O';
    if (player == PLAYER_NULL) {
      if (IsLegalAction(turn_player, i)) {
        std::cerr << "#";
      } else {
        std::cerr << ".";
      }
    }
    if (i % kNOGO_BOARD_SIZE == kNOGO_BOARD_SIZE - 1)
      std::cerr << '\n';
  }
}

std::string NoGoState::ToString() const {
  std::ostringstream oss;
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    PLAYER player = GetPlayer(i);
    if (player == PLAYER_0)
      oss << '@';
    if (player == PLAYER_1)
      oss << 'O';
    if (player == PLAYER_NULL)
      oss << '.';
    if (i % kNOGO_BOARD_SIZE == kNOGO_BOARD_SIZE - 1)
      oss << '\n';
  }
  return oss.str();
}

void NoGoState::PrintNeighborNum() const {
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    std::cerr << neighbor_list_[i].size() << ' ';
    if (i % kNOGO_BOARD_SIZE == kNOGO_BOARD_SIZE - 1)
      std::cerr << '\n';
  }
}

void NoGoState::PrintLiberty() {
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (!bm_board_[0].GetPosition(i) && !bm_board_[1].GetPosition(i)) {
      std::cerr << ".\t";
    } else if (FindParent(i) == i) {
      std::cerr << liberty_[i].Count() << '\t';
    } else {
      std::cerr << "0\t";
    }
    if (i % kNOGO_BOARD_SIZE == kNOGO_BOARD_SIZE - 1)
      std::cerr << '\n';
  }
}

void NoGoState::PrintLibertyIsOne(bool check_again) {
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (!bm_board_[0].GetPosition(i) && !bm_board_[1].GetPosition(i)) {
      std::cerr << ".\t";
    } else if (FindParent(i) == i) {
      if (check_again) {
        std::cerr << liberty_[i].CheckIsOne() << '\t';
        if (liberty_[i].CheckIsOne())
          liberty_is_one_.AddPosition(i);
        else
          liberty_is_one_.DeletePosition(i);
      } else {
        std::cerr << liberty_is_one_.GetPosition(i) << '\t';
      }
    } else {
      std::cerr << "0\t";
    }
    if (i % kNOGO_BOARD_SIZE == kNOGO_BOARD_SIZE - 1)
      std::cerr << '\n';
  }
}

void NoGoState::PrintParent() {
  for (int i = 0; i < kNOGO_GRIDS_NUM; i++) {
    if (!bm_board_[0].GetPosition(i) && !bm_board_[1].GetPosition(i)) {
      std::cerr << ".\t";
    } else {
      std::cerr << FindParent(i) << '\t';
    }
    if (i % kNOGO_BOARD_SIZE == kNOGO_BOARD_SIZE - 1)
      std::cerr << '\n';
  }
}

Position NoGoState::FindParent(Position position) {
  Position& parent_position = parent_[position];
  if (parent_position == parent_[parent_position])
    return parent_position;
  return parent_position = FindParent(parent_position);
}

void NoGoState::InitNeighborList() {
  for (int i = 0; i < kNOGO_BOARD_SIZE; i++) {
    for (int j = 0; j < kNOGO_BOARD_SIZE; j++) {
      int position = i * kNOGO_BOARD_SIZE + j;
      if (i > 0)
        neighbor_list_[position].push_back(position - kNOGO_BOARD_SIZE);
      if (j > 0)
        neighbor_list_[position].push_back(position - 1);
      if (i < kNOGO_BOARD_SIZE - 1)
        neighbor_list_[position].push_back(position + kNOGO_BOARD_SIZE);
      if (j < kNOGO_BOARD_SIZE - 1)
        neighbor_list_[position].push_back(position + 1);
    }
  }
}
