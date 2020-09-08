/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "game_action.h"
#include "game_player.h"
#include "nogo_position.h"
#include <sstream>
#include <string>
#include <vector>

class NoGoAction : public GameAction {
 private:
  PLAYER player_;
  Position position_;

 public:
  NoGoAction(PLAYER player = PLAYER_NULL, Position position = kNOGO_GRIDS_NUM);
  PLAYER GetPlayer() const {
    return player_;
  }
  Position GetPosition() const;
  void SetPlayer(PLAYER player) {
    player_ = player;
  }
  bool IsIllegalAction() const {
    return player_ == PLAYER_NULL;
  }
  void Set(PLAYER player, Position position);
  void SetPosition(Position position);
  bool operator==(const NoGoAction& rhs);
  bool operator!=(const NoGoAction& rhs);
  int x() const;
  int y() const;
  std::string ToString();
  std::string ToGTPString(bool with_color = false) const;
  std::string ToSgfString(bool with_color = false) const;
  void Rotate(SYMMETRYTYPE type);
  int GetID() {
    return position_;
  }
  void SetID(int id) {
    position_ = id;
  }
};
#include "nogo_action.cc"
