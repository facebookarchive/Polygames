/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "nogo_action.h"

NoGoAction::NoGoAction(PLAYER player, Position position) {
  Set(player, position);
}

void NoGoAction::Set(PLAYER player, Position position) {
  SetPlayer(player);
  SetPosition(position);
}

void NoGoAction::SetPosition(Position position) {
  position_ = position;
}

Position NoGoAction::GetPosition() const {
  return position_;
}

bool NoGoAction::operator==(const NoGoAction& rhs) {
  return ((GetPlayer() == rhs.GetPlayer()) && (position_ == rhs.position_));
}

bool NoGoAction::operator!=(const NoGoAction& rhs) {
  return !(*this == rhs);
}

int NoGoAction::x() const {
  return position_ / kNOGO_BOARD_SIZE;
}

int NoGoAction::y() const {
  return position_ % kNOGO_BOARD_SIZE;
}

std::string NoGoAction::ToString() {
  return ToGTPString(true);
}

std::string NoGoAction::ToGTPString(bool with_color) const {
  std::ostringstream oss;
  if (with_color) {
    if (GetPlayer() == PLAYER_0) {
      oss << "B(";
    } else {
      oss << "W(";
    }
  }
  oss << (char)(y() + 'A' + (y() >= 8))
      << (char)(kNOGO_BOARD_SIZE - x() - 1 + '1');
  if (with_color) {
    oss << ")";
  }

  return oss.str();
}

std::string NoGoAction::ToSgfString(bool with_color) const {
  std::ostringstream oss;
  if (with_color) {
    if (GetPlayer() == PLAYER_0) {
      oss << "B[";
    } else {
      oss << "W[";
    }
  }
  if (GetPlayer() == PLAYER_NULL || GetPosition() == kNOGO_GRIDS_NUM) {
    oss << "tt";
  } else {
    oss << (char)(y() + 'a') << (char)(x() + 'a');
  }
  if (with_color) {
    oss << "]";
  }
  return oss.str();
}

void NoGoAction::Rotate(SYMMETRYTYPE type) {
  Point point(position_);
  point.ToSymmetryOf(type);
  position_ = point.GetPosition();
}
