/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CZF_NOGO_NOGO_POSITION_H_
#define CZF_NOGO_NOGO_POSITION_H_

#include <cassert>
#include <string>
#include <vector>

const int kNOGO_BOARD_SIZE = 9;
const int kNOGO_GRIDS_NUM = kNOGO_BOARD_SIZE * kNOGO_BOARD_SIZE;
typedef int Position;

// just for represent X and Y and do some symmetric
enum SYMMETRYTYPE {
  SYM_NORMAL,
  SYM_ROTATE_90,
  SYM_ROTATE_180,
  SYM_ROTATE_270,
  SYM_HORIZONTAL_REFLECTION,
  SYM_HORIZONTAL_REFLECTION_ROTATE_90,
  SYM_HORIZONTAL_REFLECTION_ROTATE_180,
  SYM_HORIZONTAL_REFLECTION_ROTATE_270,
  SYMMETRY_SIZE
};

namespace symmetry {
const std::vector<SYMMETRYTYPE> kSYMMETRY_LIST{
    SYM_NORMAL,
    SYM_ROTATE_90,
    SYM_ROTATE_180,
    SYM_ROTATE_270,
    SYM_HORIZONTAL_REFLECTION,
    SYM_HORIZONTAL_REFLECTION_ROTATE_90,
    SYM_HORIZONTAL_REFLECTION_ROTATE_180,
    SYM_HORIZONTAL_REFLECTION_ROTATE_270};

const std::string kSYMMETRY_TYPE_STRING[SYMMETRY_SIZE] = {
    "SYM_NORMAL",
    "SYM_ROTATE_90",
    "SYM_ROTATE_180",
    "SYM_ROTATE_270",
    "SYM_HORIZONTAL_REFLECTION",
    "SYM_HORIZONTAL_REFLECTION_ROTATE_90",
    "SYM_HORIZONTAL_REFLECTION_ROTATE_180",
    "SYM_HORIZONTAL_REFLECTION_ROTATE_270"};

inline std::string GetSymmetryTypeString(SYMMETRYTYPE type) {
  return kSYMMETRY_TYPE_STRING[type];
}

inline SYMMETRYTYPE GetSymmetryType(std::string sType) {
  for (int i = 0; i < SYMMETRY_SIZE; i++) {
    if (sType == kSYMMETRY_TYPE_STRING[i]) {
      return static_cast<SYMMETRYTYPE>(i);
    }
  }
  return SYMMETRY_SIZE;
}

const SYMMETRYTYPE kREVERSE_SYMMETRIC_TYPE[SYMMETRY_SIZE] = {
    SYM_NORMAL,
    SYM_ROTATE_270,
    SYM_ROTATE_180,
    SYM_ROTATE_90,
    SYM_HORIZONTAL_REFLECTION,
    SYM_HORIZONTAL_REFLECTION_ROTATE_90,
    SYM_HORIZONTAL_REFLECTION_ROTATE_180,
    SYM_HORIZONTAL_REFLECTION_ROTATE_270};
}  // namespace symmetry

class Point {
 public:
  int x_;
  int y_;
  Point() {
    ;
  }
  Point(int x, int y) {
    x_ = x;
    y_ = y;
  }
  Point(const Position p) {
    x_ = p % kNOGO_BOARD_SIZE;
    y_ = p / kNOGO_BOARD_SIZE;
  }
  inline bool operator==(const Point& rhs) const {
    return (x_ == rhs.x_ && y_ == rhs.y_) ? true : false;
  }
  inline bool operator!=(const Point& rhs) const {
    return !(*this == rhs);
  }
  Point& operator=(const Point& rhs) {
    x_ = rhs.x_;
    y_ = rhs.y_;
    return *this;
  }
  Point& operator=(const Position& rhs) {
    *this = Point(rhs);
    return *this;
  }
  inline Position GetPosition() {
    return y_ * kNOGO_BOARD_SIZE + x_;
  }
  inline void ToSymmetryOf(SYMMETRYTYPE type) {
    /*
    symmetric radius pattern:           ( ChangeXY   x*(-1)
    y*(-1) ) 0 NORMAL                 : 1
    ROTATE_90 : ChangeXY             y*(-1) 2 ROTATE_180 :            x*(-1)
    y*(-1) 3 ROTATE_270 : ChangeXY   x*(-1) 4 HORIZONTAL_REFLECTION
    : x*(-1) 5 HORIZONTAL_REFLECTION_ROTATE_90	: ChangeXY 6
    HORIZONTAL_REFLECTION_ROTATE_180	:                     y*(-1) 7
    HORIZONTAL_REFLECTION_ROTATE_270	: ChangeXY   x*(-1)   y*(-1)
    */
    Shift();
    switch (type) {
    case SYM_NORMAL:
      break;
    case SYM_ROTATE_90:
      ChangeXY();
      MinusY();
      break;
    case SYM_ROTATE_180:
      MinusX();
      MinusY();
      break;
    case SYM_ROTATE_270:
      ChangeXY();
      MinusX();
      break;
    case SYM_HORIZONTAL_REFLECTION:
      MinusX();
      break;
    case SYM_HORIZONTAL_REFLECTION_ROTATE_90:
      ChangeXY();
      break;
    case SYM_HORIZONTAL_REFLECTION_ROTATE_180:
      MinusY();
      break;
    case SYM_HORIZONTAL_REFLECTION_ROTATE_270:
      ChangeXY();
      MinusX();
      MinusY();
      break;
    default:
      // should not be here
      assert(false);
    }
    ShiftBack();
    return;
  }

 private:
  inline void MinusX() {
    x_ = -x_;
  }
  inline void MinusY() {
    y_ = -y_;
  }
  inline void ChangeXY() {
    int tmp = x_;
    x_ = y_;
    y_ = tmp;
  }
  inline void Shift() {
    int center = kNOGO_BOARD_SIZE / 2;
    x_ -= center;
    y_ -= center;
  }
  inline void ShiftBack() {
    int center = kNOGO_BOARD_SIZE / 2;
    x_ += center;
    y_ += center;
  }
};

#endif  // CZF_NOGO_NOGO_POSITION_H_
