/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CZF_NOGO_NOGO_BITBOARD_H_
#define CZF_NOGO_NOGO_BITBOARD_H_

#include "nogo_position.h"
class NoGoBitBoard {
  static const long long kMASK_55 = 0x5555555555555555ULL;
  static const long long kMASK_33 = 0x3333333333333333ULL;
  static const long long kMASK_0F = 0x0f0f0f0f0f0f0f0fULL;
  static const long long kMASK_01 = 0x0101010101010101ULL;
  static const long long kMASK_00FF = 0x00ff00ff00ff00ffULL;
  static const long long kMASK_0000FFFF = 0x0000ffff0000ffffULL;
  static const long long kMASK_00000000FFFFFFFF = 0x00000000ffffffffULL;
  static const long long kMASK_FFFFFFFFFFFFFFFF = 0xffffffffffffffffULL;
  long long bitboard_[(kNOGO_GRIDS_NUM / 64) + 1];

 public:
  NoGoBitBoard() {
    Reset();
  }
  void Reset() {
    bitboard_[0] = 0;
    bitboard_[1] = 0;
  }

  NoGoBitBoard& operator=(const NoGoBitBoard& rhs) {
    bitboard_[0] = rhs.bitboard_[0];
    bitboard_[1] = rhs.bitboard_[1];
    return *this;
  }

  int Count() const {
    unsigned long long v, v1;
    v = (bitboard_[0] & kMASK_55) + ((bitboard_[0] >> 1) & kMASK_55);
    v1 = (bitboard_[1] & kMASK_55) + ((bitboard_[1] >> 1) & kMASK_55);
    v = (v & kMASK_33) + ((v >> 2) & kMASK_33);
    v1 = (v1 & kMASK_33) + ((v1 >> 2) & kMASK_33);
    v += v1;
    v = (v & kMASK_0F) + ((v >> 4) & kMASK_0F);
    v = (v & kMASK_00FF) + ((v >> 8) & kMASK_00FF);
    v = (v & kMASK_0000FFFF) + ((v >> 16) & kMASK_0000FFFF);
    return (int)((v & kMASK_00000000FFFFFFFF) + (v >> 32));
  }

  bool GetPosition(int i) const {
    return (bitboard_[i >> 6] & (1LL << (i & 63))) != 0;
  }

  void DeletePosition(int i) {
    bitboard_[i >> 6] &= ~(1LL << (i & 63));
  }

  void AddPosition(int i) {
    bitboard_[i >> 6] |= (1LL << (i & 63));
  }

  void operator|=(NoGoBitBoard rhs) {
    bitboard_[0] |= rhs.bitboard_[0];
    bitboard_[1] |= rhs.bitboard_[1];
    return;
  }

  bool Isempty() const {
    return (bitboard_[0] == 0) && (bitboard_[1] == 0);
  }

  bool CheckIsOne() const {
    if (bitboard_[0] == 0) {
      return (bitboard_[1] != 0) &&
             ((bitboard_[1] ^ ((-bitboard_[1]) & (bitboard_[1]))) == 0);
    } else if (bitboard_[1] == 0) {
      return (bitboard_[0] ^ ((-bitboard_[0]) & (bitboard_[0]))) == 0;
    }
    return false;
  }
};

#endif  // CZF_NOGO_NOGO_BITBOARD_H_
