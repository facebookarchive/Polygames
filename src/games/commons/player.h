/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: 林鈺錦 (Yù-Jǐn Lín)
// - Github: https://github.com/abc1236762
// - Email:  abc1236762@outlook.com

#pragma once

class Player {
 public:
  enum Index : int { none = -1, first, second };

  static constexpr Player set(int i) {
    return Player(static_cast<Index>(i));
  }

  constexpr Player()
      : _i(Index::none) {
  }

  constexpr Player(Index i)
      : _i(i) {
  }

  constexpr bool operator==(const Player& p) const {
    return _i == p._i;
  }

  constexpr bool operator!=(const Player& p) const {
    return _i != p._i;
  }

  constexpr int index() {
    return static_cast<int>(_i);
  }

 private:
  Index _i;
};
