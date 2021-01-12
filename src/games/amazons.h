/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: 林鈺錦 (Yù-Jǐn Lín)
// - Github: https://github.com/abc1236762
// - Email:  abc1236762@outlook.com
// Facilitator: 邱顯棟 (Xiǎn-Dòng Qiū)
// - Github: https://github.com/YumJelly
// - Email:  yumjelly@gmail.com

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

#include "../core/state.h"
#include "commons/chessboard.h"
#include "commons/player.h"

namespace Amazons {

class ChessKind {
 public:
  static constexpr Chess empty = 0;
  static constexpr Chess whiteQueen = 1;
  static constexpr Chess blackQueen = 2;
  static constexpr Chess whiteArrow = 3;
  static constexpr Chess blackArrow = 4;
};

struct Move {
  int fromX, fromY;
  int toX, toY;
  int arrowX, arrowY;
};

class State : public ::State {
 public:
  using Board = Chessboard<10, 10>;

  State(int seed);
  void Initialize() override;
  std::unique_ptr<mcts::State> clone_() const override;
  void ApplyAction(const ::_Action& action) override;
  void DoGoodAction() override;
  void printCurrentBoard() const override;
  std::string stateDescription() const override;
  std::string actionDescription(const ::_Action& action) const override;
  std::string actionsDescription() override;
  int parseAction(const std::string& str) override;
  int humanInputAction(
      std::function<std::optional<int>(std::string)> specialAction) override;

 private:
  template <typename R> static void setupBoard(const R& re);
  static constexpr Player chessToPlayer(Chess chess);
  static constexpr Chess playerToQueenChess(Player player);
  static constexpr Chess playerToArrowChess(Player player);

  void setInitialChesses();
  void play(const Move& move);
  bool canGoNext(Player nextPlayer);
  void findLegalActions(Player player);
  inline Player turnPlayer();
  inline void setTerminatedStatus(Player loser);
  void fillFeatures();

  static constexpr int players = 2;
  static constexpr int chessKinds = 4;
  static constexpr int maxHands = Board::squares - 8;
  // Maximum count of legal moves in ideal situations: ⌈(((9*4-9)*(10*4-4)+(9*4-
  // 7)*(8*4-4)+(9*4-5)*(6*4-4)+(9*4-3)*(4*4-4)+(9*4-1)*(2*4-4))/10^2)^2*4⌉=3458
  static constexpr int maxLegalActionsCnt = 3458;
  static constexpr std::tuple<int, int> directions[8] = {
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  static constexpr std::tuple<int, int>
      initialQueenChessesPos[players][chessKinds] = {
          {{0, 6}, {3, 9}, {6, 9}, {9, 6}}, {{0, 3}, {3, 0}, {6, 0}, {9, 3}}};
  static inline std::once_flag setupCalled;

  Board board;
  std::array<std::array<std::tuple<int, int>, 4>, 2> queenChessesPos;
};

class Action : public ::_Action {
 public:
  Action(int i, int fromXY, int toXY, int arrowRelRD);
};

}  // namespace Amazons
