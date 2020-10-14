/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#include "../core/state.h"
#include "commons/chessboard.h"

using namespace std;

namespace GomokuSwap2 {

typedef int Player;
class Players {
 public:
  static constexpr Player player0 = 0;
  static constexpr Player player1 = 1;
};

class Chesses {
 public:
  static constexpr Chess empty = 0;
  static constexpr Chess black = 1;
  static constexpr Chess white = 2;
};

class Move {
 public:
  int x, y;
  Chess chess;
  bool isColorChanged;
};

class Game {
 public:
  void initGame();
  void play(Move& m);
  bool isBoardFulled();
  bool isWon(Move& m);
  void findLegalMoves(Player player);
  Player chessToPlayer(Chess chess);
  Chess playerToChess(Player player);

  template <typename R> static void setupBoard(const R& re);

  static constexpr int chesses = 2;
  static constexpr int boardRadix = 15;
  static constexpr int boardSize = boardRadix * boardRadix;
  static constexpr int maxLegalMovesCnt = 2 * boardRadix * boardRadix;
  static constexpr int maxHands = maxLegalMovesCnt;
  static inline std::once_flag setupCalled;

  using Board = Chessboard<boardRadix, boardRadix>;
  Board board;
  int hands;
  bool isTurned;
  Player winner;
  Move legalMoves[maxLegalMovesCnt];
  int legalMovesCnt;
};

class Action : public ::_Action {
 public:
  Action(int x, int y, bool isColorChanged);
};

class State : public ::State, public Game {
 public:
  State(int seed);
  void Initialize() override;
  unique_ptr<mcts::State> clone_() const override;
  void ApplyAction(const ::_Action& action) override;
  void DoGoodAction() override;
  // void printCurrentBoard() const override;
  // string stateDescription() const override;
  // std::string actionDescription(const ::_Action& action) const override;
  // string actionsDescription() override;
  // int parseAction(const string& str) override;

 private:
  bool canGoNext(Move& m);
  void findActions();
  void findFeatures();

  static constexpr int timesteps = 2;
  static constexpr size_t featuresSizeX = chesses * timesteps + 1;
  static constexpr size_t featuresSizeY = boardRadix;
  static constexpr size_t featuresSizeZ = boardRadix;
  static constexpr size_t featuresSize =
      featuresSizeX * featuresSizeY * featuresSizeZ;
};

}  // namespace GomokuSwap2
