/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: 葉士誠 (SHI-CHENG YE)
// affiliation: National Dong Hwa University(NDHU)
// email: 410521206@gms.ndhu.edu.tw / 0930164@gmail.com

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>

#include "../core/state.h"
#include "chinesecheckers_defines.h"
#include "commons/chessboard.h"

using namespace std;

namespace ChineseCheckers {
typedef int Player;
class Players {
 public:
  static constexpr Player player0 = 0;
  static constexpr Player player1 = 1;
};

class Chesses {  // define Chess in commons/bitboard.h
 public:
  static constexpr Chess empty = 0;
  static constexpr Chess White = 1;
  static constexpr Chess Black = 2;
};

class Board : public ::Chessboard<11, 11> {
 public:
  string sprintBoard(string_view prefix = "",
                     const set<tuple<int, int>>& markedPos = {}) const override;
  // string_view getPosStr(int x, int y) const override;
  string getPosStr(int p) const override;

  string showBoard(bool prePlay) const;

  vector<int> legal;
};

struct Move {
  // int x, y, tx, ty;
  int x, tx, method;
  Chess chess;
};

class Game {
 public:
  void initialize();
  void setInitialChesses();
  void play(Move& m);
  bool Won();
  void findLegalMoves(Player player);
  // void LegalMove(int x, int y, Chess chess, bool isJump);
  void LegalMove(int cur, Chess chess, bool isJump);
  int canGo(int cur, int md, Chess chess);

  template <typename R> static void setupBoard(const R& re);
  static constexpr Player chessToPlayer(Chess chess);
  static constexpr Chess playerToChess(Player player);

  static constexpr size_t players = 2;
  static constexpr size_t chesses = 2;
  static constexpr int boardWH = 11;             // width and height in board
  static constexpr int maxLegalMovesCnt = 5000;  // boardWH * boardWH - 40;
  static constexpr int maxHands =
      maxLegalMovesCnt;  // maxLegalMovesCnt * 2 * 10;
  static inline once_flag setupCalled;

  Board board;
  int hands, prePos, preChess;
  bool continue_jump;  // for action description(enable to continue jump )
  Player winner;
  vector<Move> legalMoves, way;

  Move Pass = {pass, pass, Chesses::empty, true};

  static constexpr short int move_table[121][6] = {
      //  position {LU,RU,L,R,LD,RD}
      {-1, -1, -1, -1, P_A2, P_A3},                // A1
      {-1, P_A1, -1, P_A3, P_A4, P_A5},            // A2
      {P_A1, -1, P_A2, -1, P_A5, P_A6},            // A3
      {-1, P_A2, -1, P_A5, P_A7, P_A8},            // A4
      {P_A2, P_A3, P_A4, P_A6, P_A8, P_A9},        // A5
      {P_A3, -1, P_A5, -1, P_A9, P_A10},           // A6
      {-1, P_A4, -1, P_A8, P_G1, P_G2},            // A7
      {P_A4, P_A5, P_A7, P_A9, P_G2, P_G3},        // A8
      {P_A5, P_A6, P_A8, P_A10, P_G3, P_G4},       // A9
      {P_A6, -1, P_A9, -1, P_G4, P_G5},            // A10
      {-1, -1, P_B2, -1, P_B3, -1},                // B1
      {-1, -1, P_B4, P_B1, P_B5, P_B3},            // B2
      {P_B2, P_B1, P_B5, -1, P_B6, -1},            // B3
      {-1, -1, P_B7, P_B2, P_B8, P_B5},            // B4
      {P_B4, P_B2, P_B8, P_B3, P_B9, P_B6},        // B5
      {P_B5, P_B3, P_B9, -1, P_B10, -1},           // B6
      {-1, -1, P_G5, P_B4, P_G11, P_B8},           // B7
      {P_B7, P_B4, P_G11, P_B5, P_G18, P_B9},      // B8
      {P_B8, P_B5, P_G18, P_B6, P_G26, P_B10},     // B9
      {P_B9, P_B6, P_G26, -1, P_G35, -1},          // B10
      {P_C2, -1, P_C3, -1, -1, -1},                // C1
      {P_C4, -1, P_C5, -1, P_C3, P_C1},            // C2
      {P_C5, P_C2, P_C6, P_C1, -1, -1},            // C3
      {P_C7, -1, P_C8, -1, P_C5, P_C2},            // C4
      {P_C8, P_C4, P_C9, P_C2, P_C6, P_C3},        // C5
      {P_C9, P_C5, P_C10, P_C3, -1, -1},           // C6
      {P_G35, -1, P_G43, -1, P_C8, P_C4},          // C7
      {P_G43, P_C7, P_G50, P_C4, P_C9, P_C5},      // C8
      {P_G50, P_C8, P_G56, P_C5, P_C10, P_C6},     // C9
      {P_G56, P_C9, P_G61, P_C6, -1, -1},          // C10
      {P_D3, P_D2, -1, -1, -1, -1},                // D1
      {P_D5, P_D4, P_D3, -1, P_D1, -1},            // D2
      {P_D6, P_D5, -1, P_D2, -1, P_D1},            // D3
      {P_D8, P_D7, P_D5, -1, P_D2, -1},            // D4
      {P_D9, P_D8, P_D6, P_D4, P_D3, P_D2},        // D5
      {P_D10, P_D9, -1, P_D5, -1, P_D3},           // D6
      {P_G60, P_G61, P_D8, -1, P_D4, -1},          // D7
      {P_G59, P_G60, P_D9, P_D7, P_D5, P_D4},      // D8
      {P_G58, P_G59, P_D10, P_D8, P_D6, P_D5},     // D9
      {P_G57, P_G58, -1, P_D9, -1, P_D6},          // D10
      {-1, P_E3, -1, P_E2, -1, -1},                // E1
      {P_E3, P_E5, P_E1, P_E4, -1, -1},            // E2
      {-1, P_E6, -1, P_E5, P_E1, P_E2},            // E3
      {P_E5, P_E8, P_E2, P_E7, -1, -1},            // E4
      {P_E6, P_E9, P_E3, P_E8, P_E2, P_E4},        // E5
      {-1, P_E10, -1, P_E9, P_E3, P_E5},           // E6
      {P_E8, P_G51, P_E4, P_G57, -1, -1},          // E7
      {P_E9, P_G44, P_E5, P_G51, P_E4, P_E7},      // E8
      {P_E10, P_G36, P_E6, P_G44, P_E5, P_E8},     // E9
      {-1, P_G27, -1, P_G36, P_E6, P_E9},          // E10
      {-1, -1, -1, P_F3, -1, P_F2},                // F1
      {P_F1, P_F3, -1, P_F5, -1, P_F4},            // F2
      {-1, -1, P_F1, P_F6, P_F2, P_F5},            // F3
      {P_F2, P_F5, -1, P_F8, -1, P_F7},            // F4
      {P_F3, P_F6, P_F2, P_F9, P_F4, P_F8},        // F5
      {-1, -1, P_F3, P_F10, P_F5, P_F9},           // F6
      {P_F4, P_F8, -1, P_G19, -1, P_G27},          // F7
      {P_F5, P_F9, P_F4, P_G12, P_F7, P_G19},      // F8
      {P_F6, P_F10, P_F5, P_G6, P_F8, P_G12},      // F9
      {-1, -1, P_F6, P_G1, P_F9, P_G6},            // F10
      {-1, P_A7, P_F10, P_G2, P_G6, P_G7},         // G1
      {P_A7, P_A8, P_G1, P_G3, P_G7, P_G8},        // G2
      {P_A8, P_A9, P_G2, P_G4, P_G8, P_G9},        // G3
      {P_A9, P_A10, P_G3, P_G5, P_G9, P_G10},      // G4
      {P_A10, -1, P_G4, P_B7, P_G10, P_G11},       // G5
      {P_F10, P_G1, P_F9, P_G7, P_G12, P_G13},     // G6
      {P_G1, P_G2, P_G6, P_G8, P_G13, P_G14},      // G7
      {P_G2, P_G3, P_G7, P_G9, P_G14, P_G15},      // G8
      {P_G3, P_G4, P_G8, P_G10, P_G15, P_G16},     // G9
      {P_G4, P_G5, P_G9, P_G11, P_G16, P_G17},     // G10
      {P_G5, P_B7, P_G10, P_B8, P_G17, P_G18},     // G11
      {P_F9, P_G6, P_F8, P_G13, P_G19, P_G20},     // G12
      {P_G6, P_G7, P_G12, P_G14, P_G20, P_G21},    // G13
      {P_G7, P_G8, P_G13, P_G15, P_G21, P_G22},    // G14
      {P_G8, P_G9, P_G14, P_G16, P_G22, P_G23},    // G15
      {P_G9, P_G10, P_G15, P_G17, P_G23, P_G24},   // G16
      {P_G10, P_G11, P_G16, P_G18, P_G24, P_G25},  // G17
      {P_G11, P_B8, P_G17, P_B9, P_G25, P_G26},    // G18
      {P_F8, P_G12, P_F7, P_G20, P_G27, P_G28},    // G19
      {P_G12, P_G13, P_G19, P_G21, P_G28, P_G29},  // G20
      {P_G13, P_G14, P_G20, P_G22, P_G29, P_G30},  // G21
      {P_G14, P_G15, P_G21, P_G23, P_G30, P_G31},  // G22
      {P_G15, P_G16, P_G22, P_G24, P_G31, P_G32},  // G23
      {P_G16, P_G17, P_G23, P_G25, P_G32, P_G33},  // G24
      {P_G17, P_G18, P_G24, P_G26, P_G33, P_G34},  // G25
      {P_G18, P_B9, P_G25, P_B10, P_G34, P_G35},   // G26
      {P_F7, P_G19, -1, P_G28, P_E10, P_G36},      // G27
      {P_G19, P_G20, P_G27, P_G29, P_G36, P_G37},  // G28
      {P_G20, P_G21, P_G28, P_G30, P_G37, P_G38},  // G29
      {P_G21, P_G22, P_G29, P_G31, P_G38, P_G39},  // G30
      {P_G22, P_G23, P_G30, P_G32, P_G39, P_G40},  // G31
      {P_G23, P_G24, P_G31, P_G33, P_G40, P_G41},  // G32
      {P_G24, P_G25, P_G32, P_G34, P_G41, P_G42},  // G33
      {P_G25, P_G26, P_G33, P_G35, P_G42, P_G43},  // G34
      {P_G26, P_B10, P_G34, -1, P_G43, P_C7},      // G35
      {P_G27, P_G28, P_E10, P_G37, P_E9, P_G44},   // G36
      {P_G28, P_G29, P_G36, P_G38, P_G44, P_G45},  // G37
      {P_G29, P_G30, P_G37, P_G39, P_G45, P_G46},  // G38
      {P_G30, P_G31, P_G38, P_G40, P_G46, P_G47},  // G39
      {P_G31, P_G32, P_G39, P_G41, P_G47, P_G48},  // G40
      {P_G32, P_G33, P_G40, P_G42, P_G48, P_G49},  // G41
      {P_G33, P_G34, P_G41, P_G43, P_G49, P_G50},  // G42
      {P_G34, P_G35, P_G42, P_C7, P_G50, P_C8},    // G43
      {P_G36, P_G37, P_E9, P_G45, P_E8, P_G51},    // G44
      {P_G37, P_G38, P_G44, P_G46, P_G51, P_G52},  // G45
      {P_G38, P_G39, P_G45, P_G47, P_G52, P_G53},  // G46
      {P_G39, P_G40, P_G46, P_G48, P_G53, P_G54},  // G47
      {P_G40, P_G41, P_G47, P_G49, P_G54, P_G55},  // G48
      {P_G41, P_G42, P_G48, P_G50, P_G55, P_G56},  // G49
      {P_G42, P_G43, P_G49, P_C8, P_G56, P_C9},    // G50
      {P_G44, P_G45, P_E8, P_G52, P_E7, P_G57},    // G51
      {P_G45, P_G46, P_G51, P_G53, P_G57, P_G58},  // G52
      {P_G46, P_G47, P_G52, P_G54, P_G58, P_G59},  // G53
      {P_G47, P_G48, P_G53, P_G55, P_G59, P_G60},  // G54
      {P_G48, P_G49, P_G54, P_G56, P_G60, P_G61},  // G55
      {P_G49, P_G50, P_G55, P_C9, P_G61, P_C10},   // G56
      {P_G51, P_G52, P_E7, P_G58, -1, P_D10},      // G57
      {P_G52, P_G53, P_G57, P_G59, P_D10, P_D9},   // G58
      {P_G53, P_G54, P_G58, P_G60, P_D9, P_D8},    // G59
      {P_G54, P_G55, P_G59, P_G61, P_D8, P_D7},    // G60
      {P_G55, P_G56, P_G60, P_C10, P_D7, -1}       // G61
  };

 private:
  int JumpOBS(int cur, int dirct);
};

class State : public core::State, public Game {
 public:
  State(int seed);
  void Initialize() override;
  unique_ptr<core::State> clone_() const override;
  void ApplyAction(const _Action& action) override;
  void DoGoodAction() override;
  void printCurrentBoard() const override;
  string stateDescription() const override;
  string actionsDescription() const override;
  string actionDescription(const ::_Action& action) const override;
  int parseAction(const string& str) const override;
  int humanInputAction(
      std::function<std::optional<int>(std::string)> specialAction) override;

 private:
  Player changeTurn(Player player);
  bool canChange(Player Player);
  void findActions();
  void findFeatures();

  int seed;

  static constexpr size_t featuresSizeX = chesses;
  static constexpr size_t featuresSizeY = boardWH;
  static constexpr size_t featuresSizeZ = boardWH;
  static constexpr size_t featuresSize =
      featuresSizeX * featuresSizeY * featuresSizeZ;
};

}  // namespace ChineseCheckers
