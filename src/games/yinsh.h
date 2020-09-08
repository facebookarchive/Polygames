/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "../core/state.h"
// declare helper classes and functions
#define NUM_ACTIONS 59
// (6 directions * 9 max jump) + 1(initial placement) + 1(ring selection for
// removal) + 1(seq selection via choosing a marker)
#define BOARD_X 11
#define BOARD_Y 11
#define NUM_PIECES 5

using namespace std;
// PE the coordinates returned by the output of the net are actually shifted

// namespace Yinsh{
enum class piece : int {
  invalid,
  empty,
  p0_marker,
  p0_ring,
  p1_marker,
  p1_ring
};

class ActionForYinsh : public _Action {
 public:
  ActionForYinsh(int action_num, int x, int y, int index)
      : _Action() {
    assert(action_num >= 0 && action_num < NUM_ACTIONS);
    assert(x >= 0 && x < BOARD_X);
    assert(y >= 0 && y < BOARD_Y);
    _loc[0] = action_num;
    _loc[1] = x;
    _loc[2] = y;
    _hash = (uint64_t)(action_num * (BOARD_X * BOARD_Y) + y * BOARD_X + x);
    _i = (int)index;  // DNU
  }
};

class StateForYinsh : public core::State {
 public:
  StateForYinsh(int seed)
      : State(seed) {
  }
  void Initialize() override;
  // virtual unique_ptr<core::State> clone_() const override;
  unique_ptr<core::State> clone_(void) const override;
  void ApplyAction(const _Action& action) override;
  void DoGoodAction(void) override;
  void printCurrentBoard(void) const override;
  bool ended();
  string stateDescription(void) const override;  // DNU
  string actionsDescription(void) const override;      // DNU
  static void fill_hash_table();
  static uint64_t hash_table[5][BOARD_X][BOARD_Y];
  static std::once_flag table_flag;

 private:
  void findActions(void);  // i have to maintain legal actions by myself
  void findFeatures(
      void);  // after every move i have to call these two because the
              // base(State class expects that it would be filled)
  char map_piece_to_char(int p) const;
  // void printCurrentBoard();
  tuple<int, int> map_num_to_direction(int n);
  int map_direction_to_num(int i, int j);
  void set_vars();
  tuple<int, int> find_first_invalid(int x, int y, int d0, int d1);
  vector<int> find_first_5_for_specific_pt(int x, int y);
  vector<vector<int>> find_all_5s(bool my);

  int places_filled;
  int initial_fill;
  bool still_have_to_remove_ring;
  bool still_have_to_remove_marker;
  int board[13][13];
  // vector <tuple<int,int>> *my_rings;//not necessary but makes it faster
  // vector <tuple<int,int>> *opp_rings;
  // vector <tuple<int,int>> p0_rings;
  // vector <tuple<int,int>> p1_rings;
  vector<vector<tuple<int, int>>> rings;
  int player, my_ring, my_marker, opp_ring,
      opp_marker;  // set all these with set_vars()
  bool free_lunch;
};

// }
