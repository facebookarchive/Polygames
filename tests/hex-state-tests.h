/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for Hex Action/State.

#pragma once

#include <hex_state.h>


/*

///////////////////////////////////////////////////////////////////////////////
// helpers
///////////////////////////////////////////////////////////////////////////////

namespace Hex {

 template <int SIZE, bool PIE> class StateTest : public Hex::State<SIZE, PIE> {
  public:
   StateTest<SIZE, PIE>(int seed, int history, bool turnFeatures) : 
    Hex::State<SIZE, PIE>(seed, history, turnFeatures) {}
   GameStatus GetStatus() { return ::State::_status; };
 };

};


///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(HexStateGroup, init_1) {

 Hex::StateTest<7,true> state(0, 0, false);

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // features
 ASSERT_EQ((std::vector<int64_t>{2, 7, 7}), state.GetFeatureSize());
 ASSERT_EQ((std::vector<float>(2*7*7, 0.f)), state.GetFeatures());
 // ASSERT_EQ((std::vector<int64_t>{7, 7, 7}), state.GetFeatureSize());
 // ASSERT_EQ((std::vector<float>(7*7*7, 0.f)), state.GetFeatures());

 // actions
 ASSERT_EQ((std::vector<int64_t>{1, 7, 7}), state.GetActionSize());
 ASSERT_EQ(7*7, state.GetLegalActions().size());
 for (int k=0; k<state.GetLegalActions().size(); ++k) {
  int i = k / 7;
  int j = k % 7;
  auto a = state.GetLegalActions()[k];
  ASSERT_EQ(0, a->GetX());
  ASSERT_EQ(i, a->GetY());
  ASSERT_EQ(j, a->GetZ());
  ASSERT_EQ(k, a->GetHash());
  ASSERT_EQ(k, a->GetIndex());
 }
}


TEST(HexStateGroup, play_1) {

 Hex::StateTest<7,true> state(0, 0, false);

 Hex::Action<7> a(2, 3);
 state.ApplyAction(a);

 ASSERT_EQ(GameStatus::player1Turn, state.GetStatus());

 // features
 ASSERT_EQ((std::vector<int64_t>{2, 7, 7}), state.GetFeatureSize());
 for (int p=0; p<2; ++p) {
  for (int i=0; i<2; ++i) {
   for (int j=0; j<2; ++j) {
    int k = (p*2+i)*7+j;
    auto f_k = state.GetFeatures()[k];
    if (p==0 and i==2 and j==3)
     ASSERT_EQ(1, f_k);
    else 
     ASSERT_EQ(0, f_k);
   }
  }
 }

 // actions
 ASSERT_EQ((std::vector<int64_t>{1, 7, 7}), state.GetActionSize());
 ASSERT_EQ(7*7, state.GetLegalActions().size());
 for (int i=0; i<2; ++i) {
  for (int j=0; j<2; ++j) {
   int k = i*7+j;
   if (k<2*7+3) {
   auto a = state.GetLegalActions()[k];
    ASSERT_EQ(0, a->GetX());
    ASSERT_EQ(i, a->GetY());
    ASSERT_EQ(j, a->GetZ());
    ASSERT_EQ(k, a->GetHash());
    ASSERT_EQ(k, a->GetIndex());
   }
   else if (k>2*7+3) {
    int k2 = k-1;
    auto a = state.GetLegalActions()[k2];
    ASSERT_EQ(0, a->GetX());
    ASSERT_EQ(i, a->GetY());
    ASSERT_EQ(j, a->GetZ());
    ASSERT_EQ(k2, a->GetHash());
    ASSERT_EQ(k2, a->GetIndex());
   }
  }
 }

}


TEST(HexStateGroup, clone_1) {

 try {
  Hex::State<7,true> state(0);
  auto clone = state.clone();
  auto ptrClone = dynamic_cast<Hex::State<7,true> *>(clone.get());

  ASSERT_NE(&state, ptrClone);
  ASSERT_EQ(49, state.GetLegalActions().size());
  ASSERT_EQ(49, ptrClone->GetLegalActions().size());

  Hex::Action<7> a(2, 3);
  state.ApplyAction(a);

  ASSERT_EQ(49, state.GetLegalActions().size());
  ASSERT_EQ(49, ptrClone->GetLegalActions().size());
 }
 catch (std::bad_cast) {
  FAIL() << "not a Hex::State<7,true>"; 
 }

}


TEST(HexStateGroup, features_1) {

 Hex::StateTest<3,true> state(0, 2, true);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, 3, 3}), state.GetActionSize());

 std::vector<Hex::Action<3>> actions {{
     {1,0},
     {0,0}
 }};

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());
 ASSERT_EQ(9, state.GetLegalActions().size());
 state.ApplyAction(actions[0]);

 ASSERT_EQ(GameStatus::player1Turn, state.GetStatus());
 ASSERT_EQ(9, state.GetLegalActions().size());
 state.ApplyAction(actions[1]);

 ASSERT_EQ(7, state.GetLegalActions().size());
 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{7, 3, 3}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {

     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     1.f, 1.f, 1.f,
     1.f, 1.f, 1.f,
     1.f, 1.f, 1.f

 };

 // DEBUG
 // printPlanes<std::vector<float>>(state.GetFeatures(), 7, 3, 3);
 // printPlanes<std::vector<float>>(expectedFeatures, 7, 3, 3);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HexStateGroup, features_2) {

 Hex::StateTest<3,false> state(0, 2, true);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, 3, 3}), state.GetActionSize());

 std::vector<Hex::Action<3>> actions {{
     {1,1}, {0,0},
     {2,2}, {2,0},
     {1,0}
 }};

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = 9;
 for (const auto & a : actions) {
     ASSERT_EQ(currentPlayer, state.GetStatus());
     ASSERT_EQ(k, state.GetLegalActions().size());
     state.ApplyAction(a);
     std::swap(currentPlayer, nextPlayer);
     k--;
     ASSERT_EQ(k, state.GetLegalActions().size());
 }
 ASSERT_EQ(GameStatus::player1Turn, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{7, 3, 3}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {

     0.f, 0.f, 0.f,
     0.f, 1.f, 0.f,
     0.f, 0.f, 1.f,

     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     0.f, 1.f, 0.f,
     0.f, 0.f, 1.f,

     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     1.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     1.f, 1.f, 0.f,
     0.f, 0.f, 1.f,

     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     1.f, 0.f, 0.f,

     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f

 };

 // DEBUG
 // printPlanes<std::vector<float>>(state.GetFeatures(), 7, 3, 3);
 // printPlanes<std::vector<float>>(expectedFeatures, 7, 3, 3);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HexStateGroup, features_3) {

 Hex::StateTest<9,false> state(0, 2, true);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, 9, 9}), state.GetActionSize());

 std::vector<Hex::Action<9>> actions {{
   {4,1}, {2,3},
   {5,2}, {2,5},
   {4,4}, {2,6},
   {5,5}, {7,4},
   {4,7}, {7,6},
   {3,8}, {5,6},
   {4,6}, {4,5},
   {5,4}, {5,3},
   {4,3}, {4,2},
   {5,1}, {5,0},
   {4,0}
 }};

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = 9*9;
 for (const auto & a : actions) {
     ASSERT_EQ(currentPlayer, state.GetStatus());
     ASSERT_EQ(k, state.GetLegalActions().size());
     state.ApplyAction(a);
     std::swap(currentPlayer, nextPlayer);
     k--;
     ASSERT_EQ(k, state.GetLegalActions().size());
     // std::cout << a.to_string() << std::endl;
 }
 ASSERT_EQ(GameStatus::player0Win, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{7, 9, 9}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
     0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
     0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
     0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
     0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
     1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
     1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
     0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
     1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f


 };

 // DEBUG
 // printPlanes<std::vector<float>>(state.GetFeatures(), 7, 9, 9);
 // printPlanes<std::vector<float>>(expectedFeatures, 7, 9, 9);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 // // Just a hack for converting to actions some indices 
 // // (obtained using the GUI: https://gitlab.com/juliendehos/hex_hav).
 // Hex::Board<9> b;
 // b.reset();
 // std::vector<int> gameIndices = {
 //  37, 21,
 //  47, 23,
 //  40, 24,
 //  50, 67,
 //  43, 69,
 //  35, 51,
 //  42, 41,
 //  49, 48,
 //  39, 38,
 //  46, 45,
 //  36
 // };
 // for (int i : gameIndices) {
 //  auto c = b.convertIndexToCell(i);
 //  std::cout << "{" << c.first << "," << c.second << "}, " << std::endl;
 // }

}

*/

