/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for Havannah Action/State.

#pragma once

#include <havannah_state.h>

///////////////////////////////////////////////////////////////////////////////
// helpers
///////////////////////////////////////////////////////////////////////////////

namespace Havannah {

 template <int SIZE, bool PIE, bool EXTENDED> class StateTest :
   public Havannah::State<SIZE, PIE, EXTENDED> {
  public:
   StateTest<SIZE, PIE, EXTENDED>(int seed, int history, bool turnFeatures) :
    Havannah::State<SIZE, PIE, EXTENDED>(seed, history, turnFeatures) {}
   GameStatus GetStatus() { return ::State::_status; };
 };

};


///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(HavannahStateGroup, init_0) {

 const int size = 5;
 const int history = 0;
 const bool turnFeatures = true;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, true, false> state(0, history, turnFeatures);

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // features
 std::vector<float> expectedFeatures(nbChannels*fullsize*fullsize, 0.f);
 const std::vector<float> boardFeatures = {
  0, 0, 0, 0, 1, 1, 1, 1, 1, 
  0, 0, 0, 1, 1, 1, 1, 1, 1, 
  0, 0, 1, 1, 1, 1, 1, 1, 1, 
  0, 1, 1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 1, 0, 
  1, 1, 1, 1, 1, 1, 1, 0, 0, 
  1, 1, 1, 1, 1, 1, 0, 0, 0, 
  1, 1, 1, 1, 1, 0, 0, 0, 0
 };
 const int f2 = fullsize*fullsize;
 std::copy(boardFeatures.begin(), boardFeatures.end(), expectedFeatures.begin() + 2*f2);

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());
 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 // actions
 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());
 ASSERT_EQ(nbActions, state.GetLegalActions().size());

}


TEST(HavannahStateGroup, init_1) {

 const int size = 8;
 const int history = 2;
 const bool turnFeatures = true;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, true, false> state(0, history, turnFeatures);

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // features
 std::vector<float> expectedFeatures(nbChannels*fullsize*fullsize, 0.f);
 const std::vector<float> boardFeatures = {
  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
  1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
  1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
 };
 const int f2 = fullsize*fullsize;
 std::copy(boardFeatures.begin(), boardFeatures.end(), expectedFeatures.begin() + 2*f2);
 std::copy(boardFeatures.begin(), boardFeatures.end(), expectedFeatures.begin() + 5*f2);
 std::copy(boardFeatures.begin(), boardFeatures.end(), expectedFeatures.begin() + 8*f2);

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());
 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 // actions
 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());
 ASSERT_EQ(nbActions, state.GetLegalActions().size());

}


TEST(HavannahStateGroup, init_2) {

 const int size = 3;
 const int history = 0;
 const bool turnFeatures = false;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, true, false> state(0, history, turnFeatures);

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // features
 std::vector<float> expectedFeatures(nbChannels*fullsize*fullsize, 0.f);
 const std::vector<float> boardFeatures = {
  0, 0, 1, 1, 1, 
  0, 1, 1, 1, 1, 
  1, 1, 1, 1, 1, 
  1, 1, 1, 1, 0, 
  1, 1, 1, 0, 0
 };
 const int f2 = fullsize*fullsize;
 std::copy(boardFeatures.begin(), boardFeatures.end(), expectedFeatures.begin() + 2*f2);

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());
 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 // actions
 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());
 ASSERT_EQ(nbActions, state.GetLegalActions().size());

 std::vector<std::pair<int,int>> actions {{
            {0,2}, {0,3}, {0,4},

        {1,1}, {1,2}, {1,3}, {1,4},

     {2,0}, {2,1}, {2,2}, {2,3}, {2,4},

        {3,0}, {3,1}, {3,2}, {3,3},

            {4,0}, {4,1}, {4,2}
 }};

 for (int k=0; k<nbActions; k++) {
  auto expectedAction = actions[k];
  auto action = state.GetLegalActions()[k];
  int i = expectedAction.first;
  int j = expectedAction.second;
  int h = i*fullsize + j;
  ASSERT_EQ(0, action->GetX());
  ASSERT_EQ(i, action->GetY());
  ASSERT_EQ(j, action->GetZ());
  ASSERT_EQ(h, action->GetHash());
  ASSERT_EQ(k, action->GetIndex());
 }

}


TEST(HavannahStateGroup, clone_1) {

 try {
  Havannah::State<4, true, false> state(0);
  auto clone = state.clone();
  auto ptrClone = dynamic_cast<Havannah::State<4, true, false> *>(clone.get());

  ASSERT_NE(&state, ptrClone);
  ASSERT_EQ(37, state.GetLegalActions().size());
  ASSERT_EQ(37, ptrClone->GetLegalActions().size());

  Havannah::Action<4> a(2, 3, 11);
  // 0 0 0 1 1 1 1 
  // 0 0 1 1 1 1 1 
  // 0 1 1 A 1 1 1 
  // 1 1 1 1 1 1 1 
  // 1 1 1 1 1 1 0 
  // 1 1 1 1 1 0 0 
  // 1 1 1 1 0 0 0 
  // DEBUG
  // std::cout << state.actionDescription(a) << std::endl;
  // auto pa11 = state.GetLegalActions()[11];
  // std::cout << state.actionDescription(*pa11) << std::endl;
  // for (auto pa : state.GetLegalActions())
  //     std::cout << state.actionDescription(*pa) << " ";
  // std::cout << std::endl;

  state.ApplyAction(a);

  ASSERT_EQ(37, state.GetLegalActions().size());
  // still 37 actions becauseof swap

  ASSERT_EQ(37, ptrClone->GetLegalActions().size());
 }
 catch (std::bad_cast) {
  FAIL() << "not a Havannah::State<4, true, false>"; 
 }

}


TEST(HavannahStateGroup, clone_2) {

 try {
  Havannah::State<4, true, false> state(0);
  auto clone = state.clone();
  auto ptrClone = dynamic_cast<Havannah::State<4, true, false> *>(clone.get());

  ASSERT_NE(&state, ptrClone);
  ASSERT_EQ(37, state.GetLegalActions().size());
  ASSERT_EQ(37, ptrClone->GetLegalActions().size());

  Havannah::Action<4> a(2, 3, -1);
  state.ApplyAction(a);
  state.ApplyAction(a);  // swap
  ASSERT_EQ(36, state.GetLegalActions().size());
  ASSERT_EQ(37, ptrClone->GetLegalActions().size());
 }
 catch (std::bad_cast) {
  FAIL() << "not a Havannah::State<4, true, false>"; 
 }

}


TEST(HavannahStateGroup, features_1_pie) {

 const int size = 3;
 const int history = 2;
 const bool turnFeatures = true;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, true, false> state(0, history, turnFeatures);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = nbActions;
 ASSERT_EQ(currentPlayer, state.GetStatus());
 ASSERT_EQ(k, state.GetLegalActions().size());

 // first action
 const Havannah::Action<fullsize> a0 {1,2,-1};
 state.ApplyAction(a0);
 std::swap(currentPlayer, nextPlayer);
 ASSERT_EQ(currentPlayer, state.GetStatus());
 ASSERT_EQ(k, state.GetLegalActions().size());

 // second action
 const Havannah::Action<fullsize> a1 {2,2,-1};
 state.ApplyAction(a1);
 std::swap(currentPlayer, nextPlayer);
 k -= 2;
 ASSERT_EQ(currentPlayer, state.GetStatus());
 ASSERT_EQ(k, state.GetLegalActions().size());

 // next actions
 const std::vector<Havannah::Action<fullsize>> actions {{
     {3,0,-1}
 }};
 for (const auto & a : actions) {
     state.ApplyAction(a);
     std::swap(currentPlayer, nextPlayer);
     k--;
     ASSERT_EQ(currentPlayer, state.GetStatus());
     ASSERT_EQ(k, state.GetLegalActions().size());
 }
 ASSERT_EQ(GameStatus::player1Turn, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {
  // history - 2, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 2, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 2, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // history - 1, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 1, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 1, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // history - 0, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     1.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // turn
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f

 };

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HavannahStateGroup, features_1_nopie) {

 const int size = 3;
 const int history = 2;
 const bool turnFeatures = true;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, false, false> state(0, history, turnFeatures);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 std::vector<Havannah::Action<fullsize>> actions {{
     {1,2,-1}, {2,2,-1},
     {3,0,-1}
 }};

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = nbActions;
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

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {
  // history - 2, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 2, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 2, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // history - 1, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 1, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 1, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // history - 0, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     1.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // turn
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f

 };

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HavannahStateGroup, features_2_nopie) {

 const int size = 3;
 const int history = 2;
 const bool turnFeatures = true;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, false, false> state(0, history, turnFeatures);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 std::vector<Havannah::Action<fullsize>> actions {{
     {1,2,-1}, {2,2,-1}
 }};

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = nbActions;
 for (const auto & a : actions) {
     ASSERT_EQ(currentPlayer, state.GetStatus());
     ASSERT_EQ(k, state.GetLegalActions().size());
     state.ApplyAction(a);
     std::swap(currentPlayer, nextPlayer);
     k--;
     ASSERT_EQ(k, state.GetLegalActions().size());
 }
 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {
  // history - 2, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 2, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 2, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // history - 1, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 1, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 1, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // history - 0, player 0
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, board cells
     0.f, 0.f, 1.f, 1.f, 1.f, 
     0.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 0.f, 
     1.f, 1.f, 1.f, 0.f, 0.f, 

  // turn
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f,

 };

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 // std::cout << "*** expected ***" << std::endl;
 // printData<const std::vector<float>&>(expectedFeatures);
 // std::cout << "*** actual ***" << std::endl;
 // printData<const std::vector<float>&>(state.GetFeatures());

}


TEST(HavannahStateGroup, features_3_nopie) {

 const int size = 4;
 const int history = 2;
 const bool turnFeatures = true;
 const int fullsize = 2*size - 1;
 const int nbChannels = 3*(1+history) + (turnFeatures ? 1 : 0);
 const int nbActions = fullsize*fullsize - size*(size-1);

 Havannah::StateTest<size, false, false> state(0, history, turnFeatures);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 std::vector<Havannah::Action<fullsize>> actions {{
     {2,2,-1}, {5,3,-1}, 
     {1,4,-1}, {2,3,-1}, 
     {3,3,-1}, {3,5,-1}, 
     {2,4,-1}, {6,2,-1}, 
     {3,2,-1}, {4,4,-1}, 
     {0,4,-1}, {2,6,-1}, 
     {1,3,-1}
 }};

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = nbActions;
 for (const auto & a : actions) {
     ASSERT_EQ(currentPlayer, state.GetStatus());
     ASSERT_EQ(k, state.GetLegalActions().size());
     state.ApplyAction(a);
     std::swap(currentPlayer, nextPlayer);
     k--;
     ASSERT_EQ(k, state.GetLegalActions().size());
 }
 ASSERT_EQ(GameStatus::player0Win, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {
   // history - 2, player 0
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 

   // history - 2, player 1
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 

   // history - 2, board cells
   0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 
   0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 
   1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 

   // history - 1, player 0
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 

   // history - 1, player 1
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 

   // history - 1, board cells
   0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 
   0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 
   1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 

   // history - 0, player 0
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 

   // history - 0, player 1
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 

   // history - 0, board cells
   0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 
   0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 
   1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 

   // turn
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 
   1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f

 };

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 // std::cout << "*** expected ***" << std::endl;
 // printData<const std::vector<float>&>(expectedFeatures);
 // std::cout << "*** actual ***" << std::endl;
 // printData<const std::vector<float>&>(state.GetFeatures());

 // Just a hack for converting to actions some indices 
 // (obtained using the GUI: https://gitlab.com/juliendehos/hex_hav).
 // Havannah::Board<4> b;
 // b.reset();
 // std::vector<int> gameIndices = {
 //  16, 38,
 //  11, 17,
 //  24, 26,
 //  18, 44,
 //  23, 32,
 //   4, 20,
 //  10,
 // };
 // for (int i : gameIndices) {
 //  auto c = b.convertIndexToCell(i);
 //  std::cout << "{" << c.first << "," << c.second << "}, " << std::endl;
 // }

}


