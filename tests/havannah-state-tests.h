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

 template <int SIZE> class StateTest : public Havannah::State<SIZE> {
  public:
   StateTest<SIZE>(int seed, int history, bool turnFeatures) :
    Havannah::State<SIZE>(seed, history, turnFeatures) {}
   GameStatus GetStatus() { return ::State::_status; };
 };

};


///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(HavannahStateGroup, init_1) {

 const int size = 8;
 const int fullsize = 15;    // 2*size - 1
 const int nbChannels = 7;   // turnFeatures + 2*(1+history)
 const int nbActions = 169;  // fullsize*fullsize - size*(size-1)

 Havannah::StateTest<size> state(0, 2, true);

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // features
 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());
 ASSERT_EQ((std::vector<float>(nbChannels*fullsize*fullsize, 0.f)), state.GetFeatures());

 // actions
 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());
 ASSERT_EQ(nbActions, state.GetLegalActions().size());

}


TEST(HavannahStateGroup, init_2) {

 const int size = 3;
 const int fullsize = 5;     // 2*size - 1
 const int nbChannels = 2;   // turnFeatures + 2*(1+history)
 const int nbActions = 19;   // fullsize*fullsize - size*(size-1)

 Havannah::StateTest<size> state(0, 0, false);

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // features
 ASSERT_EQ((std::vector<int64_t>{nbChannels, fullsize, fullsize}), state.GetFeatureSize());
 ASSERT_EQ((std::vector<float>(nbChannels*fullsize*fullsize, 0.f)), state.GetFeatures());

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
  Havannah::State<4> state(0);
  auto clone = state.clone();
  auto ptrClone = dynamic_cast<Havannah::State<4> *>(clone.get());

  ASSERT_NE(&state, ptrClone);
  ASSERT_EQ(37, state.GetLegalActions().size());
  ASSERT_EQ(37, ptrClone->GetLegalActions().size());

  Havannah::Action<4> a(2, 3);
  state.ApplyAction(a);

  ASSERT_EQ(36, state.GetLegalActions().size());
  ASSERT_EQ(37, ptrClone->GetLegalActions().size());
 }
 catch (std::bad_cast) {
  FAIL() << "not a Havannah::State<4>"; 
 }

}


TEST(HavannahStateGroup, features_1) {

 const int size = 3;
 const int fullsize = 5;     // 2*size - 1
 const int nbChannels = 7;   // turnFeatures + 2*(1+history)
 const int nbActions = 19;   // fullsize*fullsize - size*(size-1)

 Havannah::StateTest<size> state(0, 2, true);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 std::vector<Havannah::Action<fullsize>> actions {{
     {1,2}, {2,2},
     {3,0}
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

  // turn
     0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f, 
     0.f, 0.f, 0.f, 0.f, 0.f

 };

 /*
 std::cout << "*** expected ***" << std::endl;
 printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 std::cout << "*** actual ***" << std::endl;
 printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);
 */

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HavannahStateGroup, features_2) {

 const int size = 3;
 const int fullsize = 5;     // 2*size - 1
 const int nbChannels = 7;   // turnFeatures + 2*(1+history)
 const int nbActions = 19;   // fullsize*fullsize - size*(size-1)

 Havannah::StateTest<size> state(0, 2, true);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 std::vector<Havannah::Action<fullsize>> actions {{
     {1,2}, {2,2}
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

  // turn
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f, 
     1.f, 1.f, 1.f, 1.f, 1.f

 };

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 /*
 // DEBUG
 std::cout << "*** expected ***" << std::endl;
 printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 std::cout << "*** actual ***" << std::endl;
 printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 std::cout << "*** expected ***" << std::endl;
 printData<const std::vector<float>&>(expectedFeatures);
 std::cout << "*** actual ***" << std::endl;
 printData<const std::vector<float>&>(state.GetFeatures());
 */

}


TEST(HavannahStateGroup, features_3) {

 const int size = 4;
 const int fullsize = 7;     // 2*size - 1
 const int nbChannels = 7;   // turnFeatures + 2*(1+history)
 const int nbActions = 37;   // fullsize*fullsize - size*(size-1)

 Havannah::StateTest<size> state(0, 2, true);

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, fullsize, fullsize}), state.GetActionSize());

 std::vector<Havannah::Action<fullsize>> actions {{
     {2,2}, {5,3}, 
     {1,4}, {2,3}, 
     {3,3}, {3,5}, 
     {2,4}, {6,2}, 
     {3,2}, {4,4}, 
     {0,4}, {2,6}, 
     {1,3}
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

   // turn

   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
   0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f

 };

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

 /*
 // DEBUG
 std::cout << "*** expected ***" << std::endl;
 printPlanes<const std::vector<float>&>(expectedFeatures, nbChannels, fullsize, fullsize);
 std::cout << "*** actual ***" << std::endl;
 printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, fullsize, fullsize);

 std::cout << "*** expected ***" << std::endl;
 printData<const std::vector<float>&>(expectedFeatures);
 std::cout << "*** actual ***" << std::endl;
 printData<const std::vector<float>&>(state.GetFeatures());
 */

 /*
 // Just a hack for converting to actions some indices 
 // (obtained using the GUI: https://gitlab.com/juliendehos/hex_hav).
 Havannah::Board<4> b;
 b.reset();
 std::vector<int> gameIndices = {
  16, 38,
  11, 17,
  24, 26,
  18, 44,
  23, 32,
   4, 20,
  10,
 };
 for (int i : gameIndices) {
  auto c = b.convertIndexToCell(i);
  std::cout << "{" << c.first << "," << c.second << "}, " << std::endl;
 }
 */

}


