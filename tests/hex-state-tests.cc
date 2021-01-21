/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for Hex Action/State.

#include <games/hex_state.h>
#include <gtest/gtest.h>
#include "utils.h"

///////////////////////////////////////////////////////////////////////////////
// helpers
///////////////////////////////////////////////////////////////////////////////

namespace Hex {

 template <int SIZE, bool PIE> class StateTest : public Hex::State<SIZE, PIE> {
  public:
   core::FeatureOptions _opts;

   StateTest<SIZE, PIE>(int seed, int history, bool turnFeatures) : 
    Hex::State<SIZE, PIE>(seed) {
     _opts.history = history;
     _opts.turnFeaturesMultiChannel = turnFeatures;
     core::State::setFeatures(&_opts);
    }

   StateTest<SIZE, PIE>(int seed) : 
    Hex::State<SIZE, PIE>(seed) {
    }

   GameStatus GetStatus() { return core::State::_status; }
   void addAction(int x, int y, int z) { core::State::addAction(x, y, z); }
 };

};


///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(HexStateGroup, init_1) {

 Hex::StateTest<7,true> state(0, 0, false);
 state.Initialize();

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
  ASSERT_EQ(0, a.GetX());
  ASSERT_EQ(i, a.GetY());
  ASSERT_EQ(j, a.GetZ());
  ASSERT_EQ(0, a.GetHash());
  ASSERT_EQ(k, a.GetIndex());
 }
}


TEST(HexStateGroup, clone_1) {

 try {
  Hex::State<7,true> state(0);
  state.Initialize();
  auto clone = state.clone();
  auto ptrClone = dynamic_cast<Hex::State<7,true> *>(clone.get());

  ASSERT_NE(nullptr, ptrClone);
  ASSERT_NE(&state, ptrClone);
  ASSERT_EQ(49, state.GetLegalActions().size());
  ASSERT_EQ(49, ptrClone->GetLegalActions().size());
 }
 catch (std::bad_cast) {
  FAIL() << "not a Hex::State<7,true>"; 
 }

}


TEST(HexStateGroup, features_1) {

 Hex::StateTest<3,true> state(0, 2, true);
 state.Initialize();

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, 3, 3}), state.GetActionSize());

 // DEBUG printActions(state.GetLegalActions());

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;

 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());
 ASSERT_EQ(9, state.GetLegalActions().size());
 state.ApplyAction( _Action(3, 0, 1, 0) );

 ASSERT_EQ(GameStatus::player1Turn, state.GetStatus());
 ASSERT_EQ(9, state.GetLegalActions().size());
 state.ApplyAction( _Action(0, 0, 0, 0) );

 ASSERT_EQ(7, state.GetLegalActions().size());
 ASSERT_EQ(GameStatus::player0Turn, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{8, 3, 3}), state.GetFeatureSize());

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
     1.f, 1.f, 1.f,

     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f

 };

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<std::vector<float>>(expectedFeatures, 8, 3, 3);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<std::vector<float>>(state.GetFeatures(), 8, 3, 3);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HexStateGroup, features_2) {

 Hex::StateTest<3,false> state(0, 2, true);
 state.Initialize();

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, 3, 3}), state.GetActionSize());

 std::vector<_Action> actions {
     _Action(0, 0, 1, 1), _Action(0, 0, 0, 0),
     _Action(0, 0, 2, 2), _Action(0, 0, 2, 0),
     _Action(0, 0, 1, 0)
 };

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

 ASSERT_EQ((std::vector<int64_t>{8, 3, 3}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {

     // history - 2, player 0
     0.f, 0.f, 0.f,
     0.f, 1.f, 0.f,
     0.f, 0.f, 1.f,

     // history - 2, player 1
     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     // history - 1, player 0
     0.f, 0.f, 0.f,
     0.f, 1.f, 0.f,
     0.f, 0.f, 1.f,

     // history - 1, player 1
     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     1.f, 0.f, 0.f,

     // history - 1, player 0
     0.f, 0.f, 0.f,
     1.f, 1.f, 0.f,
     0.f, 0.f, 1.f,

     // history - 0, player 1
     1.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     1.f, 0.f, 0.f,

     // 
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,
     0.f, 0.f, 0.f,

     // turn
     1.f, 1.f, 1.f,
     1.f, 1.f, 1.f,
     1.f, 1.f, 1.f

 };

 // DEBUG
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<std::vector<float>>(expectedFeatures, 8, 3, 3);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<std::vector<float>>(state.GetFeatures(), 8, 3, 3);

 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(HexStateGroup, features_3) {
 const int history = 2;
 const int size = 9;
 const bool turnFeatures = true;
 const int nbChannels = 2*(1 + history) + (turnFeatures ? 1 : 0) + 1;

 Hex::StateTest<size,false> state(0, history, turnFeatures);
 state.Initialize();

 // apply actions

 ASSERT_EQ((std::vector<int64_t>{1, size, size}), state.GetActionSize());

 std::vector<_Action> actions {
   _Action(0,0,0,0), _Action(0,0,4,1),
   _Action(0,0,2,3), _Action(0,0,5,2),
   _Action(0,0,2,5), _Action(0,0,4,4),
   _Action(0,0,2,6), _Action(0,0,5,5),
   _Action(0,0,7,4), _Action(0,0,4,7),
   _Action(0,0,7,6), _Action(0,0,3,8),
   _Action(0,0,5,6), _Action(0,0,4,6),
   _Action(0,0,4,5), _Action(0,0,5,4),
   _Action(0,0,5,3), _Action(0,0,4,3),
   _Action(0,0,4,2), _Action(0,0,5,1),
   _Action(0,0,5,0), _Action(0,0,4,0)
 };

 auto currentPlayer = GameStatus::player0Turn;
 auto nextPlayer = GameStatus::player1Turn;
 int k = size*size;
 for (const auto & a : actions) {
     ASSERT_EQ(currentPlayer, state.GetStatus());
     ASSERT_EQ(k, state.GetLegalActions().size());
     state.ApplyAction(a);
     std::swap(currentPlayer, nextPlayer);
     k--;
     ASSERT_EQ(k, state.GetLegalActions().size());
     // std::cout << a.to_string() << std::endl;
 }
 ASSERT_EQ(GameStatus::player1Win, state.GetStatus());

 // check features

 ASSERT_EQ((std::vector<int64_t>{nbChannels, size, size}), state.GetFeatureSize());

 std::vector<float> expectedFeatures {

     // history - 2, player 0
     1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     // history - 2, player 1
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
     0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
     0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     // history - 1, player 0
     1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
     1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     // history - 1, player 1
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
     0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
     0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     // history - 0, player 0
     1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f,
     1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     // history - 0, player 1
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
     1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
     0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

     //
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

     // turn
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
 // std::cout << "*** expected ***" << std::endl;
 // printPlanes<std::vector<float>>(expectedFeatures, nbChannels, size, size);
 // std::cout << "*** actual ***" << std::endl;
 // printPlanes<std::vector<float>>(state.GetFeatures(), nbChannels, size, size);

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

