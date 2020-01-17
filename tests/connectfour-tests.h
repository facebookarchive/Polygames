/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Just a few tests for understanding the public methods of Action and State,
// with an already implemented game. It may be interesting to write "real" unit
// tests.

#pragma once

#include <connectfour.h>

TEST(Connectfour, init_1) {

 StateForConnectFour state(0);
 state.Initialize();

 ASSERT_EQ((std::vector<int64_t>{3, 6, 7}), state.GetFeatureSize());
 ASSERT_EQ((std::vector<int64_t>{7, 1, 1}), state.GetActionSize());
 ASSERT_EQ(GameStatus::player0Turn, GameStatus(state.getCurrentPlayer()));

 for (int i=0; i<7; ++i) {
  auto a_i = std::dynamic_pointer_cast<ActionForConnectFour>(state.GetLegalActions()[i]);
  ASSERT_EQ(i, a_i->GetX());
  ASSERT_EQ(0, a_i->GetY());
  ASSERT_EQ(0, a_i->GetZ());
  ASSERT_EQ(i, a_i->GetHash());
  ASSERT_EQ(i, a_i->GetIndex());
 }

 std::vector<float> expectedFeatures {

  // history - 0, player 0
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

 };

 // DEBUG
 // printPlanes<std::vector<float>>(state.GetFeatures(), 3, 6, 7);
 // printPlanes<std::vector<float>>(expectedFeatures, 3, 6, 7);
 // printData<std::vector<float>>(state.GetFeatures());
 // printData<std::vector<float>>(expectedFeatures);

 ASSERT_EQ(expectedFeatures.size(), 3*6*7);
 ASSERT_EQ(state.GetFeatures().size(), 3*6*7);
 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


TEST(Connectfour, play_1) {

 StateForConnectFour state(0);
 state.Initialize();

 ActionForConnectFour action(1, 7);
 state.ApplyAction(action);

 ASSERT_EQ((std::vector<int64_t>{3, 6, 7}), state.GetFeatureSize());
 ASSERT_EQ((std::vector<int64_t>{7, 1, 1}), state.GetActionSize());
 ASSERT_EQ(GameStatus::player1Turn, GameStatus(state.getCurrentPlayer()));

 for (int i=0; i<7; ++i) {
  auto a_i = std::dynamic_pointer_cast<ActionForConnectFour>(state.GetLegalActions()[i]);
  ASSERT_EQ(i, a_i->GetX());
  ASSERT_EQ(0, a_i->GetY());
  ASSERT_EQ(0, a_i->GetZ());
  ASSERT_EQ(i, a_i->GetHash());
  ASSERT_EQ(i, a_i->GetIndex());
 }

 std::vector<float> expectedFeatures {

  // history - 0, player 0
     0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

  // history - 0, player 1
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
     1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,

 };

 // DEBUG
 // printPlanes<std::vector<float>>(state.GetFeatures(), 3, 6, 7);
 // printPlanes<std::vector<float>>(expectedFeatures, 3, 6, 7);

 ASSERT_EQ(expectedFeatures.size(), 3*6*7);
 ASSERT_EQ(state.GetFeatures().size(), 3*6*7);
 ASSERT_EQ(expectedFeatures, state.GetFeatures());

}


