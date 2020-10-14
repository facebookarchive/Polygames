/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ludii/jni_utils.h>
#include <ludii/ludii_state_wrapper.h>

#include "utils.h"
#include <gtest/gtest.h>

///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

TEST(LudiiGameGroup, ludii_yavalath_0) {
  Ludii::JNIUtils::InitJVM("");  // Use default /ludii/Ludii.jar path
  JNIEnv* jni_env = Ludii::JNIUtils::GetEnv();
  EXPECT_TRUE(jni_env);

  Ludii::LudiiGameWrapper game_wrapper("Yavalath.lud");
  Ludii::LudiiStateWrapper state =
      Ludii::LudiiStateWrapper(0, std::move(game_wrapper));
  state.Initialize();

  ASSERT_EQ((std::vector<int64_t>{10, 9, 17}), state.GetFeatureSize());
  ASSERT_EQ((std::vector<int64_t>{3, 9, 17}), state.GetActionSize());
  ASSERT_EQ(GameStatus::player0Turn, GameStatus(state.getCurrentPlayer()));

  // We expect the following meanings for Yavalath state tensor channels:
  // 0: Piece Type 1 (Ball1)
  // 1: Piece Type 2 (Ball2)
  // 2: Is Player 1 the current mover?
  // 3: Is Player 2 the current mover?
  // 4: Did Swap Occur?
  // 5: Does position exist in container 0 (Board)?
  // 6: Last move's from-position
  // 7: Last move's to-position
  // 8: Second-to-last move's from-position
  // 9: Second-to-last move's to-position

  // TODO guess we really need a channel to indicate that swap happened
  const std::vector<float> features = state.GetFeatures();

  // We expect empty board initial state, so first two channels
  // should be all-zero
  size_t i = 0;
  while (i < 2 * 9 * 17) {
    ASSERT_EQ(0, features[i]);
    ++i;
  }

  // Player 1 should be mover, so expect channel filled with 1s next
  while (i < 3 * 9 * 17) {
    ASSERT_EQ(1, features[i]);
    ++i;
  }

  // Player 2 not current mover, so full channel of 0s
  while (i < 4 * 9 * 17) {
    ASSERT_EQ(0, features[i]);
    ++i;
  }
  
  // No swap occured yet, so expect full channel of 0s
  while (i < 5 * 9 * 17) {
    ASSERT_EQ(0, features[i]);
    ++i;
  }
  
  // Channel: Does position exist in container 0 (Board)?
  // First and last column have 5 cells each, 
  // expected pattern: 0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0
  const float _5_cells_pattern[17] = {0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0};
  for (size_t j = 0; j < 17; ++j) {
    ASSERT_EQ(features[i + 0 * 17 + j], _5_cells_pattern[j]);
	ASSERT_EQ(features[i + 8 * 17 + j], _5_cells_pattern[j]);
  }
  
  // Second and second-to-last column have 6 cells each,
  // expected pattern: 0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0
  const float _6_cells_pattern[17] = {0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0};
  for (size_t j = 0; j < 17; ++j) {
    ASSERT_EQ(features[i + 1 * 17 + j], _6_cells_pattern[j]);
	ASSERT_EQ(features[i + 7 * 17 + j], _6_cells_pattern[j]);
  }
  
  // Third and third-to-last column have 7 cells each, 
  // expected pattern: 0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0
  const float _7_cells_pattern[17] = {0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0};
  for (size_t j = 0; j < 17; ++j) {
    ASSERT_EQ(features[i + 2 * 17 + j], _7_cells_pattern[j]);
	ASSERT_EQ(features[i + 6 * 17 + j], _7_cells_pattern[j]);
  }
  
  // Fourth and fourth-to-last column have 8 cells each,
  // expected pattern: 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0
  const float _8_cells_pattern[17] = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0};
  for (size_t j = 0; j < 17; ++j) {
    ASSERT_EQ(features[i + 3 * 17 + j], _8_cells_pattern[j]);
	ASSERT_EQ(features[i + 5 * 17 + j], _8_cells_pattern[j]);
  }
  
  // Middle column has 9 cells,
  // expected pattern: 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
  const float _9_cells_pattern[17] = {1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
  for (size_t j = 0; j < 17; ++j) {
    ASSERT_EQ(features[i + 4 * 17 + j], _9_cells_pattern[j]);
  }
  i += 9 * 17;
  
  // All remaining channels should be all-zero; no moves played
  while (i < 10 * 9 * 17) {
    ASSERT_EQ(0, features[i]);
    ++i;
  }
}
