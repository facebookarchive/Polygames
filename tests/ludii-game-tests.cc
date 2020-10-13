/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ludii/jni_utils.h>
#include <ludii/ludii_state_wrapper.h>

#include <gtest/gtest.h>
#include "utils.h"

///////////////////////////////////////////////////////////////////////////////
// unit tests
///////////////////////////////////////////////////////////////////////////////

extern JNIEnv * JNI_ENV;

TEST(LudiiGameGroup, ludii_yavalath_0) {
  if (JNI_ENV == nullptr) {
    return;
  }
  
  Ludii::LudiiGameWrapper game_wrapper("LudiiYavalath.lud");
  Ludii::LudiiStateWrapper state = std::make_unique<Ludii::LudiiStateWrapper>(
      seed, std::move(game_wrapper));
  state.Initialize();
  
  ASSERT_EQ((std::vector<int64_t>{9, 18, 18}), state.GetFeatureSize());
  ASSERT_EQ((std::vector<int64_t>{3, 18, 18}), state.GetActionSize());
  ASSERT_EQ(GameStatus::player0Turn, GameStatus(state.getCurrentPlayer()));
  
  // We expect the following meanings for Yavalath state tensor channels:
  // 0: Piece Type 1 (Ball1)
  // 1: Piece Type 2 (Ball2)
  // 2: Is Player 1 the current mover?
  // 3: Is Player 2 the current mover?
  // 4: Does position exist in container 0 (Board)?
  // 5: Last move's from-position
  // 6: Last move's to-position
  // 7: Second-to-last move's from-position
  // 8: Second-to-last move's to-position
  
  // TODO guess we really need a channel to indicate that swap happened
  const std::vector<float> features = state.GetFeatures();
  
  // We expect empty board initial state, so first two channels
  // should be all-zero
  size_t i = 0;
  while (i < 2 * 18 * 18) {
    ASSERT_EQ(0, features[i]);
	++i;
  }
  
  // Player 1 should be mover, so expect channel filled with 1s next
  while (i < 3 * 18 * 18) {
    ASSERT_EQ(1, features[i]);
	++i;
  }
  
  // Player 2 not current mover, so full channel of 0s
  while (i < 4 * 18 * 18) {
    ASSERT_EQ(0, features[i]);
	++i;
  }
}

