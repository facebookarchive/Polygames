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
}
