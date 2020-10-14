/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CZF_GAME_GAME_PLAYER_H_
#define CZF_GAME_GAME_PLAYER_H_

enum PLAYER {
  PLAYER_0 = 0u,
  PLAYER_1 = 1u,
  PLAYER_NULL = 2u,  // if an action's player is null, it is illegal
  PLAYER_SIZE = 3u
};

inline PLAYER operator!(PLAYER player) {
  return (PLAYER)((unsigned int)player ^ 1);
}

inline PLAYER IntToPlayer(int i) {
  if (i == 0)
    return PLAYER_0;
  else
    return PLAYER_1;
}

#endif  // CZF_GAME_GAME_PLAYER_H_
