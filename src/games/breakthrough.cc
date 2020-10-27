/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "breakthrough.h"
#include <list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

unsigned long long BTHashArray[2][BTDx][BTDy];
unsigned long long BTHashTurn;

// int level = 1;

// unsigned long long nbPlay = 0;

// timeval stop, start;
// unsigned long long previousTime = 0;

// bool BTinitHashCalled = false;
std::once_flag BTinitHashCalled;

void BTinitHash() {
  for (int player = 0; player < 2; player++)
    for (int i = 0; i < BTDx; i++)
      for (int j = 0; j < BTDy; j++) {
        BTHashArray[player][i][j] = 0;
        for (int k = 0; k < 64; k++)
          if ((rand() / (RAND_MAX + 1.0)) > 0.5)
            BTHashArray[player][i][j] |= (1ULL << k);
      }
  BTHashTurn = 0;
  for (int k = 0; k < 64; k++)
    if ((rand() / (RAND_MAX + 1.0)) > 0.5)
      BTHashTurn |= (1ULL << k);
}
