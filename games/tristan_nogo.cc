/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "tristan_nogo.h"

using namespace std;

unsigned long long HashArray[3000];

bool useOrderMoves = true;

bool MonteCarloMoveOrdering = true;

bool printGame = false;

unsigned long long nbPlay = 0;

int level = 1;

timeval stop, start;
unsigned long long previousTime = 0;

bool useNotLosing = false;

bool useOrderPPAF = false;

void initHash() {
  for (int i = 0; i < 3000; i++) {
    HashArray[i] = 0;
    for (int k = 0; k < 64; k++)
      if ((rand() / (RAND_MAX + 1.0)) > 0.5)
        HashArray[i] |= (1ULL << k);
  }
}

bool useCode = true;

double history[MaxMoveNumber];

int interMove[MaxSize], moveInter[MaxSize];

bool ajoute(int* stack, int elt) {
  for (int i = 1; i <= stack[0]; i++)
    if (stack[i] == elt)
      return false;
  stack[0]++;
  stack[stack[0]] = elt;
  return true;
}
