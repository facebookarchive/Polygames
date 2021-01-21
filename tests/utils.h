/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Some utility functions for writing unit tests.

#pragma once

#include <iomanip>
#include <iostream>
#include <vector>

// Print a feature plane:
// printPlanes<const std::vector<float>&>(state.GetFeatures(), indexChannels, nbRows, nbCols);
template <typename T>
void printPlane(T data, int c, int ni, int nj) {
 
 for (int i=0; i<ni; i++) {
  for (int j=0; j<nj; j++) {
   std::cout << data[(c*ni + i)*nj +j] << " ";
  }
 std::cout << std::endl;
 }
 std::cout << std::endl;
}

// Print several feature planes:
// printPlanes<const std::vector<float>&>(state.GetFeatures(), nbChannels, nbRows, nbCols);
template <typename T>
void printPlanes(T data, int nc, int ni, int nj) {
 for (int c=0; c<nc; c++)
  printPlane<T>(data, c, ni, nj);
}

// Print raw data:
// printData<const std::vector<float>&>(state.GetFeatures());
template <typename T>
void printData(T data) {
 for (const auto & x : data)
  std::cout << x << " ";
 std::cout << std::endl;
}

template <typename T>
void printActions(std::vector<T> actions) {
 for (const auto & a : actions)
  std::cout << a.GetIndex() << " "
   << a.GetX() << " "
   << a.GetY() << " "
   << a.GetZ() << std::endl;
 std::cout << std::endl;
}

