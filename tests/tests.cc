/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests runner.

#include <gtest/gtest.h>

std::string LUDII_PATH = "";

int main(int argc, char** argv) {
  if (argc > 1) LUDII_PATH = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
