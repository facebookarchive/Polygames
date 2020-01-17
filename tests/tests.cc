/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests runner.

// Include your unit test files here.

#include <cassert>
#include <cmath>
#include <gtest/gtest.h>

#include "utils.h"

#include "connectfour-tests.h"

#include "havannah-tests.h"
#include "havannah-state-tests.h"

#include "hex-tests.h"
#include "hex-state-tests.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

