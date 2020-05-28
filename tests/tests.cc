/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests runner.

#include <gtest/gtest.h>

#include <ludii/jni_utils.h>
JNIEnv * JNI_ENV = nullptr;

int main(int argc, char **argv) {
    if (argc == 2) {
        Ludii::JNIUtils::InitJVM(argv[1]);
        JNI_ENV = Ludii::JNIUtils::GetEnv();
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    else {
        ::testing::InitGoogleTest(&argc, argv);
        int res = RUN_ALL_TESTS();
        std::cout << "WARNING: no ludii jar file specified!\n";
        return res;
    }
}

