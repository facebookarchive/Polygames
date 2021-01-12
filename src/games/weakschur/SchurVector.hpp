/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

// 1-based indexed vector of int
class SchurVector {
    private:
        const int _maxIndex;
        std::vector<int> _data;
    public:
        SchurVector(int maxIndex);
        void reset(int value);
        int get(int i) const;
        void set(int i, int value);
        const std::vector<int> & data() const;
};

