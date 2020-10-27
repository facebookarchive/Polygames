/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

// A _nbSubsets x _maxNumber matrix storing if subset i can host number j,
// where (i, j) in [1, _nbSubsets] x [1, _maxNumber]
class SchurMatrix {
    private:
        int _nbSubsets;
        int _maxNumber;
        std::vector<bool> _data;
    public:
        SchurMatrix(int nbSubsets, int maxNumber);
        void reset(bool value);
        bool get(int i, int j) const;
        void set(int i, int j, bool b);
        const std::vector<bool> & data() const;
};

