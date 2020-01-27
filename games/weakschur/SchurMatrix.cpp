/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SchurMatrix.hpp"

#include <cassert>

SchurMatrix::SchurMatrix(int nbSubsets, int maxNumber) :
    _nbSubsets(nbSubsets),
    _maxNumber(maxNumber),
    _data(_nbSubsets*_maxNumber)
{}

void SchurMatrix::reset(bool value) {
    std::fill(_data.begin(), _data.end(), value);
}

bool SchurMatrix::get(int i, int j) const {
    assert(i >= 1);
    assert(i <= _nbSubsets);
    assert(j >= 1);
    assert(j <= _maxNumber);
    return _data[(i-1) * _maxNumber + (j-1)];
}

void SchurMatrix::set(int i, int j, bool b) {
    assert(i >= 1);
    assert(i <= _nbSubsets);
    assert(j >= 1);
    assert(j <= _maxNumber);
    _data[(i-1) * _maxNumber + (j-1)] = b;
}

const std::vector<bool> & SchurMatrix::data() const {
    return _data;
}

