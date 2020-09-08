/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SchurVector.hpp"

#include <cassert>

SchurVector::SchurVector(int maxIndex) :
    _maxIndex(maxIndex),
    _data(maxIndex) {
}

void SchurVector::reset(int value) {
    std::fill(_data.begin(), _data.end(), value);
}

int SchurVector::get(int i) const {
    assert(i >= 1);
    assert(i <= _maxIndex);
    return _data[i-1];
}

void SchurVector::set(int i, int value) {
    assert(i >= 1);
    assert(i <= _maxIndex);
    _data[i-1] = value;
}

const std::vector<int> & SchurVector::data() const {
    return _data;
}

