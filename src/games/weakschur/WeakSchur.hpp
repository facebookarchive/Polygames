/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "SchurMatrix.hpp"
#include "SchurVector.hpp"

#include <iostream>

class WeakSchur {

    // Action = (subset, number)
    using Action = std::pair<int, int>;

    public:
        const int _nbSubsets;
        const int _maxNumber;

    protected:
    public:  // todo getters ?
        SchurMatrix _freeActions;

        int _nbFreeActions;

        // how many possible numbers, for each subset
        SchurVector _nbFreeNumbersOfSubset; // TODO
        SchurVector _nbNumbersOfSubset;

        // how many possible subsets, for each number
        SchurVector _nbFreeSubsetsOfNumber;

        // vector storing, for each number, the selected subset
        // (or 0 where no subset has been selected)
        SchurVector _subsetOfNumber;

        // the current score is the max of the successive numbers validly
        // placed in the subsets, i.e. n where 1, ..., n are validly placed
        // in the subsets and (n+1) is not
        int _score;

    public:
        WeakSchur(int nbSubsets, int maxNumber);
        void reset();

        void applyAction(const Action & action);
        bool isTerminated() const;
        int getScore() const;

        int getFirstLegalNumber() const;
        int getMostConstrainedNumber() const;
        std::vector<int> getLegalSubsets(int number) const;

        std::pair<int, int> getLongestSeq(int subset) const;

        friend std::ostream & operator<<(std::ostream & os, const WeakSchur & ws);

    protected:
        void removeAction(const Action & action);

};

std::ostream & operator<<(std::ostream & os, const WeakSchur & ws);

