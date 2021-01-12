/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "WeakSchur.hpp"

#include <iostream>
#include <cassert>

WeakSchur::WeakSchur(int nbSubsets, int maxNumber) :
    _nbSubsets(nbSubsets),
    _maxNumber(maxNumber),
    _freeActions(nbSubsets, maxNumber),
    _nbFreeNumbersOfSubset(nbSubsets),
    _nbNumbersOfSubset(nbSubsets),
    _nbFreeSubsetsOfNumber(maxNumber),
    _subsetOfNumber(maxNumber)
{
    reset();
}

void WeakSchur::reset() {
    _freeActions.reset(true);
    _nbFreeActions = _nbSubsets * _maxNumber;
    _nbFreeNumbersOfSubset.reset(_maxNumber);
    _nbNumbersOfSubset.reset(0);
    _nbFreeSubsetsOfNumber.reset(_nbSubsets);
    _subsetOfNumber.reset(0);
    _score = 0;
    applyAction({1, 1});
}

bool WeakSchur::isTerminated() const {
    return _score == _maxNumber or _nbFreeSubsetsOfNumber.get(_score+1) == 0;
}

int WeakSchur::getScore() const {
    return _score;
}

int WeakSchur::getFirstLegalNumber() const {
    assert(not isTerminated());
    return _score + 1;
}

int WeakSchur::getMostConstrainedNumber() const {
    assert(not isTerminated());
    int bestNumber = _score + 1;
    int nbSubsetsBest = _nbFreeSubsetsOfNumber.get(bestNumber);
    for (int number = bestNumber + 1; number <  _maxNumber; number++) {
        const int nbSubsets = _nbFreeSubsetsOfNumber.get(number);
        if (nbSubsets != 0 and nbSubsets < nbSubsetsBest) {
            nbSubsetsBest = nbSubsets;
            bestNumber = number;
        }
    }
    return bestNumber;
}

std::vector<int> WeakSchur::getLegalSubsets(int number) const {
    const int nbSubsetsOfNumber = _nbFreeSubsetsOfNumber.get(number);
    std::vector<int> legalSubsets;
    legalSubsets.reserve(nbSubsetsOfNumber);
    for (int i = 1; i <= _nbSubsets; i++)
        //if ((_freeActions.get(i, number) and i<=2) or 
	//	(_freeActions.get(i, number) and i>2 and _nbNumbersOfSubset.get(i-1) > 0))
        if (_freeActions.get(i, number) and (i<=2 or _nbNumbersOfSubset.get(i-1) > 0))
            legalSubsets.push_back(i);
    //assert(int(legalSubsets.size()) == nbSubsetsOfNumber);
    // TODO
    return legalSubsets;
}

std::pair<int, int> WeakSchur::getLongestSeq(int subset) const {
    int longest = 0;
    int nbLongest = 0;
    int currLongest = 0;
    for (int n=1; n<=_maxNumber; n++) {
        if (_subsetOfNumber.get(n) == subset) {
            currLongest++;
        }
        else {
            if (currLongest > longest) {
                longest = currLongest;
                nbLongest = 1;
            }
            else if (currLongest == longest) {
                nbLongest++;
            }
            currLongest = 0;
        }
    }
    return {longest, nbLongest};
}

void WeakSchur::applyAction(const Action & action) {

    const int subset = action.first;
    const int number = action.second;

    // assert action
    assert(subset >= 1);
    assert(subset <= _nbSubsets);	
    assert(number >= 1);
    assert(number <= _maxNumber);	
    assert(_nbFreeSubsetsOfNumber.get(number) > 0);
    assert(_subsetOfNumber.get(number) == 0);
    assert(_freeActions.get(subset, number));

    // update subset data
    for (int s = 1; s <= _nbSubsets; s++)
        _freeActions.set(s, number, false);
    _nbFreeActions -= _nbFreeSubsetsOfNumber.get(number);
    _nbFreeSubsetsOfNumber.set(number, 0);
    _nbNumbersOfSubset.set(subset, _nbNumbersOfSubset.get(subset) + 1);

    // update number data
    for (int n = 1; n <= _maxNumber; n++) {
        if (subset == _subsetOfNumber.get(n)) {
            removeAction({subset, n + number});
            removeAction({subset, n - number});
            removeAction({subset, number - n});

            _nbFreeNumbersOfSubset.set(subset, 
                    _nbFreeNumbersOfSubset.get(subset)-1);
        }
    }

    // store action
    _subsetOfNumber.set(number, subset);

    // update score
    while (_score < _maxNumber and _subsetOfNumber.get(_score+1) != 0)
        _score++;

}

void WeakSchur::removeAction(const Action & action) {
    const int subset = action.first;
    const int numberToRemove = action.second;
    if (numberToRemove >= 1 and numberToRemove <= _maxNumber
            and _freeActions.get(subset, numberToRemove)) {
        _freeActions.set(subset, numberToRemove, false);
        _nbFreeActions--;
        const int oldNbSubsets = _nbFreeSubsetsOfNumber.get(numberToRemove);
        _nbFreeSubsetsOfNumber.set(numberToRemove, oldNbSubsets - 1);
    }
}

std::ostream & operator<<(std::ostream & os, const WeakSchur & ws) {

    os << "freeActions: " << std::endl;
    for (int s = 1; s <= ws._nbSubsets; s++) {
        os << " ";
        for (int n = 1; n <= ws._maxNumber; n++) {
            os << " " << ws._freeActions.get(s, n);
        }
        os << std::endl;
    }

    os << "nbFreeActions: \n  " << ws._nbFreeActions << std::endl;

    os << "nbSubsetsOfNumber:\n ";
    for (int n = 1; n <= ws._maxNumber; n++) {
        os << " " << ws._nbFreeSubsetsOfNumber.get(n);
    }
    os << std::endl;

    os << "subsetOfNumber:\n ";
    for (int n = 1; n <= ws._maxNumber; n++) {
        os << " " << ws._subsetOfNumber.get(n);
    }
    os << std::endl;

    os << "score: \n  " << ws._score << std::endl;

    os << "subsets: " << std::endl;
    for (int s = 1; s <= ws._nbSubsets; s++) {
        os << " ";
        for (int n = 1; n <= ws._maxNumber; n++) {
            if (ws._subsetOfNumber.get(n) == s)
                os << " " << n;
        }
        os << std::endl;
    }
    os << std::endl;

    return os;
}

