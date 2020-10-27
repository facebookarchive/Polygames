/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "SolutionSet.h"

namespace csp {
namespace vkms {

template <size_t WIDTH, size_t HEIGHT, size_t MINES> class SolutionSetSampler {

  using _GameDefs = Minesweeper::GameDefs<WIDTH, HEIGHT, MINES>;
  using Mines = typename _GameDefs::Mines;
  using _SolutionSet = SolutionSet<WIDTH, HEIGHT, MINES>;
  using _Mask = Minesweeper::Mask<WIDTH, HEIGHT, MINES>;
  using CountSample = std::vector<size_t>;
  using CountSampleList = std::list<CountSample>;

 public:
  SolutionSetSampler(const std::vector<_SolutionSet>& solutionSets,
                     const _Mask& unconstrainedVariables,
                     const _Mask& mines)
      : _solutionSets(solutionSets)
      , _unconstrainedVariables(unconstrainedVariables)
      , _mines(mines)
      , _numMinesRemaining(MINES - mines.sparse().size())
      , _unconstrVarSetIdx(solutionSets.size())
      , _numUnconstrVars(unconstrainedVariables.sparse().size())
      , _minMines(solutionSets.size() + 1, 0)
      , _maxMines(solutionSets.size() + 1, 0)
      , _firstBranchingSet(0)
      , _solutionSetsOrder(solutionSets.size() + 1, 0)
      , _numMinesToSample(solutionSets.size() + 1, 0) {
    assert(mines.sparse().size() <= MINES);
    assert(_numMinesRemaining <= MINES);
    initializeMinMaxMinesStats();
    prepareSampling();
  }  // SolutionSetSampler::SolutionSetSampler

  std::vector<std::pair<CountSample, float>> countsWithProbabilities() const {
    std::vector<std::pair<CountSample, float>> countSamplesWithProbas;
    countSamplesWithProbas.reserve(_countSamples.size());
    auto countSamplesIt = _countSamples.begin();
    for (size_t k = 0; k < _countSamples.size(); ++k) {
      const auto& countSampleOrdered = *countSamplesIt++;
      CountSample countSample(countSampleOrdered.size());
      for (size_t i = 0; i < countSample.size(); ++i) {
        countSample[_solutionSetsOrder[i]] = countSampleOrdered[i];
      }
      countSamplesWithProbas.emplace_back(countSample, _countSampleProbas[k]);
    }
    return countSamplesWithProbas;
  }  // countsWithProbabilities

  template <typename RngEngine> void sampleMines(Mines& mines, RngEngine& rng) {
    auto mineSampleIt = mines.begin();
    // add all marked mines
    for (const auto& v : _mines.sparse()) {
      *mineSampleIt++ = rowColToIdx<WIDTH>(v.row(), v.col());
    }
    // sample counts for each solution set
    CountSample countSample;
    if (_countSamples.size() > 1) {
      MINESWEEPER_DEBUG(debug(std::cout) << "Unnormalized probas: ";
                        dumpCollection(std::cout, _countSampleProbas)
                        << std::endl);
      std::discrete_distribution<size_t> sampleDistribution(
          _countSampleProbas.begin(), _countSampleProbas.end());
      size_t countSampleIdx = sampleDistribution(rng);
      auto countSampleIt = _countSamples.begin();
      for (size_t i = 0; i < countSampleIdx; ++i, ++countSampleIt)
        ;
      countSample = *countSampleIt;
    } else {
      countSample = _countSamples.front();
    }
    MINESWEEPER_DEBUG(debug(std::cout) << "Count sample: ";
                      dumpCollection(std::cout, countSample) << std::endl);
    // sample according to counts
    size_t nMinesInSetJ, j;
    for (size_t i = 0; i < countSample.size(); ++i) {
      j = _solutionSetsOrder[i];
      nMinesInSetJ = countSample[i];
      if (!nMinesInSetJ) {
        continue;
      }
      if (j != _unconstrVarSetIdx) {
        auto mineSample = _solutionSets[j].sample(nMinesInSetJ, rng);
        mineSampleIt =
            std::copy(mineSample.begin(), mineSample.end(), mineSampleIt);
      } else {
        auto mineSample = sampleUnconstrained(nMinesInSetJ, rng);
        assert(mineSample.size() == nMinesInSetJ);
        mineSampleIt =
            std::copy(mineSample.begin(), mineSample.end(), mineSampleIt);
      }
    }
    assert(mineSampleIt == mines.end());
    std::sort(mines.begin(), mines.end());
  }  // olutionSetSampler::sampleMines

 private:
  static constexpr size_t INVALID_COUNT = static_cast<size_t>(-1);

  void prepareSampling() {
    adjustMinMaxMinesStats();
    computeSolutionSetsOrder();
    _countSamples = enumeratePlausibleCountSamples();
    assert(!_countSamples.empty());
    MINESWEEPER_DEBUG(dumpCountSamples(std::cout, _countSamples));
    _countSampleProbas = computeSampleProbabilities(_countSamples);
  }  // prepareSampling

  template <typename RngEngine>
  std::vector<int> sampleUnconstrained(size_t n, RngEngine& rng) {
    std::vector<int> sample;
    if (!n) {
      return sample;
    }
    sample.reserve(n);
    std::vector<Minesweeper::BoardPosition> unconstrained(
        _unconstrainedVariables.sparse());
    std::shuffle(unconstrained.begin(), unconstrained.end(), rng);
    for (size_t i = 0; i < n; ++i) {
      std::uniform_int_distribution<size_t> distribution(
          i, unconstrained.size() - 1);
      size_t varIdx = distribution(rng);
      const auto& boardPosition = unconstrained[varIdx];
      sample.push_back(
          rowColToIdx<WIDTH>(boardPosition.row(), boardPosition.col()));
      std::swap(unconstrained[i], unconstrained[varIdx]);
    }
    return sample;
  }

  std::vector<float> computeSampleProbabilities(
      const CountSampleList& samples) {
    std::vector<float> probas;
    probas.reserve(samples.size());
    float unnormalizedLogProba;
    float maxLogProba = 0;
    size_t nSamples, nMinesInSetJ, j;
    for (const auto& sample : samples) {
      MINESWEEPER_DEBUG(debug(std::cout)
                        << "Computing probabilities for sample counts:"
                        << std::endl);
      MINESWEEPER_DEBUG(dumpCollection(debug(std::cout), sample) << std::endl);
      assert(sample.size() == _solutionSetsOrder.size());
      unnormalizedLogProba = 0;
      for (size_t i = 0; i < sample.size(); ++i) {
        j = _solutionSetsOrder[i];
        nMinesInSetJ = sample[i];
        if (nMinesInSetJ > 0) {
          if (j != _unconstrVarSetIdx) {
            assert(_solutionSets[j].hasSamples(nMinesInSetJ));
            nSamples = _solutionSets[j].numSamples(nMinesInSetJ);
            unnormalizedLogProba += log(nSamples);
            MINESWEEPER_DEBUG(debug(std::cout)
                              << "Set " << j << " (constrained)"
                              << ", #mines " << nMinesInSetJ << ", log proba "
                              << log(nSamples) << std::endl);
          } else {
            assert(nMinesInSetJ <= _numUnconstrVars);
            MINESWEEPER_DEBUG(debug(std::cout)
                              << "Set " << j << " (unconstrained)"
                              << ", #mines " << nMinesInSetJ << ", log proba "
                              << logCnk(_numUnconstrVars, nMinesInSetJ)
                              << std::endl);
            unnormalizedLogProba += logCnk(_numUnconstrVars, nMinesInSetJ);
          }
        }
      }
      if (maxLogProba < unnormalizedLogProba) {
        maxLogProba = unnormalizedLogProba;
      }
      probas.push_back(unnormalizedLogProba);
    }
    // normalize
    float probaSum = 0;
    for (auto& proba : probas) {
      proba = exp(proba - maxLogProba);
      probaSum += proba;
    }
    MINESWEEPER_DEBUG(debug(std::cout) << "Probas: ");
    for (auto& proba : probas) {
      proba /= probaSum;
      MINESWEEPER_DEBUG(std::cout << probas << ", ");
    }
    MINESWEEPER_DEBUG(std::cout << std::endl);
    return probas;
  }  // computeSampleProbabilities

  float logCnk(size_t n, size_t k) {
    // n! / (k! (n-k)!)
    float result = 0;
    assert(n >= k);
    result += sumLog(max(k, n - k) + 1, n);
    result -= sumLog(2, min(k, n - k));
    return result;
  }

  float sumLog(size_t n1, size_t n2) {
    float result = 0;
    for (size_t n = n1; n <= n2; ++n) {
      result += log(n);
    }
    return result;
  }

  template <typename T>
  std::ostream& dumpCollection(std::ostream& os, const T& collection) {
    for (const auto& v : collection) {
      os << v << ' ';
    }
    return os;
  }  // dumpSample

  template <typename T>
  std::ostream& dumpCountSamples(std::ostream& os, const T& samples) {
    debug(os) << "Count samples:" << std::endl;
    for (const auto& sample : samples) {
      debug(os) << "  ";
      dumpCollection(os, sample) << std::endl;
    }
    return os;
  }  // dumpCountSamples

  std::list<std::vector<size_t>> enumeratePlausibleCountSamples() {
    std::list<std::vector<size_t>> samples;
    // set mine counts for sets with deterministic counts
    size_t initNumMines = 0;
    for (size_t i = 0; i < _firstBranchingSet; ++i) {
      size_t j = _solutionSetsOrder[i];
      assert(_minMines[j] == _maxMines[j]);
      _numMinesToSample[i] = _minMines[j];
      initNumMines += _minMines[j];
    }
    // finish if there are only deterministic counts
    if (_firstBranchingSet == _solutionSetsOrder.size()) {
      samples.push_back(_numMinesToSample);
      return samples;
    }
    // traverse the rest and select valid combinations
    std::vector<size_t> currentCounts;
    assert(_solutionSetsOrder.size() > _firstBranchingSet);
    const size_t currentCountsCapacity =
        _solutionSetsOrder.size() - _firstBranchingSet;
    currentCounts.reserve(currentCountsCapacity);
    size_t cursor = _firstBranchingSet;
    size_t setIdx = _solutionSetsOrder[cursor];
    size_t numMines = nextMinesCount(_minMines[setIdx], setIdx);
    assert(numMines != INVALID_COUNT);
    currentCounts.push_back(numMines);
    size_t sumNumMines = initNumMines + numMines;
    while (!currentCounts.empty()) {
      // fill with min number of mines for each set
      if (currentCounts.size() < currentCountsCapacity - 1) {
        ++cursor;
        setIdx = _solutionSetsOrder[cursor];
        numMines = nextMinesCount(_minMines[setIdx], setIdx);
        assert(numMines != INVALID_COUNT);
        currentCounts.push_back(numMines);
        sumNumMines += numMines;
        if (sumNumMines > _numMinesRemaining) {
          break;
        }
        continue;
      }
      assert(currentCounts.size() == currentCountsCapacity - 1);
      assert(cursor == _solutionSetsOrder.size() - 2);
      assert(sumNumMines ==
             initNumMines + std::accumulate(currentCounts.begin(),
                                            currentCounts.end(), 0u));
      // check if can sample the missing mines from the last set
      if (sumNumMines <= _numMinesRemaining) {
        ++cursor;
        setIdx = _solutionSetsOrder[cursor];
        numMines = _numMinesRemaining - sumNumMines;
        if (canSampleNumMinesFromSet(numMines, setIdx)) {
          currentCounts.push_back(numMines);
          std::copy(currentCounts.begin(), currentCounts.end(),
                    &_numMinesToSample[_firstBranchingSet]);
          samples.push_back(_numMinesToSample);
          currentCounts.pop_back();
        }
        // next counts configuration
        --cursor;
        setIdx = _solutionSetsOrder[cursor];
        numMines = nextMinesCount(currentCounts.back() + 1, setIdx);
      } else {
        // next counts configuration
        numMines = INVALID_COUNT;
      }
      while ((numMines == INVALID_COUNT) && !currentCounts.empty()) {
        sumNumMines -= currentCounts.back();
        currentCounts.pop_back();
        if (cursor > _firstBranchingSet) {
          --cursor;
          setIdx = _solutionSetsOrder[cursor];
          numMines = nextMinesCount(currentCounts.back() + 1, setIdx);
        } else {
          assert(currentCounts.empty());
        }
      }
      if (numMines != INVALID_COUNT) {
        sumNumMines -= currentCounts.back();
        sumNumMines += numMines;
        currentCounts.back() = numMines;
      }
    }
    return samples;
  }  // determineNumberOfSamples

  bool canSampleNumMinesFromSet(size_t numMines, size_t setIdx) {
    if (setIdx == _unconstrVarSetIdx) {
      // set of unconstrained variables
      return numMines <= _numUnconstrVars;
    } else {
      return _solutionSets[setIdx].hasSamples(numMines);
    }
  }  // canSampleNumMinesFromSet

  size_t nextMinesCount(size_t countHint, size_t setIdx) {
    assert(setIdx < _maxMines.size());
    if (countHint > _maxMines[setIdx]) {
      return INVALID_COUNT;
    }
    if (setIdx == _unconstrVarSetIdx) {
      return countHint;
    }
    while (!_solutionSets[setIdx].hasSamples(countHint)) {
      ++countHint;
      if (countHint > _maxMines[setIdx]) {
        return INVALID_COUNT;
      }
    }
    assert(_solutionSets[setIdx].hasSamples(countHint));
    return countHint;
  }  // nextMinesCount

  void computeSolutionSetsOrder() {
    assert(_solutionSetsOrder.size() > 0);
    assert(_solutionSetsOrder.size() == _minMines.size());
    assert(_solutionSetsOrder.size() == _maxMines.size());
    _firstBranchingSet = 0;
    size_t head = 0;
    size_t tail = _solutionSetsOrder.size() - 1;
    for (size_t i = 0; i < _solutionSetsOrder.size(); ++i) {
      if (_minMines[i] == _maxMines[i]) {
        _solutionSetsOrder[head++] = i;
      } else {
        _solutionSetsOrder[tail--] = i;
      }
    }
    assert(tail + 1 == head);
    // if unconstrained set does not have a fixed number of mines to sample
    // swap it with the last one
    if (_solutionSetsOrder[head] == _unconstrVarSetIdx) {
      size_t tmp = _solutionSetsOrder.back();
      _solutionSetsOrder.back() = _unconstrVarSetIdx;
      _solutionSetsOrder[head] = tmp;
    }
    _firstBranchingSet = head;
  }  // computeSolutionSetsOrder

  void initializeMinMaxMinesStats() {
    for (size_t i = 0; i < _solutionSets.size(); ++i) {
      _minMines[i] = _solutionSets[i].minNumMines();
      _maxMines[i] = _solutionSets[i].maxNumMines();
    }
    _minMines[_solutionSets.size()] = 0;
    _maxMines[_solutionSets.size()] = _unconstrainedVariables.sparse().size();
    MINESWEEPER_DEBUG(
        debug(std::cout) << "Min mines init: "; for (size_t n
                                                     : _minMines) {
          std::cout << n << " ";
        } std::cout << std::endl;);
    MINESWEEPER_DEBUG(
        debug(std::cout) << "Max mines init: "; for (size_t n
                                                     : _maxMines) {
          std::cout << n << " ";
        } std::cout << std::endl;);
  }  // SolutionSetSampler::initializeMinMaxMinesStats

  void adjustMinMaxMinesStats() {
    size_t sumMax = std::accumulate(_maxMines.begin(), _maxMines.end(), 0u);
    size_t sumMin = std::accumulate(_minMines.begin(), _minMines.end(), 0u);
    const size_t nsets = _maxMines.size();
    size_t delta;
    bool changed;
    do {
      changed = false;
      for (size_t i = 0; i < nsets; ++i) {
        if (_maxMines[i] + sumMin > _numMinesRemaining + _minMines[i]) {
          delta = _maxMines[i] + sumMin - (_numMinesRemaining + _minMines[i]);
          _maxMines[i] -= delta;
          sumMax -= delta;
          changed = true;
        }
        if (_minMines[i] + sumMax < _numMinesRemaining + _maxMines[i]) {
          delta = _numMinesRemaining + _maxMines[i] - (_minMines[i] + sumMax);
          _minMines[i] += delta;
          sumMin += delta;
          changed = true;
        }
      }
    } while (changed);
    MINESWEEPER_DEBUG(debug(std::cout) << "Min mines adj: "; for (size_t n
                                                                  : _minMines) {
      std::cout << n << " ";
    } std::cout << std::endl;);
    MINESWEEPER_DEBUG(debug(std::cout) << "Max mines adj: "; for (size_t n
                                                                  : _maxMines) {
      std::cout << n << " ";
    } std::cout << std::endl;);
  }  // SolutionSetSampler::adjustMinMaxMinesStats

  const std::vector<_SolutionSet>& _solutionSets;
  const _Mask& _unconstrainedVariables;
  const _Mask& _mines;
  const size_t _numMinesRemaining;
  const size_t _unconstrVarSetIdx;
  const size_t _numUnconstrVars;

  std::vector<size_t> _minMines;
  std::vector<size_t> _maxMines;
  size_t _firstBranchingSet;
  std::vector<size_t> _solutionSetsOrder;
  std::vector<size_t> _numMinesToSample;

  CountSampleList _countSamples;
  std::vector<float> _countSampleProbas;

};  // class SolutionSetSampler

}  // namespace vkms
}  // namespace csp
