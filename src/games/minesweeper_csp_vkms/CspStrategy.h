/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "SolutionSetSampler.h"
#include <chrono>
#include <thread>

namespace csp {
namespace vkms {

template <size_t WIDTH, size_t HEIGHT, size_t MINES> class CspStrategy {

  using _GameDefs = Minesweeper::GameDefs<WIDTH, HEIGHT, MINES>;
  using Board = typename _GameDefs::Board;
  using BoardMask = typename _GameDefs::BoardMask;
  using Mines = typename _GameDefs::Mines;
  using _Mask = Minesweeper::Mask<WIDTH, HEIGHT, MINES>;
  using _SolutionSet = SolutionSet<WIDTH, HEIGHT, MINES>;
  using _SolutionSetSampler = SolutionSetSampler<WIDTH, HEIGHT, MINES>;

 public:
  using MineProbas = std::array<float, HEIGHT * WIDTH>;

  CspStrategy()
      : _minesMask(MINES)
      , _notMinesMask(WIDTH * HEIGHT)
      , _activeConstraints(WIDTH * HEIGHT)
      , _unconstrainedVariables(WIDTH * HEIGHT)
      , _processed(WIDTH * HEIGHT) {
  }  // CspStrategy

  template <typename RngEngine>
  void sampleMines(Mines& mines, const Board& board, RngEngine& rng) {
    initializeMinesMasks(board);
    initializeActiveConstraints(board);
    initializeUnconstrainedVariables(board);
    MINESWEEPER_DEBUG(dumpMasks(std::cout));
    std::vector<ConnectedComponent> connectedActiveConstraints =
        connectedConstraints(board);
    MINESWEEPER_DEBUG(dumpConstraints(std::cout, connectedActiveConstraints));
    std::vector<_SolutionSet> solutionSets;
    solutionSets.reserve(connectedActiveConstraints.size());
    for (const ConnectedComponent& component : connectedActiveConstraints) {
      solutionSets.emplace_back(component, board, _minesMask);
    }
    // sampleFromSolutionSets(solutionSets, mines, rng);
    _SolutionSetSampler sampler(
        solutionSets, _unconstrainedVariables, _minesMask);
    // computeMineProbabilities(solutionSets, sampler);
    sampler.sampleMines(mines, rng);
    std::sort(mines.begin(), mines.end());
  }  // sampleMines

  std::vector<int> locateForSureMines(const Board& board) {
    initializeMinesMasks(board);
    const auto& minePositions = _minesMask.sparse();
    std::vector<int> mineIndices;
    mineIndices.reserve(minePositions.size());
    for (const auto& minePosition : minePositions) {
      mineIndices.push_back(
          rowColToIdx<WIDTH>(minePosition.row(), minePosition.col()));
    }
    std::sort(mineIndices.begin(), mineIndices.end());
    return mineIndices;
  }  // getForSureMines

  void computeMineProbabilities(const Board& board) {
    initializeMinesMasks(board);
    initializeActiveConstraints(board);
    initializeUnconstrainedVariables(board);
    std::vector<ConnectedComponent> connectedActiveConstraints =
        connectedConstraints(board);
    std::vector<_SolutionSet> solutionSets;
    solutionSets.reserve(connectedActiveConstraints.size());
    for (const ConnectedComponent& component : connectedActiveConstraints) {
      solutionSets.emplace_back(component, board, _minesMask);
    }
    _SolutionSetSampler sampler(
        solutionSets, _unconstrainedVariables, _minesMask);
    computeMineProbabilities(solutionSets, sampler);
  }  // computeMineProbabilities

  template <typename RngEngine>
  void computeMineProbabilitiesAndSampleMines(Mines& mines,
                                              const Board& board,
                                              RngEngine& rng) {
    initializeMinesMasks(board);
    initializeActiveConstraints(board);
    initializeUnconstrainedVariables(board);
    std::vector<ConnectedComponent> connectedActiveConstraints =
        connectedConstraints(board);
    std::vector<_SolutionSet> solutionSets;
    solutionSets.reserve(connectedActiveConstraints.size());
    for (const ConnectedComponent& component : connectedActiveConstraints) {
      solutionSets.emplace_back(component, board, _minesMask);
    }
    _SolutionSetSampler sampler(
        solutionSets, _unconstrainedVariables, _minesMask);
    computeMineProbabilities(solutionSets, sampler);
    sampler.sampleMines(mines, rng);
    std::sort(mines.begin(), mines.end());
  }  // computeMineProbabilities

  const MineProbas& getMineProbabilities() const {
    return _mineProbas;
  }  // getMineProbabilities

 private:
  void computeMineProbabilities(const std::vector<_SolutionSet>& solutionSets,
                                const _SolutionSetSampler& solutionSetSampler) {
    memset(_mineProbas.data(), 0,
           WIDTH * HEIGHT * sizeof(typename MineProbas::value_type));
    // mines are 100%
    for (const auto& pos : _minesMask.sparse()) {
      arrGet<MineProbas, WIDTH>(_mineProbas, pos.row(), pos.col()) = 1.0;
    }
    // not mines are 0%
    for (const auto& pos : _notMinesMask.sparse()) {
      arrGet<MineProbas, WIDTH>(_mineProbas, pos.row(), pos.col()) = 0.0;
    }
    auto countsWithProbas = solutionSetSampler.countsWithProbabilities();
    for (const auto& countsWithProba : countsWithProbas) {
      const auto& counts = countsWithProba.first;
      auto proba = countsWithProba.second;
      MINESWEEPER_DEBUG(debug(std::cout) << "Counts: ");
      MINESWEEPER_DEBUG(for (auto n : counts) { std::cout << n << " "; });
      MINESWEEPER_DEBUG(std::cout << ", weight=" << proba << std::endl);
      assert(counts.size() == solutionSets.size() + 1);
      for (size_t j = 0; j < solutionSets.size(); ++j) {
        MINESWEEPER_DEBUG(debug(std::cout)
                          << "Solution set " << j << ": " << counts[j]
                          << " mines" << std::endl);
        auto count = counts[j];
        if (!count) {
          continue;
        }
        const auto& vars = solutionSets[j].getVariables();
        const auto& varProbas = solutionSets[j].getVarProbas(count);
        MINESWEEPER_DEBUG(debug(std::cout) << "Variable probabilities: ");
        MINESWEEPER_DEBUG(for (auto p : varProbas) { std::cout << p << " "; });
        MINESWEEPER_DEBUG(std::cout << std::endl);
        assert(vars.size() == varProbas.size());
        for (size_t i = 0; i < vars.size(); ++i) {
          arrGet<MineProbas, WIDTH>(_mineProbas, vars[i].row(),
                                    vars[i].col()) += proba * varProbas[i];
        }
      }
      auto count = counts[solutionSets.size()];
      if (!count) {
        continue;
      }
      MINESWEEPER_DEBUG(debug(std::cout) << "Unconstrained solution set: ");
      MINESWEEPER_DEBUG(std::cout << _unconstrainedVariables.sparse().size());
      MINESWEEPER_DEBUG(std::cout << " variables, " << count << " mines, ");
      MINESWEEPER_DEBUG(std::cout << "proba="
                                  << getUnconstrainedVarProba(count));
      MINESWEEPER_DEBUG(std::cout << std::endl);
      for (const auto& pos : _unconstrainedVariables.sparse()) {
        arrGet<MineProbas, WIDTH>(_mineProbas, pos.row(), pos.col()) +=
            proba * getUnconstrainedVarProba(count);
      }
    }
  }  // computeMineProbabilities

  float getUnconstrainedVarProba(size_t count) const {
    assert(count > 0);
    size_t nUnconstr = _unconstrainedVariables.sparse().size();
    assert(nUnconstr >= count);
    if (count == 1) {
      return 1.0f / nUnconstr;
    }
    return static_cast<float>(count) /
           (static_cast<float>(nUnconstr) * (nUnconstr - count + 1));
  }  // getUnconstrainedVarProba

  std::ostream& dumpMasks(std::ostream& os) {
    debug(os) << "Mines mask:" << std::endl
              << _GameDefs::boardMaskToString(_minesMask.dense());
    debug(os) << "Not mines mask:" << std::endl
              << _GameDefs::boardMaskToString(_notMinesMask.dense());
    debug(os) << "Active constraints mask:" << std::endl
              << _GameDefs::boardMaskToString(_activeConstraints.dense());
    debug(os) << "Unconstrained variables mask:" << std::endl
              << _GameDefs::boardMaskToString(_unconstrainedVariables.dense());
    return os;
  }  // dumpMasks

  template <typename T>
  std::ostream& dumpConstraints(std::ostream& os, const T& constraints) {
    debug(std::cout) << "Active constraints:" << std::endl;
    size_t i = 0;
    for (const auto& component : constraints) {
      debug(std::cout) << "Component " << i++ << std::endl;
      debug(std::cout) << "Constraints: "
                       << sparseMaskToString(component._constraints)
                       << std::endl;
      debug(std::cout) << "Variables: "
                       << sparseMaskToString(component._variables) << std::endl;
    };
    return os;
  }  // dumpActiveConstraints

  std::vector<ConnectedComponent> connectedConstraints(const Board& board) {
    std::vector<ConnectedComponent> connectedComponents;
    const Minesweeper::SparseMask& constraintsSparse =
        _activeConstraints.sparse();
    connectedComponents.reserve(constraintsSparse.size());
    BoardMask processed;
    memset(processed.data(), 0,
           WIDTH * HEIGHT * sizeof(typename BoardMask::value_type));
    for (const Minesweeper::BoardPosition& bp : constraintsSparse) {
      if (arrGet<BoardMask, WIDTH>(processed, bp.row(), bp.col())) {
        continue;
      }
      ConnectedComponent component =
          connectedConstraintsFromSeed(board, bp.row(), bp.col(), processed);
      connectedComponents.push_back(component);
    }
    return connectedComponents;
  }  // connectedComponents

  ConnectedComponent connectedConstraintsFromSeed(const Board& board,
                                                  int row,
                                                  int col,
                                                  BoardMask& processed) {
    // collect together active constraints connected through variables
    const Minesweeper::SparseMask& constraintsSparse =
        _activeConstraints.sparse();
    ConnectedComponent component;
    component._constraints.reserve(constraintsSparse.size());
    component._variables.reserve(constraintsSparse.size());
    auto select_unprocessed_variables = [this, &processed](
                                            int v, int row, int col) {
      return (v == Minesweeper::UNKNOWN) &&
             !this->_notMinesMask.get(row, col) &&
             !this->_minesMask.get(row, col) &&
             !arrGet<BoardMask, WIDTH>(processed, row, col);
    };
    auto select_unprocessed_constraints = [this, &processed](
                                              int UNUSED(v), int row, int col) {
      return this->_activeConstraints.get(row, col) &&
             !arrGet<BoardMask, WIDTH>(processed, row, col);
    };
    std::vector<Minesweeper::BoardPosition> currentVariables =
        _GameDefs::getNeighbors(board, row, col, select_unprocessed_variables);
    // add variables to the queue
    std::list<Minesweeper::BoardPosition> var_queue(
        currentVariables.begin(), currentVariables.end());
    // mark variables as processed
    for (const Minesweeper::BoardPosition& bp : currentVariables) {
      arrGet<BoardMask, WIDTH>(processed, bp.row(), bp.col()) = 1;
    }
    // add constraint to the connected component
    component._constraints.emplace_back(row, col);
    // mark constraint as processed
    arrGet<BoardMask, WIDTH>(processed, row, col) = 1;
    // process the queue of variables
    while (!var_queue.empty()) {
      Minesweeper::BoardPosition currentVar = var_queue.front();
      component._variables.push_back(currentVar);
      var_queue.pop_front();
      std::vector<Minesweeper::BoardPosition> currentConstraints =
          _GameDefs::getNeighbors(board, currentVar.row(), currentVar.col(),
                                  select_unprocessed_constraints);
      for (const auto& currentConstraint : currentConstraints) {
        currentVariables = _GameDefs::getNeighbors(
            board, currentConstraint.row(), currentConstraint.col(),
            select_unprocessed_variables);
        // add variables to the queue
        var_queue.insert(
            var_queue.end(), currentVariables.begin(), currentVariables.end());
        // mark variables as processed
        for (const auto& var : currentVariables) {
          arrGet<BoardMask, WIDTH>(processed, var.row(), var.col()) = 1;
        }
        // add constraint to the connected component
        component._constraints.emplace_back(
            currentConstraint.row(), currentConstraint.col());
        // mark constraint as processed
        arrGet<BoardMask, WIDTH>(
            processed, currentConstraint.row(), currentConstraint.col()) = 1;
      }
    }
    return component;
  }  // connectedComponentFromSeed

  void initializeMinesMasks(const Board& board) {
    _notMinesMask.zero();
    _minesMask.zero();
    _processed.zero();
    auto select_all = [](int UNUSED(v), int UNUSED(row), int UNUSED(col)) {
      return true;
    };
    auto select_potential_mines = [=](int v, int row, int col) {
      return (v == Minesweeper::UNKNOWN) && !_notMinesMask.get(row, col) &&
             !_minesMask.get(row, col);
    };
    auto select_not_marked_mines = [=](int UNUSED(v), int row, int col) {
      return !_minesMask.get(row, col);
    };
    auto select_marked_mines = [=](int UNUSED(v), int row, int col) {
      return _minesMask.get(row, col);
    };
    int v;
    bool notchanged;
    do {
      notchanged = true;
      for (size_t row = 0; row < HEIGHT; ++row) {
        for (size_t col = 0; col < WIDTH; ++col) {
          if (_processed.get(row, col)) {
            continue;
          }
          v = arrGet<Board, WIDTH>(board, row, col);
          switch (v) {
          case Minesweeper::UNKNOWN:
            _processed.set(row, col);
            break;
          case Minesweeper::BOOM:
            _minesMask.set(row, col);
            _processed.set(row, col);
            notchanged = false;
            break;
          case 0:
            // none of the neighbors are mines
            _GameDefs::markNeighbors(
                board, row, col, _notMinesMask, select_all);
            _notMinesMask.set(row, col);
            _processed.set(row, col);
            notchanged = false;
            break;
          default:
            assert(v > 0);
            size_t num_mines_total = static_cast<size_t>(v);
            size_t num_mines =
                _GameDefs::countNeighbors(board, row, col, select_marked_mines);
            size_t num_potential_mines = _GameDefs::countNeighbors(
                board, row, col, select_potential_mines);
            assert(num_mines <= num_mines_total);
            assert(num_potential_mines + num_mines >= num_mines_total);
            if (num_mines == num_mines_total) {
              // the rest are not mines
              _GameDefs::markNeighbors(
                  board, row, col, _notMinesMask, select_not_marked_mines);
              _processed.set(row, col);
              notchanged = false;
            } else if (num_potential_mines + num_mines == num_mines_total) {
              // all candidates are mines
              _GameDefs::markNeighbors(
                  board, row, col, _minesMask, select_potential_mines);
              // the rest are not mines
              _GameDefs::markNeighbors(
                  board, row, col, _notMinesMask, select_not_marked_mines);
              _processed.set(row, col);
              notchanged = false;
            }
            if (!_notMinesMask.get(row, col)) {
              _notMinesMask.set(row, col);
              notchanged = false;
            }
          }  // switch(v)
        }    // for col
      }      // for row
    } while (!notchanged);
  }  // initializeMinesMasks

  void initializeActiveConstraints(const Board& board) {
    _activeConstraints.zero();
    auto select_potential_mines = [=](int v, int row, int col) {
      return (v == Minesweeper::UNKNOWN) && !_notMinesMask.get(row, col) &&
             !_minesMask.get(row, col);
    };
    auto select_mines = [=](int UNUSED(v), int row, int col) {
      return _minesMask.get(row, col);
    };
    typename Board::value_type v;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        v = arrGet<Board, WIDTH>(board, row, col);
        if (v > 0) {
          size_t num_mines_total = static_cast<size_t>(v);
          size_t num_mines =
              _GameDefs::countNeighbors(board, row, col, select_mines);
          size_t num_potential_mines = _GameDefs::countNeighbors(
              board, row, col, select_potential_mines);
          assert(num_mines <= num_mines_total);
          assert(num_potential_mines + num_mines >= num_mines_total);
          if (num_potential_mines + num_mines > num_mines_total) {
            // has unknown neighbors, not all of them are mines
            _activeConstraints.set(row, col);
          }
        }
      }  // for col
    }    // for row
  }      // initializeActiveConstraints

  void initializeUnconstrainedVariables(const Board& board) {
    _unconstrainedVariables.zero();
    auto select_active_constraints = [this](int UNUSED(v), int row, int col) {
      return this->_activeConstraints.get(row, col);
    };
    typename Board::value_type v;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        v = arrGet<Board, WIDTH>(board, row, col);
        if ((v == Minesweeper::UNKNOWN) && !_minesMask.get(row, col) &&
            !_notMinesMask.get(row, col) &&
            !_GameDefs::countNeighbors(
                board, row, col, select_active_constraints)) {
          _unconstrainedVariables.set(row, col);
        }
      }  // for col
    }    // for row
  }      // initializeUnconstrainedVariables

  _Mask _minesMask;
  _Mask _notMinesMask;
  _Mask _activeConstraints;
  _Mask _unconstrainedVariables;
  _Mask _processed;
  MineProbas _mineProbas;

};  // class CspStrategy

}  // namespace vkms
}  // namespace csp
