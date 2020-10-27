/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ConnectedComponent.h"

namespace csp {
namespace vkms {

#define debug Minesweeper::debug
#define rowColToIdx Minesweeper::rowColToIdx
#define arrGet Minesweeper::arrGet

template <size_t WIDTH, size_t HEIGHT, size_t MINES> class SolutionSet {

  using _GameDefs = Minesweeper::GameDefs<WIDTH, HEIGHT, MINES>;
  using Board = typename _GameDefs::Board;
  using _Mask = Minesweeper::Mask<WIDTH, HEIGHT, MINES>;
  using IdxSizePair = std::pair<size_t, size_t>;
  using IdxSizePairs = std::vector<IdxSizePair>;
  using Indices = std::list<size_t>;
  using Solution = std::list<size_t>;
  using Solutions = std::list<Solution>;

 public:
  SolutionSet(const ConnectedComponent& cc,
              const Board& board,
              const _Mask& mines)
      : _variables(cc._variables)
      , _constraints(cc._constraints)
      , _varToConstr(_variables.size())
      , _constrToVar(_constraints.size())
      , _minNumMines(MINES)
      , _maxNumMines(0)
      , _varStates(_variables.size(), -1) {
    initializeVariableIdxMap();
    initializeMaps();
    _constrCounts = computeConstraintNminesLeft(board, mines);
    _constrStates = _constrCounts;
    _varOrder = variablesOrderedByNconstraintsDescending();
    enumerateSolutions(board, mines);
    MINESWEEPER_DEBUG(dumpSolutionStats(debug(std::cout)) << std::endl);
  }  // SolutionSet

  size_t minNumMines() const {
    return _minNumMines;
  }  // min_num_mines

  size_t maxNumMines() const {
    return _maxNumMines;
  }  // max_num_mines

  bool hasSamples(size_t nmines) const {
    return _solutions.find(nmines) != _solutions.end();
  }  // hasSamples

  size_t numSamples(size_t nmines) const {
    return _solutions.at(nmines).size();
  }  // hasSamples

  const Minesweeper::SparseMask& getVariables() const {
    return _variables;
  }  // getVariables

  std::vector<float> getVarProbas(size_t nmines) const {
    std::vector<float> probas(_variables.size(), 0);
    if (!hasSamples(nmines)) {
      return probas;
    }
    const auto& solutions = _solutions.at(nmines);
    for (const auto& solution : solutions) {
      for (auto mineIdx : solution) {
        probas[_mineIdxToVarIdx.at(mineIdx)] += 1;
      }
    }
    for (auto& proba : probas) {
      proba /= solutions.size();
    }
    return probas;
  }  // getVarProbas

  template <typename RngEngine>
  std::vector<int> sample(size_t nmines, RngEngine& rng) const {
    std::vector<int> sample;
    if (!nmines) {
      return sample;
    }
    sample.reserve(nmines);
    assert(hasSamples(nmines));
    const auto& solutions = _solutions.at(nmines);
    std::uniform_int_distribution<size_t> distribution(0, solutions.size() - 1);
    size_t sampleIdx = distribution(rng);
    auto solutionsIt = solutions.begin();
    for (size_t i = 0; i < sampleIdx; ++i, ++solutionsIt)
      ;
    const auto& solution = *solutionsIt;
    assert(solution.size() == nmines);
    for (auto mineIdx : solution) {
      sample.push_back(mineIdx);
    }
    return sample;
  }  // sample

 private:
  void initializeVariableIdxMap() {
    int mineIdx;
    for (size_t i = 0; i < _variables.size(); ++i) {
      mineIdx = rowColToIdx<WIDTH>(_variables[i].row(), _variables[i].col());
      assert(mineIdx >= 0);
      _mineIdxToVarIdx[static_cast<size_t>(mineIdx)] = i;
    }
  }  // initializeVariableIdxMap

  void initializeMaps() {
    for (size_t i = 0; i < _variables.size(); ++i) {
      Minesweeper::BoardPosition bp_i = _variables[i];
      for (size_t j = 0; j < _constraints.size(); ++j) {
        Minesweeper::BoardPosition bp_j = _constraints[j];
        if ((bp_i.col() - bp_j.col() >= -1) && (bp_i.col() - bp_j.col() <= 1) &&
            (bp_i.row() - bp_j.row() >= -1) && (bp_i.row() - bp_j.row() <= 1)) {
          _varToConstr[i].push_back(j);
          _constrToVar[j].push_back(i);
        }
      }  // for j
    }    // for i
  }      // initializeMaps

  std::vector<size_t> computeConstraintNminesLeft(const Board& board,
                                                  const _Mask& mines) {
    std::vector<size_t> counts(_constrToVar.size(), 0);
    auto select_mines = [&](int UNUSED(v), int row, int col) {
      return mines.get(row, col);
    };
    int v, row, col;
    size_t count, nmines;
    for (size_t j = 0; j < _constrToVar.size(); ++j) {
      row = _constraints[j].row();
      col = _constraints[j].col();
      v = arrGet<Board, WIDTH>(board, row, col);
      assert(v > 0);
      count = static_cast<size_t>(v);
      nmines = _GameDefs::countNeighbors(board, row, col, select_mines);
      assert(count > nmines);
      counts[j] = count - nmines;
    }
    return counts;
  }  // computeConstraintNminesLeft

  std::vector<size_t> variablesOrderedByNconstraintsDescending() {
    std::vector<size_t> order(_varToConstr.size());
    for (size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [this](size_t i, size_t j) {
      return this->_varToConstr[i].size() > this->_varToConstr[j].size();
    });
    return order;
  }  // variablesOrderedByNconstraintsDescending

  void assignMine(size_t i) {
    _varStates[i] = 1;
    for (size_t j : _varToConstr[i]) {
      if (_constrStates[j]) {
        // IMPORTANT: this conditional constraint state decrease may lead to
        // inconsistent states, where some cells have more neighbor mines
        // than declared; consistency check is required after updates
        _constrStates[j]--;
      }
    }
  }  // assignMine

  void assignNotMine(size_t i) {
    if (_varStates[i] == 1) {
      for (size_t j : _varToConstr[i]) {
        _constrStates[j]++;
      }
    }
    _varStates[i] = 0;
  }  // assignNotMine

  int nextUnassignedVariable() const {
    for (size_t i = 0; i < _varOrder.size(); ++i) {
      if (_varStates[_varOrder[i]] == -1) {
        return static_cast<int>(_varOrder[i]);
      }
    }
    return -1;
  }  // nextUnassignedVariable

  bool constraintsSatisfied() const {
    for (size_t constrState : _constrStates) {
      if (constrState) {
        return false;
      }
    }
    return true;
  }  // constraintsSatisfied

  bool updateStates() {
    bool changed = false;
    for (size_t j = 0; j < _constrStates.size(); ++j) {
      if (!_constrStates[j]) {
        // no more mines for this constraint, mark all free variables as
        // not mines
        for (size_t i : _constrToVar[j]) {
          if (_varStates[i] == -1) {
            assignNotMine(i);
            changed = true;
          }
        }
      } else {
        // check if the number of free variables is equal to the number of
        // mines left; assign all free variables to mines, if true
        size_t freeVars = 0;
        for (size_t i : _constrToVar[j]) {
          if (_varStates[i] == -1) {
            ++freeVars;
          }
        }
        if (freeVars == _constrStates[j]) {
          for (size_t i : _constrToVar[j]) {
            if (_varStates[i] == -1) {
              assignMine(i);
            }
          }
          changed = true;
        }
      }
    }  // for j
    return changed;
  }  // updateConstrStates

  void updateFromAssignments() {
    std::copy(
        _constrCounts.begin(), _constrCounts.end(), _constrStates.begin());
    std::vector<int> varStates(_varStates);
    std::fill(_varStates.begin(), _varStates.end(), -1);
    for (size_t v : _assignedVars) {
      assert((varStates[v] == 1) || (varStates[v] == 0));
      if (varStates[v]) {
        assignMine(v);
      } else {
        assignNotMine(v);
      }
    }
    bool changed;
    do {
      changed = updateStates();
    } while (changed);
  }  // updateConstraint

  void assignVariable(size_t i) {
    _assignedVars.push_back(i);
    _varStates[i] = 1;
    updateFromAssignments();
  }  // assignVariable

  void triggerLastAssignment() {
    if (_assignedVars.empty()) {
      return;
    }
    size_t lastAssigned = _assignedVars.back();
    while (!_varStates[lastAssigned]) {
      _assignedVars.pop_back();
      if (_assignedVars.empty()) {
        return;
      }
      lastAssigned = _assignedVars.back();
    }
    assert(_varStates[lastAssigned] == 1);
    _varStates[lastAssigned] = 0;
    updateFromAssignments();
  }  // triggerLastAssignment

  void enumerateSolution() {
    Solution mines;
    for (size_t i = 0; i < _varStates.size(); ++i) {
      if (_varStates[i] == 1) {
        int idx = rowColToIdx<WIDTH>(_variables[i].row(), _variables[i].col());
        assert(idx >= 0);
        mines.push_back(static_cast<size_t>(idx));
      }
    }
    if (_minNumMines > mines.size()) {
      _minNumMines = mines.size();
    }
    if (_maxNumMines < mines.size()) {
      _maxNumMines = mines.size();
    }
    _solutions[mines.size()].push_back(mines);
  }  // enumerateSolution

  bool checkSolutionAgainstBoard(const Board& board, const _Mask& mines) {
    for (size_t i = 0; i < _constraints.size(); ++i) {
      const auto& varIndices = _constrToVar[i];
      size_t nMines = 0;
      for (const auto& varIdx : varIndices) {
        if (_varStates[varIdx] > 0) {
          ++nMines;
        }
      }
      auto select_mines = [&](int UNUSED(v), int row, int col) {
        return mines.get(row, col);
      };
      size_t nMinesMarked = _GameDefs::countNeighbors(
          board, _constraints[i].row(), _constraints[i].col(), select_mines);
      nMines += nMinesMarked;
      int v = arrGet<Board, WIDTH>(
          board, _constraints[i].row(), _constraints[i].col());
      assert(v > 0);
      if (static_cast<size_t>(v) != nMines) {
        /*MINESWEEPER_DEBUG(debug(std::cout) << "Constraint " << i \
            << " (" << _constraints[i].row() << ", " \
            << _constraints[i].col() << ")=" << v << " has " \
            << nMines << " mines" << std::endl);
        MINESWEEPER_DEBUG(debug(std::cout) << "Board:" << std::endl; \
            std::cout << _GameDefs::boardToString(board) << std::endl);
        MINESWEEPER_DEBUG(debug(std::cout) << "State Board:" << std::endl; \
            std::cout << stateToString(board) << std::endl);
        dumpDebugInfo(board);*/
        return false;
      }
    }
    return true;
  }  // checkSolutionAgainstBoard

  std::string stateToString(const Board& board) {
    using BoardChars = std::array<char, WIDTH * HEIGHT>;
    BoardChars boardChars;
    int v;
    char c;
    int k = 0;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        v = board[k];
        switch (v) {
        case Minesweeper::UNKNOWN:
          c = '.';
          break;
        case Minesweeper::BOOM:
          c = 'X';
          break;
        default:
          assert(v >= 0);
          c = '0' + v;
        }
        boardChars[k] = c;
        ++k;
      }
    }
    for (size_t i = 0; i < _varStates.size(); ++i) {
      if (_varStates[i] == 1) {
        arrGet<BoardChars, WIDTH>(
            boardChars, _variables[i].row(), _variables[i].col()) = '*';
      } else if (_varStates[i] == 0) {
        arrGet<BoardChars, WIDTH>(
            boardChars, _variables[i].row(), _variables[i].col()) = '@';
      } else {
        assert(_varStates[i] == -1);
        arrGet<BoardChars, WIDTH>(
            boardChars, _variables[i].row(), _variables[i].col()) = '?';
      }
    }
    std::ostringstream oss;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        oss << arrGet<BoardChars, WIDTH>(boardChars, row, col);
      }
      oss << std::endl;
    }
    return oss.str();
  }  // stateToString

  void enumerateSolutions(const Board& board, const _Mask& mines) {
    _solutions.clear();
    _minNumMines = MINES;
    _maxNumMines = 0;
    std::fill(_varStates.begin(), _varStates.end(), -1);
    std::copy(
        _constrCounts.begin(), _constrCounts.end(), _constrStates.begin());
    _assignedVars.clear();
    int v;
    do {
      if (constraintsSatisfied()) {
        // some constraints might have more mines than specified,
        // check more thoroughly
        if (checkSolutionAgainstBoard(board, mines)) {
          enumerateSolution();
        }
        triggerLastAssignment();
      } else {
        v = nextUnassignedVariable();
        if (v == -1) {
          triggerLastAssignment();
        } else {
          assignVariable(static_cast<size_t>(v));
          if (!checkConsistency()) {
            triggerLastAssignment();
          }
        }
      }
    } while (!_assignedVars.empty());
    MINESWEEPER_DEBUG(
        if (_solutions.empty()) { enumerateSolutionsDebug(board); });
    assert(!_solutions.empty());
  }  // enumerateSolutions

  void enumerateSolutionsDebug(const Board& board) {
    debug(std::cout) << "Debugging session for solutions enumeration!"
                     << std::endl;
    _solutions.clear();
    _minNumMines = MINES;
    _maxNumMines = 0;
    std::fill(_varStates.begin(), _varStates.end(), -1);
    std::copy(
        _constrCounts.begin(), _constrCounts.end(), _constrStates.begin());
    _assignedVars.clear();
    int v;
    do {
      if (constraintsSatisfied()) {
        debug(std::cout) << "Constraints satisfied" << std::endl;
        dumpDebugInfo(board);
        enumerateSolution();
        debug(std::cout) << "Enumerated solution" << std::endl;
        triggerLastAssignment();
        debug(std::cout) << "Triggered last assignment" << std::endl;
        dumpDebugInfo(board);
      } else {
        debug(std::cout) << "Constraints not satisfied" << std::endl;
        dumpDebugInfo(board);
        v = nextUnassignedVariable();
        debug(std::cout) << "Next variable: " << v << std::endl;
        if (v == -1) {
          debug(std::cout) << "No variable to assign" << std::endl;
          dumpDebugInfo(board);
          triggerLastAssignment();
          debug(std::cout) << "Triggered last assignment" << std::endl;
          dumpDebugInfo(board);
        } else {
          debug(std::cout) << "Assign variable " << v << std::endl;
          assignVariable(static_cast<size_t>(v));
          if (!checkConsistency()) {
            debug(std::cout) << "Inconsistency found!" << std::endl;
            dumpDebugInfo(board);
            triggerLastAssignment();
            debug(std::cout) << "Triggered last assignment" << std::endl;
            dumpDebugInfo(board);
          }
        }
      }
    } while (!_assignedVars.empty());
  }  // enumerateSolutionsDebug

  std::ostream& dumpSolutionStats(std::ostream& os) {
    os << "Solution set: ";
    for (const auto& [nmines, solution] : _solutions) {
      os << nmines << " mines (" << solution.size() << " solutions), ";
    }
    return os;
  }  // printSolutionStats

  bool checkConsistency() {
    size_t i = 0;
    size_t nUnassigned;
    size_t nMines;
    for (const Indices& indices : _constrToVar) {
      nUnassigned = 0;
      nMines = 0;
      size_t constrCount = _constrCounts[i];
      size_t constrState = _constrStates[i];
      for (size_t j : indices) {
        switch (_varStates[j]) {
        case -1:
          nUnassigned++;
          break;
        case 1:
          nMines++;
          break;
        default:;
        }
      }
      if ((constrCount < constrState) || (constrCount < nMines) ||
          (nUnassigned < constrState) || (nUnassigned + nMines < constrCount)) {
        // debug(std::cout) << "Inconsistency for constraint " << i
        //    << ": constrCount=" << constrCount
        //    << ", constrState=" << constrState
        //    << ", varCount=" << varCount << std::endl;
        // dumpDebugInfo();
        return false;
      }
      ++i;
    }
    return true;
  }  // SolutionSet::checkConsistency

  void checkCanAssignVar(size_t varIdx) {
    for (size_t j : _varToConstr[varIdx]) {
      if (!_constrStates[j]) {
        /*debug(std::cout) << "Attempt to assign variable " << varIdx
            << " that must not be assigned: constraint " << j
            << " has 0" << std::endl;
        dumpDebugInfo();*/
        assert(false);
      }
    }
  }  // SolutionSet::checkCanAssignVar

  void dumpDebugInfo(const Board& board) {
    debug(std::cout) << "Solution set:" << std::endl;
    debug(std::cout) << "Variables: " << sparseMaskToString(_variables)
                     << std::endl;
    debug(std::cout) << "Constraints: " << sparseMaskToString(_constraints)
                     << std::endl;
    debug(std::cout) << "Variables to constraints: " << std::endl;
    size_t i = 0;
    for (const Indices& indices : _varToConstr) {
      debug(std::cout) << "  " << i++ << ": " << valuesToString(indices)
                       << std::endl;
    }
    debug(std::cout) << "Constraints to variables: " << std::endl;
    i = 0;
    for (const Indices& indices : _constrToVar) {
      debug(std::cout) << "  " << i++ << ": " << valuesToString(indices)
                       << std::endl;
    }
    debug(std::cout) << "Assigned variables: ";
    for (size_t i : _assignedVars) {
      std::cout << i << "=" << _varStates[i] << ", ";
    }
    std::cout << std::endl;
    debug(std::cout) << "Variable states: ";
    i = 0;
    for (int s : _varStates) {
      std::cout << i++ << "=" << s << ", ";
    }
    std::cout << std::endl;
    debug(std::cout) << "Variables order: " << valuesToString(_varOrder)
                     << std::endl;
    debug(std::cout) << "Constraint counts: " << valuesToString(_constrCounts)
                     << std::endl;
    debug(std::cout) << "Constraint states: " << valuesToString(_constrStates)
                     << std::endl;
    debug(std::cout) << "State Board:" << std::endl;
    std::cout << stateToString(board) << std::endl;
  }  // SolutionSet::dumpDebugInfo

  template <typename T> std::string valuesToString(const T& values) {
    std::ostringstream oss;
    for (typename T::value_type v : values) {
      oss << v << ", ";
    }
    return oss.str();
  }  // SolutionSet::indicesToString

 private:
  const Minesweeper::SparseMask& _variables;
  const Minesweeper::SparseMask& _constraints;
  std::vector<Indices> _varToConstr;
  std::vector<Indices> _constrToVar;
  std::unordered_map<size_t, size_t> _mineIdxToVarIdx;

  std::unordered_map<size_t, Solutions> _solutions;
  size_t _minNumMines;
  size_t _maxNumMines;

  // enumeration fields
  std::list<size_t> _assignedVars;
  std::vector<int> _varStates;
  std::vector<size_t> _varOrder;
  std::vector<size_t> _constrCounts;
  std::vector<size_t> _constrStates;

};  // class SolutionSet

}  // namespace vkms
}  // namespace csp
