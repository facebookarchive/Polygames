/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <sstream>

#include "../core/state.h"
#include "commons/hash.h"
#include "minesweeper_common.h"
#include "minesweeper_csp_vkms/CspStrategy.h"

#define EXPAND_ZEROS
#define FLAG_MINES

namespace Minesweeper {

template <size_t WIDTH, size_t HEIGHT, size_t MINES>
class Action : public ::_Action {
 public:

  /**
   * Probe action at (row, column)
   *   X = column, Y = row
   */
  Action(int row, int column) {
    assert(0 <= column);
    assert(column < static_cast<int>(WIDTH));
    assert(0 <= row);
    assert(row < static_cast<int>(HEIGHT));
    _loc[0] = 0;
    _loc[1] = row;
    _loc[2] = column;
    _hash = uint32_t(static_cast<uint32_t>(row) * WIDTH +
        static_cast<uint32_t>(column));
    _i = -1;  // to be updated after the action is created (position in
              // _legalActions)
  } // Action::Action

  int row() const {
    return _loc[1];
  }

  int column() const {
    return _loc[2];
  }

  friend std::ostream& operator<<(std::ostream& os, const Action& action) {
    os << '(' << row() << ',' << column() << ')';
    return os;
  }
};

template <size_t WIDTH, size_t HEIGHT, size_t MINES>
class State : public core::State {

  static constexpr size_t HASHBOOK_SIZE = WIDTH * HEIGHT * 11;
  using _Mask = Mask<WIDTH, HEIGHT, MINES>;
  using _HashBook = HashBook<uint64_t, HASHBOOK_SIZE>;
  using _Hasher = Hasher<uint64_t, HASHBOOK_SIZE>;

 public:

  using Act = Action<WIDTH, HEIGHT, MINES>;
  using _GameDefs = GameDefs<WIDTH, HEIGHT, MINES>;
  using Board = typename _GameDefs::Board;
  using BoardProbas = typename _GameDefs::BoardProbas;
  using Mines = typename _GameDefs::Mines;
  using Neighbors = typename _GameDefs::Neighbors;

  State(int seed) : core::State(seed),
      _minesToBoardDeltaIdx(
          NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::dindices),
      _minesToBoardDeltaRow(
          NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::drow),
      _minesToBoardDeltaCol(
          NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::dcol),
      _hasher(hashBook) {
    _board.fill(UNKNOWN);
    _boardSample.fill(UNKNOWN);
    _minesSample.fill(-1); 
    std::call_once(hashBookConfigured, [this]() {
        hashBook.setup(_rng);
    });
  } // State::State

  virtual void Initialize() override {
    MINESWEEPER_DEBUG(debug(std::cout) << "Initialize" << std::endl);
    _status = GameStatus::player0Turn;
    _featSize = {2, HEIGHT, WIDTH};
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2], 0);
    fillFullFeatures();
    _actionSize = {1, HEIGHT, WIDTH};
    _stochastic = true;
    _board.fill(UNKNOWN);
    _boardSample.fill(UNKNOWN);
    _legalActions.clear();
    fillLegalActions(_legalActions, _board, std::vector<int>());
    MINESWEEPER_DEBUG(debug(std::cout) << "Num legal actions: " << \
        _legalActions.size() << std::endl);
    _hash = boardHash();
  } // State::Initialize

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<Minesweeper::State<WIDTH, HEIGHT, MINES>>(*this);
  } // State::clone_
  
  virtual void ApplyAction(const _Action& action) override {
    MINESWEEPER_DEBUG(debug(std::cout) << "ApplyAction" << std::endl);
    assert(_status == GameStatus::player0Turn);
    assert(!_legalActions.empty());
    int row = action.GetY();
    int col = action.GetZ();
    assert((isInBoard<WIDTH, HEIGHT>(row, col)));
    sampleMines(_minesSample, _board, _rng, row, col);
    minesToBoard(_minesSample, _boardSample);
    applyActionToSampledBoard(row, col);
    _hash = boardHash();
  } // State::ApplyAction
  
  virtual void DoGoodAction() override {
    assert(!_legalActions.empty());
    if (isFirstMove(_board)) {
      MINESWEEPER_DEBUG(debug(std::cout) << "Apply Random Action" \
          << ", rng=\"" << _rng << "\"" << std::endl);
      DoRandomAction();
      return;
    }
    using CspStrategy = csp::vkms::CspStrategy<WIDTH, HEIGHT, MINES>;
    CspStrategy cspStrategy;
    cspStrategy.computeMineProbabilitiesAndSampleMines(
        _minesSample, _board, _rng);
    // greedy choice of the best location to probe
    using MineProbas = std::array<float, WIDTH * HEIGHT>;
    const MineProbas& mineProbas = cspStrategy.getMineProbabilities();
    MINESWEEPER_DEBUG(debug(std::cout) << "Mine probabilities:" << std::endl; \
        for (size_t row = 0; row < HEIGHT; ++row) { \
            for (size_t col = 0; col < WIDTH; ++col) { \
                std::cout << setw(10) << arrGet<MineProbas \
                MINESWEEPER_DEBUG_COMMA  WIDTH>(mineProbas, row, col); \
            } \
            std::cout << std::endl; \
        } \
    ); 
    int row = -1, row_i;
    int col = -1, col_i;
    float proba = 1.0, probaMin = -1.0;
    for (size_t i = 0; i < _legalActions.size(); i++) {
      row_i = _legalActions[i]->GetY();
      col_i = _legalActions[i]->GetZ();
      proba = arrGet<typename CspStrategy::MineProbas, WIDTH>(
          mineProbas, row_i, col_i); 
      if ((proba < probaMin) || (probaMin < 0)) {
        probaMin = proba;
        row = row_i;
        col = col_i;
      }
    }
    minesToBoard(_minesSample, _boardSample);
    applyActionToSampledBoard(row, col);
  } // State::ApplyAction

  /**
   * Expects action in format "r,c", where r and c are row and column
   * of a cell to be probed. The coordinates must be non-negative integers
   * that do not exceed board size. Action validity is performed
   */
  virtual int parseAction(const std::string& str) const override {
    int row = -1;
    int col = -1;
    char c = 0;
    std::istringstream iss(str);
    iss >> c;
    if (!iss.good() || (c != '(')) {
        return -1;
    }
    iss >> row;
    if (!iss.good() || (row < 0) || (row >= static_cast<int>(HEIGHT))) {
        return -1;
    }
    iss >> c;
    if (!iss.good() || (c != ',')) {
        return -1;
    }
    iss >> col;
    if (!iss.good() || (col < 0) || (col >= static_cast<int>(WIDTH))) {
        return -1;
    }
    if (!(iss.good() || iss.eof()) || (c != ')')) {
        return -1;
    }
    for (size_t i = 0; i < _legalActions.size(); i++) {
      if ((_legalActions[i]->GetY() == row) &&
          (_legalActions[i]->GetZ() == col)) {
        return (int)i;
      }
    }
    return -1;
  } // State::parseAction

  virtual bool isOnePlayerGame() const override {
    return true;
  }

  std::array<float, WIDTH * HEIGHT> getMineProbas() {
    using CspStrategy = csp::vkms::CspStrategy<WIDTH, HEIGHT, MINES>;
    CspStrategy cspStrategy;
    cspStrategy.computeMineProbabilitiesAndSampleMines(
        _minesSample, _board, _rng);
    using MineProbas = std::array<float, WIDTH * HEIGHT>;
    const MineProbas& mineProbas = cspStrategy.getMineProbabilities();
    return mineProbas;
  }

  /*
   * -1 means "unknown, could be a mine"
   *  k >=0 means "k mines in the neighborhood"
   */
  virtual std::string stateDescription() const override {
    std::string boardStr = _GameDefs::boardToString(_board);
    return boardStr;
  } // State::stateDescription

 private:

  void applyActionToSampledBoard(int row, int col) {
    MINESWEEPER_DEBUG(displayBoard("Current board:", _board));

    MINESWEEPER_DEBUG(displayBoard("Sampled board:", _boardSample));
    MINESWEEPER_DEBUG(checkConsistency(_boardSample, _board));
    MINESWEEPER_DEBUG(debug(std::cout) << "Probe: row=" << row << ", col=" \
        << col << ": ");

    int value = arrGet<Board, WIDTH>(_boardSample, row, col);
    if (value == BOOM) {
       MINESWEEPER_DEBUG(std::cout << "BOOM!" << std::endl);
      _status = GameStatus::player1Win;
      _legalActions.clear();
      fillFeatures(_features, _board);
      fillFullFeatures();
      return;
    }
    arrGet<Board, WIDTH>(_board, row, col) = value;
    MINESWEEPER_DEBUG(std::cout << "value=" << value << std::endl);
    #ifdef EXPAND_ZEROS
    if (!value) {
      expandZeros(_board, _boardSample, row, col);
      MINESWEEPER_DEBUG(debug(std::cout) << "Expanded zeros" << std::endl);
      MINESWEEPER_DEBUG(displayBoard("Current board:", _board));
    }
    #endif
    if (done()) {
       MINESWEEPER_DEBUG(debug(std::cout) << "Done." << std::endl);
      _status = GameStatus::player0Win;
      _legalActions.clear();
      fillFeatures(_features, _board);
      fillFullFeatures();
      return;
    }
    using CspStrategy = csp::vkms::CspStrategy<WIDTH, HEIGHT, MINES>;
    CspStrategy cspStrategy;
    auto forSureMines = cspStrategy.locateForSureMines(_board);
    _legalActions.clear();
    if (_status == GameStatus::player0Turn) {
      fillLegalActions(_legalActions, _board, forSureMines);
      MINESWEEPER_DEBUG(debug(std::cout) << "Num legal actions: " << \
          _legalActions.size() << std::endl);
    }
    fillFeatures(_features, _board);
    fillFullFeatures();
  }

  void expandZeros(Board& board, const Board& boardSample,
      int row, int col) {
    int value = arrGet<Board, WIDTH>(board, row, col);
    assert(!value);
    _expandZerosProcessedMask.zero();
    auto select_unprocessed =
        [this](int UNUSED(v), int row, int col)
        { return !this->_expandZerosProcessedMask.get(row, col); };
    std::list<int> queue;
    int idx = rowColToIdx<WIDTH>(row, col);
    queue.push_back(idx);
    _expandZerosProcessedMask.set(row, col);
    while (!queue.empty()) {
      idx = queue.front();
      queue.pop_front();
      idxToRowCol<WIDTH>(idx, row, col);
      auto neighborPositions = _GameDefs::getNeighbors(
          board, row, col, select_unprocessed);
      for (const auto& pos : neighborPositions) {
        value = arrGet<Board, WIDTH>(boardSample, pos.row(), pos.col());
        arrGet<Board, WIDTH>(board, pos.row(), pos.col()) = value;
        if (!value) {
          idx = rowColToIdx<WIDTH>(pos.row(), pos.col());
          queue.push_back(idx);
          _expandZerosProcessedMask.set(pos.row(), pos.col());
        }
      }
    }
  } // expandZeros

  template<typename RngEngine>
  void sampleMines(Mines& minesSample, const Board& board, RngEngine& rng,
      int row, int col) {
    int probeIdx = rowColToIdx<WIDTH>(row, col);
    MINESWEEPER_DEBUG(debug(std::cout) << "Is first move? " \
        << (isFirstMove(board) ? "yes" : "no") << std::endl);
    if (isFirstMove(board)) {
      MINESWEEPER_DEBUG(debug(std::cout) << "Sample mines uniformly " \
          << "without duplicates" << std::endl);
      std::vector<int> cellIndices(HEIGHT * WIDTH);
      for (size_t i = 0; i < HEIGHT * WIDTH; ++i) {
        cellIndices[i] = i;
      }
      std::swap(cellIndices[0], cellIndices[probeIdx]);
      for (size_t i = 0; i < MINES; ++i) {
        std::uniform_int_distribution<size_t> distribution(
          i + 1, cellIndices.size() - 1);
        size_t varIdx = distribution(rng);
        minesSample[i] = cellIndices[varIdx];
        std::swap(cellIndices[i+1], cellIndices[varIdx]);
      }
      std::sort(minesSample.begin(), minesSample.end());
    } else {
      // do CSP sampling
      MINESWEEPER_DEBUG(debug(std::cout) << "Sample mines with CSP: " \
          << std::endl);
      using CspStrategy = csp::vkms::CspStrategy<WIDTH, HEIGHT, MINES>;
      CspStrategy cspStrategy;
      cspStrategy.sampleMines(minesSample, board, rng);
    }
    MINESWEEPER_DEBUG(displayMines("Sampled mines:", minesSample));
    checkMinesSample(minesSample);
  } // sampleMines

  static bool isFirstMove(const Board& board) {
    size_t unknown = 0;
    for (size_t k = 0; k < HEIGHT * WIDTH; ++k) {
      if (board[k] == UNKNOWN) {
        ++unknown;
      }
    }
    return (unknown == HEIGHT * WIDTH);
  } // shouldDoRejectionSampling

  static bool hasDuplicates(const Mines& mines) {
    for (size_t k = 1; k < MINES; ++k) {
      if (mines[k - 1] == mines[k]) {
        return true;
      }
    }
    return false;
  } // hasDuplicates

  /**
   * Check that mine indices are valid and are in srtrictly ascending order
   */
  static void checkMinesSample(Mines& minesSample) {
#ifndef NDEBUG
    int prevMineIdx = -1;
    for (int mineIdx : minesSample) {
      assert(mineIdx >= 0);
      assert(mineIdx < static_cast<int>(HEIGHT * WIDTH));
      assert(mineIdx > prevMineIdx);
      prevMineIdx = mineIdx;
    }
#endif
  } // checkMinesSample

  /**
   * Check that sampled board complies with the current board 
   */
  static void checkConsistency(const Board& boardSample, const Board& board) {
    for (size_t i = 0; i < WIDTH * HEIGHT; ++i) {
      if (board[i] != UNKNOWN) {
        assert(boardSample[i] == board[i]);
      }
    }
  } // checkConsistency
  
  void minesToBoard(const Mines& mines, Board& board) {
    memset(board.data(), 0, WIDTH * HEIGHT * sizeof(int));
    int idx, row, col;
    for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
      for (size_t j = 0; j < MINES; ++j) {
        idx = mines[j];
        idxToRowCol<WIDTH>(idx, row, col);
        idx += _minesToBoardDeltaIdx[i];
        row += _minesToBoardDeltaRow[i];
        col += _minesToBoardDeltaCol[i];
        if ((row >= 0) && (row < static_cast<int>(HEIGHT)) &&
            (col >= 0) && (col < static_cast<int>(WIDTH))) {
          board[idx]++;
        }
      }
    }
    for (size_t j = 0; j < MINES; ++j) {
      board[mines[j]] = BOOM;
    }
  } // minesToBoard

  static void fillLegalActions(
      std::vector<std::shared_ptr<::_Action>>& legalActions,
      const Board& board,
      #ifdef FLAG_MINES
      const std::vector<int>& forSureMines
      #else
      const std::vector<int>& UNUSED(forSureMines)
      #endif
  ) {
    legalActions.reserve(HEIGHT * WIDTH);
    auto k = 0;
    auto n = 0;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        if (board[k++] == UNKNOWN) {
          std::shared_ptr<::_Action> act = std::make_shared<Act>(row, col);
          #ifdef FLAG_MINES
          if (std::binary_search(forSureMines.begin(), forSureMines.end(),
              /*Minesweeper::template*/ rowColToIdx<WIDTH>(row, col))) {
            continue;
          }
          #endif
          legalActions.push_back(act);
          act->SetIndex(n++);
        }
      }
    }
  } // fillLegalActions
   
  static void fillFeatures(std::vector<float>& features,
      const Board& board) {
    for (size_t k = 0; k < HEIGHT * WIDTH; ++k) {
      features[k] = board[k];
      features[HEIGHT * WIDTH + k] = (board[k] < 0 ? 1 : 0);
    }
  } // fillFeatures

  static void displayBoard(const std::string& title, const Board& board) {
      std::cout << title << std::endl;
      std::string boardStr = _GameDefs::boardToString(board);
      std::cout << boardStr;
  } // displayBoard
  
  static void displayMines(const std::string& title, const Mines& mines) {
      std::cout << title << std::endl;
      std::string minesStr = _GameDefs::minesToString(mines);
      std::cout << minesStr << std::endl;
  } // displayMines

  static bool boardAllUnknown(const Board& board) {
    for (size_t k = 0; k < HEIGHT * WIDTH; ++k) {
      if (board[k] != UNKNOWN) {
        return false;
      }
    }
    return true;
  } // boardAllUnknown 

  bool done() const {
    size_t nUnknown = 0;
    for (size_t k = 0; k < HEIGHT * WIDTH; ++k) {
      if (_board[k] == UNKNOWN) {
        ++nUnknown;
      }
    }
    return (nUnknown == MINES);
  } // done

  uint64_t boardHash() {
    _hasher.reset();
    int v;
    for (unsigned i = 0; i < _board.size(); ++i) {
      v = _board[i] + 2;
      assert (v >= 0);
      _hasher.trigger(static_cast<size_t>(v) * WIDTH * HEIGHT + i);
    }
    return _hasher.hash();
  } // boardHash 

  Board _board;
  Board _boardSample;
  Mines _minesSample;
  _Mask _expandZerosProcessedMask;
  const Neighbors _minesToBoardDeltaIdx;
  const Neighbors _minesToBoardDeltaRow;
  const Neighbors _minesToBoardDeltaCol;

  static std::once_flag hashBookConfigured;
  static _HashBook hashBook;
  _Hasher _hasher;

}; // class State

template <size_t WIDTH, size_t HEIGHT, size_t MINES>
std::once_flag State<WIDTH, HEIGHT, MINES>::hashBookConfigured;

template <size_t WIDTH, size_t HEIGHT, size_t MINES>
typename State<WIDTH, HEIGHT, MINES>::_HashBook
State<WIDTH, HEIGHT, MINES>::hashBook;

} // namespace Minesweeper

