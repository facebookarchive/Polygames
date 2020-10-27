/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define MINESWEEPER_DEBUG_COMMA ,
// Debug output disabled:
#define MINESWEEPER_DEBUG(ARG)
// Debug output enabled:
//#define MINESWEEPER_DEBUG(ARG) ARG
#define UNUSED(ARG)

namespace Minesweeper {

template <typename Offset, size_t STRIDE, size_t NEIGHBORS>
struct NeighborOffsets {};

template <typename Offset, size_t STRIDE>
struct NeighborOffsets<Offset, STRIDE, 8> {
  static constexpr std::array<Offset, 8> dindices = {
      -static_cast<Offset>(STRIDE) - 1,
      -static_cast<Offset>(STRIDE),
      -static_cast<Offset>(STRIDE) + 1,
      -1,
      1,
      static_cast<Offset>(STRIDE) - 1,
      static_cast<Offset>(STRIDE),
      static_cast<Offset>(STRIDE) + 1};
  static constexpr std::array<Offset, 8> drow = {-1, -1, -1, 0, 0, 1, 1, 1};
  static constexpr std::array<Offset, 8> dcol = {-1, 0, 1, -1, 1, -1, 0, 1};
};  // class NeighborOffsets<Offset, STRIDE, 8>

class BoardPosition {
 public:
  constexpr BoardPosition(int row, int col) noexcept
      : _r(row)
      , _c(col) {
  }
  constexpr int row() const noexcept {
    return _r;
  }
  constexpr int col() const noexcept {
    return _c;
  }

 private:
  int _r;
  int _c;
};  // class BoardPosition

template <size_t STRIDE> int rowColToIdx(int row, int col) {
  return row * static_cast<int>(STRIDE) + col;
}

template <size_t STRIDE> void idxToRowCol(int idx, int& row, int& col) {
  row = idx / STRIDE;
  col = idx % STRIDE;
}

template <size_t WIDTH, size_t HEIGHT>
constexpr bool isInBoard(int row, int col) {
  return (row >= 0) && (row < static_cast<int>(HEIGHT)) && (col >= 0) &&
         (col < static_cast<int>(WIDTH));
}

template <typename T, size_t STRIDE>
typename T::const_reference arrGet(const T& arr, int row, int col) {
  return arr[rowColToIdx<STRIDE>(row, col)];
}

template <typename T, size_t STRIDE>
typename T::reference arrGet(T& arr, int row, int col) {
  return arr[rowColToIdx<STRIDE>(row, col)];
}

using SparseMask = std::vector<BoardPosition>;

static constexpr int UNKNOWN = -1;
static constexpr int BOOM = -2;
static constexpr size_t NUM_NEIGHBORS = 8;

template <size_t WIDTH, size_t HEIGHT, size_t MINES> struct GameDefs {
  using Board = std::array<int, HEIGHT * WIDTH>;
  using BoardProbas = std::array<float, HEIGHT * WIDTH>;
  using BoardMask = std::array<bool, HEIGHT * WIDTH>;
  using Mines = std::array<int, MINES>;
  using Neighbors = std::array<int, NUM_NEIGHBORS>;

  static std::string boardMaskToString(const BoardMask& mask) {
    std::ostringstream oss;
    int k = 0;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        oss << (mask[k++] ? 1 : 0);
      }
      oss << std::endl;
    }
    return oss.str();
  }  // boardMaskToString

  static std::string boardToString(const Board& board) {
    using BoardChars = std::array<char, WIDTH * HEIGHT>;
    BoardChars boardChars;
    int v;
    char c;
    int k = 0;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        v = board[k];
        switch (v) {
        case UNKNOWN:
          c = '?';
          break;
        case BOOM:
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
    std::ostringstream oss;
    for (size_t row = 0; row < HEIGHT; ++row) {
      for (size_t col = 0; col < WIDTH; ++col) {
        oss << arrGet<BoardChars, WIDTH>(boardChars, row, col);
      }
      oss << std::endl;
    }
    return oss.str();
  }  // boardToString

  static std::string minesToString(const Mines& mines) {
    std::ostringstream oss;
    for (size_t i = 0; i < MINES; ++i) {
      oss << mines[i] << " ";
    }
    return oss.str();
  }  // minesToString

  template <typename Predicate>
  static std::vector<BoardPosition> getNeighbors(const Board& board,
                                                 int row,
                                                 int col,
                                                 Predicate predicate) {
    std::vector<BoardPosition> result;
    result.reserve(NUM_NEIGHBORS);
    int row_i, col_i;
    for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
      row_i = row + NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::drow[i];
      col_i = col + NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::dcol[i];
      if (isInBoard<WIDTH, HEIGHT>(row_i, col_i) &&
          predicate(arrGet<Board, WIDTH>(board, row_i, col_i), row_i, col_i)) {
        result.emplace_back(row_i, col_i);
      }
    }
    return result;
  }  // getNeighbors

  template <typename Predicate, typename Mask>
  static void markNeighbors(
      const Board& board, int row, int col, Mask& mask, Predicate predicate) {
    int row_i, col_i;
    for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
      row_i = row + NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::drow[i];
      col_i = col + NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::dcol[i];
      if (isInBoard<WIDTH, HEIGHT>(row_i, col_i) &&
          predicate(arrGet<Board, WIDTH>(board, row_i, col_i), row_i, col_i)) {
        mask.set(row_i, col_i);
      }
    }
  }  // markNeighbors

  template <typename Predicate>
  static size_t countNeighbors(const Board& board,
                               int row,
                               int col,
                               Predicate predicate) {
    size_t count = 0;
    int row_i, col_i;
    for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
      row_i = row + NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::drow[i];
      col_i = col + NeighborOffsets<int, WIDTH, NUM_NEIGHBORS>::dcol[i];
      if (isInBoard<WIDTH, HEIGHT>(row_i, col_i) &&
          predicate(arrGet<Board, WIDTH>(board, row_i, col_i), row_i, col_i)) {
        ++count;
      }
    }
    return count;
  }  // countNeighbors
};

template <size_t WIDTH, size_t HEIGHT, size_t MINES> class Mask {
 private:
  using BoardMask = typename GameDefs<WIDTH, HEIGHT, MINES>::BoardMask;

 public:
  explicit Mask(size_t sparseSize = MINES) {
    _maskSparse.reserve(sparseSize);
  }

  void zero() {
    memset(_maskDense.data(), 0,
           WIDTH * HEIGHT * sizeof(typename BoardMask::value_type));
    _maskSparse.clear();
  }

  void set(int row, int col) {
    if (!_maskDense[rowColToIdx<WIDTH>(row, col)]) {
      _maskDense[rowColToIdx<WIDTH>(row, col)] = 1;
      _maskSparse.emplace_back(row, col);
    }
  }

  typename BoardMask::value_type get(int row, int col) const {
    return _maskDense[rowColToIdx<WIDTH>(row, col)];
  }

  const BoardMask& dense() const {
    return _maskDense;
  }

  const std::vector<BoardPosition>& sparse() const {
    return _maskSparse;
  }

 private:
  BoardMask _maskDense;
  SparseMask _maskSparse;
};  // class Mask

std::ostream& debug(std::ostream& os);

std::string sparseMaskToString(const SparseMask& mask);

}  // namespace Minesweeper
