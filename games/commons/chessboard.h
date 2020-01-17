/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: 林鈺錦 (Yù-Jǐn Lín)
// - Github: https://github.com/abc1236762
// - Email:  abc1236762@outlook.com

#pragma once

#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using Chess = std::uint8_t;

template <int ROW, int COL> class Chessboard {
  static_assert(ROW > 0 && ROW <= 26 && COL > 0 && COL <= 26,
                "rows and columns of chessboard must be in range [1,26]");

 public:
  static constexpr int rows = ROW;
  static constexpr int columns = COL;
  static constexpr int squares = rows * columns;
  using Board = std::array<Chess, squares>;

  template <typename RE>
  static void setup(const std::vector<std::string_view>& cn,
                    const std::vector<std::string_view>& cs,
                    const RE& re);
  static constexpr bool isPosInBoard(int x, int y);
  static constexpr bool isPosInBoard(int xy);
  static constexpr int posTo1D(int x, int y);
  static constexpr std::tuple<int, int> posTo2D(int xy);
  static constexpr std::string_view getMarkSymbol();
  static constexpr std::string_view getChessName(Chess chess);
  static constexpr std::string_view getChessSymbol(Chess chess);

  void initialize();
  void initialize(const Board& b);
  Chess getChess(int x, int y) const;
  Chess getChess(int xy) const;
  void setChess(int x, int y, Chess chess);
  void setChess(int xy, Chess chess);
  std::vector<int> countChesses() const;
  void turnHash();

  std::string sprint(std::string_view prefix = "") const;
  virtual std::string sprintBoard(
      std::string_view prefix = "",
      const std::set<std::tuple<int, int>>& markedPos = {}) const;
  const Board& getBoard() const;
  std::uint64_t getHash() const;
  virtual std::string getPosStr(int xy) const;
  virtual std::string getPosStr(int x, int y) const;
  virtual std::optional<std::tuple<int, int>> parsePosStr(
      const std::string& str) const;

  constexpr bool operator==(const Chessboard& cb) const;
  constexpr bool operator!=(const Chessboard& cb) const;

 protected:
  static inline std::string_view markSymbol = "?";

 private:
  void updateHash();
  void updateHash(int xy, Chess chess);

  static inline std::vector<std::uint64_t> hashList;
  static inline std::uint64_t hashTurn;
  static inline std::size_t chessKinds;
  static inline std::vector<std::string_view> chessesName;
  static inline std::vector<std::string_view> chessesSymbol;

  Board board;
  std::uint64_t hash;
};

template <int ROW, int COL>
template <typename RE>
void Chessboard<ROW, COL>::setup(const std::vector<std::string_view>& cn,
                                 const std::vector<std::string_view>& cs,
                                 const RE& re) {
  assert(cn.size() == cs.size());
  chessKinds = cn.size();
  chessesName = cn;
  chessesSymbol = cs;

  std::independent_bits_engine<RE, 64UL, std::uint64_t> genRandomBits(re);
  hashList = std::vector<std::uint64_t>(chessKinds * squares);
  for (std::uint64_t& hash : hashList)
    hash = genRandomBits();
  hashTurn = genRandomBits();
}

template <int ROW, int COL>
constexpr bool Chessboard<ROW, COL>::isPosInBoard(int x, int y) {
  return x >= 0 && x < rows && y >= 0 && y < columns;
}

template <int ROW, int COL>
constexpr bool Chessboard<ROW, COL>::isPosInBoard(int xy) {
  auto [x, y] = posTo2D(xy);
  return isPosInBoard(x, y);
}

template <int ROW, int COL>
constexpr int Chessboard<ROW, COL>::posTo1D(int x, int y) {
  return rows * y + x;
}

template <int ROW, int COL>
constexpr std::tuple<int, int> Chessboard<ROW, COL>::posTo2D(int xy) {
  return {xy % rows, xy / rows};
}

template <int ROW, int COL>
constexpr std::string_view Chessboard<ROW, COL>::getMarkSymbol() {
  return markSymbol;
}

template <int ROW, int COL>
constexpr std::string_view Chessboard<ROW, COL>::getChessName(Chess chess) {
  return chessesName.at(chess);
}

template <int ROW, int COL>
constexpr std::string_view Chessboard<ROW, COL>::getChessSymbol(Chess chess) {
  return chessesSymbol.at(chess);
}

template <int ROW, int COL> void Chessboard<ROW, COL>::initialize() {
  board.fill(0U);
  updateHash();
}

template <int ROW, int COL>
void Chessboard<ROW, COL>::initialize(const Chessboard<ROW, COL>::Board& b) {
  board = b;
  updateHash();
}

template <int ROW, int COL>
Chess Chessboard<ROW, COL>::getChess(int x, int y) const {
  return getChess(posTo1D(x, y));
}

template <int ROW, int COL> Chess Chessboard<ROW, COL>::getChess(int xy) const {
  return board[xy];
}

template <int ROW, int COL>
void Chessboard<ROW, COL>::setChess(int x, int y, Chess chess) {
  setChess(posTo1D(x, y), chess);
}

template <int ROW, int COL>
void Chessboard<ROW, COL>::setChess(int xy, Chess chess) {
  updateHash(xy, getChess(xy));
  board[xy] = chess;
  updateHash(xy, chess);
}

template <int ROW, int COL>
std::vector<int> Chessboard<ROW, COL>::countChesses() const {
  std::vector<int> counts(chessKinds, 0);
  for (Chess chess : board)
    counts[chess]++;
  return counts;
}

template <int ROW, int COL> void Chessboard<ROW, COL>::turnHash() {
  hash ^= hashTurn;
}

template <int ROW, int COL>
std::string Chessboard<ROW, COL>::sprint(std::string_view prefix) const {
  std::ostringstream oss;
  oss << prefix;
  for (std::size_t i = 0; i < chessKinds; i++) {
    oss << getChessName(i) << "='" << getChessSymbol(i) << "'";
    if (i < chessKinds - 1)
      oss << " ";
  }
  oss << std::endl << sprintBoard(prefix);
  return oss.str();
}

template <int ROW, int COL>
std::string Chessboard<ROW, COL>::sprintBoard(
    std::string_view prefix,
    const std::set<std::tuple<int, int>>& markedPos) const {
  auto hr = [&](std::string_view l, std::string_view m, std::string_view r) {
    std::ostringstream ossE;
    ossE << (columns < 10 ? "  " : "   ") << l;
    for (int x = 0; x < rows; x++) {
      ossE << "───";
      if (x < rows - 1)
        ossE << m;
    }
    ossE << r << std::endl;
    return ossE.str();
  };

  std::ostringstream oss, ossW;
  ossW << (columns < 10 ? " " : "  ");
  for (int x = 0; x < rows; x++)
    ossW << "   " << std::string(1, 'A' + x);
  ossW << std::endl;

  oss << prefix << ossW.str() << prefix << hr("┌", "┬", "┐");
  for (int y = 0; y < columns; y++) {
    char yStr[4];
    sprintf(yStr, columns < 10 ? "%d" : "%02d", columns - y);
    oss << prefix << yStr << " │ ";
    for (int x = 0; x < rows; x++) {
      if (markedPos.count({x, y}) == 0)
        oss << getChessSymbol(getChess(x, y)) << " │ ";
      else
        oss << markSymbol << " │ ";
    }
    oss << yStr << std::endl;
    if (y < columns - 1)
      oss << prefix << hr("├", "┼", "┤");
  }
  oss << prefix << hr("└", "┴", "┘") << prefix << ossW.str();
  return oss.str();
}

template <int ROW, int COL>
const typename Chessboard<ROW, COL>::Board& Chessboard<ROW, COL>::getBoard()
    const {
  return board;
}

template <int ROW, int COL>
std::uint64_t Chessboard<ROW, COL>::getHash() const {
  return hash;
}

template <int ROW, int COL>
std::string Chessboard<ROW, COL>::getPosStr(int xy) const {
  auto [x, y] = posTo2D(xy);
  return getPosStr(x, y);
}

template <int ROW, int COL>
std::string Chessboard<ROW, COL>::getPosStr(int x, int y) const {
    char str[4] = {0};
  str[0] = 'A' + x;
  y = columns - y;
  if (columns >= 10) {
    str[1] = '0' + y / 10;
    str[2] = '0' + y % 10;
  } else {
    str[1] = '0' + y;
  }
  return std::string(str);
}

template <int ROW, int COL>
std::optional<std::tuple<int, int>> Chessboard<ROW, COL>::parsePosStr(
    const std::string& str) const {
  int x = -1, y = -1, yBegin = -1, yEnd = -1;
  for (std::size_t i = 0; i < str.size(); i++) {
    char c = str[i];
    if (x < 0) {
      if (std::isspace(c))
        continue;
      else if (!std::isalpha(c))
        return std::nullopt;
      x = std::toupper(c) - 'A';
    } else if (yBegin < 0) {
      if (std::isspace(c))
        continue;
      else if (!std::isdigit(c))
        return std::nullopt;
      yBegin = i;
    } else if (yEnd < 0) {
      if (std::isdigit(c))
        continue;
      else if (!std::isspace(c))
        return std::nullopt;
      yEnd = i;
    } else if (!std::isspace(c)) {
      return std::nullopt;
    }
  }
  if (x < 0 || yBegin < 0)
    return std::nullopt;
  if (yEnd < 0)
    yEnd = str.size();
  std::string yStr(str, yBegin, yEnd - yBegin);
  try {
    y = columns - std::stoul(yStr, nullptr, 10);
  } catch (...) {
    return std::nullopt;
  }
  if (isPosInBoard(x, y))
    return std::tuple(x, y);
  return std::nullopt;
}

template <int ROW, int COL>
constexpr bool Chessboard<ROW, COL>::operator==(const Chessboard& cb) const {
  if (hash == cb.hash && board == cb.board)
    return true;
  return false;
}

template <int ROW, int COL>
constexpr bool Chessboard<ROW, COL>::operator!=(const Chessboard& cb) const {
  if (hash != cb.hash || board != cb.board)
    return true;
  return false;
}

template <int ROW, int COL> void Chessboard<ROW, COL>::updateHash() {
  hash = 0ULL;
  for (int xy = 0; xy < squares; xy++)
    updateHash(xy, getChess(xy));
}

template <int ROW, int COL>
void Chessboard<ROW, COL>::updateHash(int xy, Chess chess) {
  hash ^= hashList[squares * chess + xy];
}
