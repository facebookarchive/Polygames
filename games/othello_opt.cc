/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <cstring>

#include "othello_opt.h"

//#define OTHELLO_DEBUG(arg)
#define OTHELLO_DEBUG(arg) arg

namespace Othello2 {

static constexpr size_t NUM_NEIGHBORS = 8;
static constexpr std::array<int, 8> DROW = {-1, -1, -1,  0, 0,  1, 1, 1};
static constexpr std::array<int, 8> DCOL = {-1,  0,  1, -1, 1, -1, 0, 1};
static constexpr char BOARD_COORD_SMALL_LETTERS[] =
    "abcdefghijkmnopqrstuvwxyz";
static constexpr char BOARD_COORD_CAPITAL_LETTERS[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

template <size_t SIZE>
static constexpr bool isInBoard(int row, int col) {
  return (row >= 0) && (col >= 0) &&
      (row < static_cast<int>(SIZE)) && (col < static_cast<int>(SIZE));
} // isInBoard

template <typename Array, size_t SIZE>
static typename Array::reference
arrGet(Array& arr, size_t row, size_t col) {
  return arr[row * SIZE + col];
} // arrGet

template <typename Array, size_t SIZE>
static typename Array::const_reference
arrGet(const Array& arr, size_t row, size_t col) {
  return arr[row * SIZE + col];
} // arrGet

//template <size_t SIZE>
//Action<SIZE>::Action(size_t row, size_t col, bool skipTurn) {
//  assert(isInBoard<SIZE>(row, col));
//  _loc[0] = (skipTurn ? 1 : 0);
//  _loc[1] = col;
//  _loc[2] = row;
//  _hash = (skipTurn ? SIZE * SIZE : 0) + SIZE * static_cast<uint64_t>(row) +
//      static_cast<uint64_t>(col);
//} // Action<SIZE>::Action

template <size_t SIZE>
State<SIZE>::State(int seed): ::State(seed), _hasher(hashBook) {
  std::call_once(hashBookConfigured, [this]() { hashBook.setup(_rng); });
} // State<SIZE>::State

template <size_t SIZE>
/* virtual */ void State<SIZE>::Initialize() {
  _status = GameStatus::player0Turn;

  initializeBoard();
  initializeHasher();
  initializeCache();

  _featSize = { NUM_PIECE_TYPES, SIZE, SIZE };
  _features.resize(NUM_PIECE_TYPES * SIZE * SIZE, 0);
  fillFeatures();
  fillFullFeatures();

  _actionSize = { NUM_PIECE_TYPES, SIZE, SIZE };
  _NewlegalActions.reserve(SIZE * SIZE - 4);
  RefillLegalActions();

  _hash = _hasher.hash();
} // State<SIZE>::Initialize

template <size_t SIZE>
/* virtual */ std::unique_ptr<mcts::State> State<SIZE>::clone_() const {
  return std::make_unique<State>(*this);
} // State<SIZE>::clone_ 

template <size_t SIZE>
/* virtual */ void State<SIZE>::ApplyAction(const ::_Action& action) {
  assert((_status == GameStatus::player0Turn) ||
      (_status == GameStatus::player1Turn));
  auto stone = stoneToPlay();
  bool skipTurn = (action.GetX() != 0);
  if (skipTurn) {
    //assert(!CanPutStone(stone));
    nextTurn();
    stone = stoneToPlay();
    if (!CanPutStone(stone)) {
      _NewlegalActions.clear();
      setTerminalStatus();
    } else {
      RefillLegalActions();
      fillFeatures();
      fillFullFeatures();
      _hash = _hasher.hash();
    }
    return;
  }
  int col = action.GetY();
  int row = action.GetZ();
  PutStone(stone, row, col);
  if (boardFilled()) {
    setTerminalStatus();
    return;
  }
  nextTurn();
  RefillLegalActions();
  fillFeatures();
  fillFullFeatures();
  _hash = _hasher.hash();
} // State<SIZE>::ApplyAction

template <size_t SIZE>
/* virtual */ void State<SIZE>::DoGoodAction() {
  DoRandomAction();
}
  
template <size_t SIZE>
/* virtual */ void State<SIZE>::printCurrentBoard() const {
  std::cout << boardToString() << std::endl;
} // State<SIZE>::printCurrentBoard

template <size_t SIZE>
std::string State<SIZE>::boardToString() const {
  static constexpr bool SHOW_BOARD_COORDS =
      SIZE < sizeof(BOARD_COORD_CAPITAL_LETTERS);
  //static constexpr bool SHOW_BOARD_COORDS = false;
  std::ostringstream oss;
  if (SHOW_BOARD_COORDS) {
    oss << std::setfill(' ') << std::setw(SIZE / 10 + 1) << ' ' << "  ";
    oss << std::string(BOARD_COORD_CAPITAL_LETTERS, SIZE) << std::endl;
    oss << std::setfill(' ') << std::setw(SIZE / 10 + 1) << ' ' << "  ";
    oss << std::string(SIZE, '-') << std::endl;
  }
  for (size_t row = 0; row < SIZE; ++row) {
    if (SHOW_BOARD_COORDS) {
      oss << std::setfill(' ') << std::setw(SIZE / 10 + 1) << SIZE - row
          << " |";
    }
    for (size_t col = 0; col < SIZE; ++col) {
      switch(arrGet<Board, SIZE>(_board, row, col)) {
      case EMPTY: oss << EMPTY_STR; break;
      case BLACK: oss << BLACK_STR; break;
      case WHITE: oss << WHITE_STR; break;
      default: oss << '?';
      }  
    }
    if (SHOW_BOARD_COORDS) {
      oss << "| " << std::setfill(' ') << std::setw(SIZE / 10 + 1)
          << SIZE - row;
    }
    oss << std::endl;
  }
  if (SHOW_BOARD_COORDS) {
    oss << std::setfill(' ') << std::setw(SIZE / 10 + 1) << ' ' << "  ";
    oss << std::string(SIZE, '-') << std::endl;
    oss << std::setfill(' ') << std::setw(SIZE / 10 + 1) << ' ' << "  ";
    oss << std::string(BOARD_COORD_CAPITAL_LETTERS, SIZE) << std::endl;
  }
  return oss.str();
} // State<SIZE>::boardToString 

template<size_t SIZE>
bool State<SIZE>::CanPutStone(Field stone) const {
  for (size_t row = 0; row < SIZE; ++row) {
    for (size_t col = 0; col < SIZE; ++col) {
      if (CanPutStone(stone, row, col)) {
        return true;
      }
    }
  }
  return false;
} // State<SIZE>::CanPutStone

template<size_t SIZE>
bool State<SIZE>::CanPutStone(Field stone, size_t row, size_t col) const {
  Field field = arrGet<Board, SIZE>(_board, row, col);
  if ((field != EMPTY) || !arrGet<Cache, SIZE>(_cache, row, col)) {
    return false;
  }
  size_t count;
  int dr, dc, r, c;
  for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
    count = 0;
    r = static_cast<int>(row);
    c = static_cast<int>(col);
    dr = DROW[i];
    dc = DCOL[i];
    while(isInBoard<SIZE>(r + dr, c + dc)) {
      field = arrGet<Board, SIZE>(_board, r + dr, c + dc);
      if (field == EMPTY) {
        break;
      } else if (field != stone) {
        // opponent piece
        ++count;
      } else if (!count) {
        // our piece straight after the location
        break;
      } else {
        // our piece after a number of opponent pieces
        return true;
      }
      r += dr;
      c += dc;
    }
  }
  return false;
} // State<SIZE>::CanPutStone

template<size_t SIZE>
void State<SIZE>::PutStone(Field stone, size_t row, size_t col) {
  assert(isInBoard<SIZE>(row, col));
  //assert(CanPutStone(stone, row, col));
  const size_t myHashOffset =
      (stone == BLACK ? HASH_BLACK_OFFSET : HASH_WHITE_OFFSET);
  const size_t theirHashOffset =
      (stone == BLACK ? HASH_WHITE_OFFSET : HASH_BLACK_OFFSET);
  Field field;
  size_t count;
  int dr, dc, r, c;
  bool isInside;
  arrGet<Board, SIZE>(_board, row, col) = stone;
  _hasher.trigger(SIZE * row + col);
  _hasher.trigger(myHashOffset + SIZE * row + col);
  for (size_t i = 0; i < NUM_NEIGHBORS; ++i) {
    count = 0;
    r = static_cast<int>(row);
    c = static_cast<int>(col);
    dr = DROW[i];
    dc = DCOL[i];
    isInside = isInBoard<SIZE>(r + dr, c + dc);
    if (isInside) {
      arrGet<Cache, SIZE>(_cache, r + dr, c + dc) = 1;
    }
    while(isInside) {
      field = arrGet<Board, SIZE>(_board, r + dr, c + dc);
      if (field == EMPTY) {
        break;
      } else if (field != stone) {
        // opponent piece
        ++count;
      } else if (!count) {
        // our piece straight after the location
        break;
      } else {
        // our piece after a number of opponent pieces
        // move back and reverse stones 
        for (size_t j = 0; j < count; ++j) {
          arrGet<Board, SIZE>(_board, r, c) = stone;
          _hasher.trigger(theirHashOffset + SIZE * r + c);
          _hasher.trigger(myHashOffset + SIZE * r + c);
          r -= dr;
          c -= dc;
        }
        break;
      }
      r += dr;
      c += dc;
      isInside = isInBoard<SIZE>(r + dr, c + dc);
    }
  }
} // State<SIZE>::PutStone

template<size_t SIZE>
bool State<SIZE>::boardFilled() const {
  for (size_t i = 0; i < SIZE * SIZE; ++i) {
    if (_board[i] == EMPTY) {
      return false;
    }
  }
  return true;
} // boardFilled

template<size_t SIZE>
constexpr typename State<SIZE>::Field State<SIZE>::stoneToPlay() const {
  switch (_status) {
  case GameStatus::player0Turn: return BLACK;
  case GameStatus::player1Turn: return WHITE;
  default: return EMPTY;
  }
} // State<SIZE>::stoneByStatus

template<size_t SIZE>
void State<SIZE>::nextTurn() {
  _status = (_status == GameStatus::player0Turn) ? 
      GameStatus::player1Turn : GameStatus::player0Turn;
  _hasher.trigger(HASHBOOK_SIZE - 1);
} // State<SIZE>::nextTurn

template<size_t SIZE>
void State<SIZE>::RefillLegalActions() {
  assert((_status == GameStatus::player0Turn) ||
         (_status == GameStatus::player1Turn));
  _NewlegalActions.clear();
  Field stoneToPlay = (_status == GameStatus::player0Turn ? BLACK : WHITE);
  for (size_t row = 0; row < SIZE; ++row) {
    for (size_t col = 0; col < SIZE; ++col) {
      if ((arrGet<Board, SIZE>(_board, row, col) == EMPTY) &&
        CanPutStone(stoneToPlay, row, col)) {
        // add action
        _NewlegalActions.emplace_back(_NewlegalActions.size(), 0, col, row);
      }
    }
  }
  if (_NewlegalActions.empty() && !boardFilled()) {
    _NewlegalActions.emplace_back(_NewlegalActions.size(), 1, SIZE / 2, SIZE / 2);
  }
} // State<SIZE>::RefillLegalAction

template<size_t SIZE>
void State<SIZE>::fillFeatures() {
  auto* featuresBlack = _features.data();
  auto* featuresWhite = featuresBlack + SIZE * SIZE;
  memset(featuresBlack, 0, NUM_PIECE_TYPES * SIZE * SIZE * sizeof(float));
  for (size_t i = 0; i < SIZE * SIZE; ++i) {
    switch(_board[i]) {
      case BLACK: featuresBlack[i] = 1.0; break;
      case WHITE: featuresWhite[i] = 1.0; break;
      default: break;
    }
  }
} // State<SIZE>::fillFeatures

template<size_t SIZE>
void State<SIZE>::initializeBoard() {
  memset(_board.data(), 0, SIZE * SIZE * sizeof(typename Board::value_type));
  _board[WHITE_INIT_OFFSET_1] = WHITE;
  _board[WHITE_INIT_OFFSET_2] = WHITE;
  _board[BLACK_INIT_OFFSET_1] = BLACK;
  _board[BLACK_INIT_OFFSET_2] = BLACK;
} // State<SIZE>::initializeBoard
  
template<size_t SIZE>
void State<SIZE>::initializeHasher() {
  _hasher.reset();
  for (unsigned i = 0; i < _board.size(); ++i) {
    _hasher.trigger(i);
  }
  // black stones
  _hasher.trigger(BLACK_INIT_OFFSET_1);
  _hasher.trigger(HASH_BLACK_OFFSET + BLACK_INIT_OFFSET_1);
  _hasher.trigger(BLACK_INIT_OFFSET_2);
  _hasher.trigger(HASH_BLACK_OFFSET + BLACK_INIT_OFFSET_2);
  // white stones
  _hasher.trigger(WHITE_INIT_OFFSET_1);
  _hasher.trigger(HASH_WHITE_OFFSET + WHITE_INIT_OFFSET_1);
  _hasher.trigger(WHITE_INIT_OFFSET_2);
  _hasher.trigger(HASH_WHITE_OFFSET + WHITE_INIT_OFFSET_2);
} // State<SIZE>::initializeHasher

template<size_t SIZE>
void State<SIZE>::initializeCache() {
  memset(_cache.data(), 0, SIZE * SIZE * sizeof(typename Cache::value_type));
  for (size_t row = SIZE / 2 - 2; row < SIZE / 2 + 2; ++row) {
    for (size_t col = SIZE / 2 - 2; col < SIZE / 2 + 2; ++col) {
      arrGet<Cache, SIZE>(_cache, row, col) = 1;
    }
  }
} // State<SIZE>::initializeCache

template<size_t SIZE>
void State<SIZE>::setTerminalStatus() {
  size_t nWhite = 0;
  size_t nBlack = 0;
  for (size_t i = 0; i < SIZE * SIZE; ++i) {
    switch(_board[i]) {
    case WHITE: ++nWhite; break;
    case BLACK: ++nBlack; break;
    default: break;
    }
  }
  if (nWhite > nBlack) {
    _status = GameStatus::player1Win;
  } else if (nBlack > nWhite) {
    _status = GameStatus::player0Win;
  } else {
    _status = GameStatus::tie;
  }
} // State<SIZE>::setTerminalStatus

template class State<4>;
template class State<6>;
template class State<8>;
template class State<10>;
template class State<12>;
template class State<14>;
template class State<16>;

} // namespace Othello2

