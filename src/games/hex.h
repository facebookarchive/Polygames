/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "game_player.h"

#include <array>
#include <cassert>
#include <optional>
#include <random>
#include <set>

namespace Hex {

enum Color { COLOR_BLACK, COLOR_WHITE, COLOR_NONE };

using Cell = std::pair<int, int>;  // (i, j) in [0, SIZE) x [0, SIZE)

template <int SIZE> class Hash {
 protected:
  unsigned long long _array[2][SIZE][SIZE];
  unsigned long long _turn;
  unsigned long long _value;

 public:
  Hash();
  void init();
  void updateArray(int Color, int j, int i);
  void updateTurn();
  unsigned long long getValue() const;
};

struct PathInfo {

  // Minimal index, in a path array, to a connected path.
  // Can index itself.
  int _mainPathIndex;

  Color _color;

  bool _isConnectedBorder1;
  bool _isConnectedBorder2;

  PathInfo() = default;

  PathInfo(int index, Color color, bool border1, bool border2)
    : _mainPathIndex(index)
    , _color(color)
    , _isConnectedBorder1(border1)
    , _isConnectedBorder2(border2) {
  }

};

template <int SIZE, bool PIE> class Board {

 protected:
  int _nbFullIndices;
  int _nbIndices;

  Color _currentColor;
  Color _winnerColor;
  bool _hasPie;

  std::optional<int> _lastIndex;
  int _nbEmptyIndices;

  // neighbours of each cell (indices)
  // end value: -1
  std::array<std::array<int, 7>, SIZE * SIZE> _neighboursBoard;

  // PathInfo of the paths indexed from _pathBoard
  int _pathsEnd;
  std::array<PathInfo, SIZE * SIZE> _paths;

  // path of each cell (index in _paths)
  std::array<int, SIZE * SIZE> _pathBoard;

  static inline Hash<SIZE> _hash;

 public:
  Board();
  bool canPie() const;

  Color getCurrentColor() const;
  Color getWinnerColor() const;
  PLAYER colorToPlayer(Color color) const;
  PLAYER getCurrentPlayer() const;
  PLAYER getWinnerPlayer() const;
  bool isGameFinished() const;
  std::optional<int> getLastIndex() const;

  static Cell convertIndexToCell(int index);
  static int convertCellToIndex(const Cell& refCell);

  unsigned long long getHashValue() const;

 protected:
  void getPathIndexAndColorAtIndex(int index,
                                    int& pathIndex,
                                    Color& color) const;

  ////////////////////////////////////////////////////////////
  // hex-specific
  ////////////////////////////////////////////////////////////

 public:
  void reset();
  void play(int index);

  bool isValidCell(const Cell& refCell) const;
  bool isValidIndex(int index) const;

  std::vector<int> findLegalIndices() const;
  std::vector<int> findWinnerPath() const;

 protected:
  void computeBorderConnection(int index,
                               Color color,
                               bool& isConnectedBorder1,
                               bool& isConnectedBorder2) const;
};

}  // namespace Hex

///////////////////////////////////////////////////////////////////////////////
// Hex::Hash
///////////////////////////////////////////////////////////////////////////////

template <int SIZE> Hex::Hash<SIZE>::Hash() {
  for (int color = 0; color < 2; color++)
    for (int j = 0; j < SIZE; j++)
      for (int i = 0; i < SIZE; i++) {
        _array[color][j][i] = 0;
        for (int k = 0; k < 64; k++)
          if ((rand() / (RAND_MAX + 1.0)) > 0.5)
            _array[color][j][i] |= (1ULL << k);
      }
  _turn = 0;
  for (int k = 0; k < 64; k++)
    if ((rand() / (RAND_MAX + 1.0)) > 0.5)
      _turn |= (1ULL << k);
}

template <int SIZE> void Hex::Hash<SIZE>::init() {
  _value = 0;
}

template <int SIZE>
void Hex::Hash<SIZE>::updateArray(int color, int j, int i) {
  _value ^= _array[color][j][i];
}

template <int SIZE> void Hex::Hash<SIZE>::updateTurn() {
  _value ^= _turn;
}

template <int SIZE> unsigned long long Hex::Hash<SIZE>::getValue() const {
  return _value;
}

///////////////////////////////////////////////////////////////////////////////
// Hex::Board
///////////////////////////////////////////////////////////////////////////////

template <int SIZE, bool PIE> Hex::Board<SIZE, PIE>::Board() {
  _hash.init();
}

template <int SIZE, bool PIE> Hex::Color Hex::Board<SIZE, PIE>::getCurrentColor() const {
  return _currentColor;
}

template <int SIZE, bool PIE> Hex::Color Hex::Board<SIZE, PIE>::getWinnerColor() const {
  return _winnerColor;
}

template <int SIZE, bool PIE> PLAYER Hex::Board<SIZE, PIE>::colorToPlayer(Color color) const {
  if (color == COLOR_NONE)
    return PLAYER_NULL;
  else if (color == COLOR_BLACK)
    return _hasPie ? PLAYER_1 : PLAYER_0;
  else 
    return _hasPie ? PLAYER_0 : PLAYER_1;
}

template <int SIZE, bool PIE> PLAYER Hex::Board<SIZE, PIE>::getCurrentPlayer() const {
  return colorToPlayer(_currentColor);
}

template <int SIZE, bool PIE> PLAYER Hex::Board<SIZE, PIE>::getWinnerPlayer() const {
  return colorToPlayer(_winnerColor);
}

template <int SIZE, bool PIE> bool Hex::Board<SIZE, PIE>::isGameFinished() const {
  return _nbEmptyIndices == 0 or _winnerColor != COLOR_NONE;
}

template <int SIZE, bool PIE> std::optional<int> Hex::Board<SIZE, PIE>::getLastIndex() const {
  return _lastIndex;
}

template <int SIZE, bool PIE> bool Hex::Board<SIZE, PIE>::canPie() const {
  return PIE and _nbEmptyIndices == _nbIndices - 1 and not _hasPie;
}

template <int SIZE, bool PIE> Hex::Cell Hex::Board<SIZE, PIE>::convertIndexToCell(int index) {
  int i = index / SIZE;
  int j = index % SIZE;
  return Cell(i, j);
}

template <int SIZE, bool PIE>
int Hex::Board<SIZE, PIE>::convertCellToIndex(const Cell& refCell) {
  return refCell.first * SIZE + refCell.second;
}

template <int SIZE, bool PIE> unsigned long long Hex::Board<SIZE, PIE>::getHashValue() const {
  return _hash.getValue();
}

template <int SIZE, bool PIE>
void Hex::Board<SIZE, PIE>::getPathIndexAndColorAtIndex(int index,
                                                    int& pathIndex,
                                                    Color& color) const {
  assert(index >= 0);
  assert(index < _nbFullIndices);

  // get path index from board
  pathIndex = _pathBoard[index];

  assert(pathIndex >= 0);
  assert(pathIndex < _pathsEnd);

  // get color from paths
  color = _paths[pathIndex]._color;
}

////////////////////////////////////////////////////////////
// hex-specific
////////////////////////////////////////////////////////////

template <int SIZE, bool PIE> void Hex::Board<SIZE, PIE>::reset() {

  _nbFullIndices = SIZE * SIZE;
  _nbIndices = _nbFullIndices;
  _nbEmptyIndices = _nbIndices;

  _currentColor = COLOR_BLACK;
  _winnerColor = COLOR_NONE;
  _hasPie = false;

  _lastIndex.reset();

  // _neighboursBoard
  // precompute all valid neighbours of each cell
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      int index = convertCellToIndex(Cell(i, j));
      int k = 0;
      std::array<Cell, 6> neighbours = {{Cell(i - 1, j), Cell(i - 1, j + 1),
                                         Cell(i, j - 1), Cell(i, j + 1),
                                         Cell(i + 1, j - 1), Cell(i + 1, j)}};
      for (const Cell& refCell : neighbours) {
        if (isValidCell(refCell)) {
          _neighboursBoard[index][k] = convertCellToIndex(refCell);
          k++;
        }
      }
      _neighboursBoard[index][k] = -1;
    }
  }

  // _paths
  // no initial path
  // path 0 for empty cells
  _paths[0] = PathInfo(0, COLOR_NONE, false, false);
  _pathsEnd = 1;

  // _pathBoard
  // set all cells to 0 (empty path)
  _pathBoard.fill(0);
}

template <int SIZE, bool PIE> void Hex::Board<SIZE, PIE>::play(int index) {
  assert(isValidIndex(index));
  assert(not isGameFinished());

  if (_lastIndex and index == *_lastIndex) {
    assert(canPie());
    _hasPie = true;

    /*
    // TODO player or color in hash ?
    Cell cell = convertIndexToCell(index);
    _hash.updateArray(_currentPlayer, cell.second, cell.first);
    _hash.updateTurn();
     */

  }
  else {
    assert(_pathBoard[index] == 0);

    // find previous path & cell at index
    int boardPathIndex;
    Color boardColor;
    getPathIndexAndColorAtIndex(index, boardPathIndex, boardColor);

    // if board cell is empty, update board
    if (boardColor == COLOR_NONE) {

      // update hash
      int color = getCurrentColor() == COLOR_BLACK ? 0 : 1;
      Cell cell = convertIndexToCell(index);
      _hash.updateArray(color, cell.second, cell.first);
      _hash.updateTurn();

      // cell data
      int mainPathIndex = _pathsEnd;
      bool isConnectedBorder1, isConnectedBorder2;
      computeBorderConnection(
          index, _currentColor, isConnectedBorder1, isConnectedBorder2);

      // find all connected paths
      std::set<int> neighbourMainPathIndices;
      for (int neighbourIndex : _neighboursBoard[index]) {
        if (neighbourIndex == -1)
          break;
        int neighbourPathIndex;
        Color neighbourColor;
        getPathIndexAndColorAtIndex(
            neighbourIndex, neighbourPathIndex, neighbourColor);
        if (neighbourColor == _currentColor) {
          int neighbourMain = _paths[neighbourPathIndex]._mainPathIndex;
          const PathInfo& neighbourPath = _paths[neighbourMain];
          // add neigbour in set
          neighbourMainPathIndices.insert(neighbourMain);
          // update cell data
          isConnectedBorder1 |= neighbourPath._isConnectedBorder1;
          isConnectedBorder2 |= neighbourPath._isConnectedBorder2;
          mainPathIndex = std::min(mainPathIndex, neighbourMain);
        }
      }

      // if the cell is not connected to any existing path, then create a
      // new path
      if (neighbourMainPathIndices.empty()) {
        _paths[_pathsEnd] = PathInfo(
            _pathsEnd, _currentColor, isConnectedBorder1, isConnectedBorder2);
        _pathsEnd++;
      }
      // if the cell is connected to an existing path, then update paths
      // and check end of game
      else {
        // update main path
        PathInfo& mainPath = _paths[mainPathIndex];
        mainPath._isConnectedBorder1 |= isConnectedBorder1;
        mainPath._isConnectedBorder2 |= isConnectedBorder2;

        // update other paths
        neighbourMainPathIndices.erase(mainPathIndex);
        if (not neighbourMainPathIndices.empty()) {
          for (int k = mainPathIndex + 1; k < _pathsEnd; k++) {
            int mainK = _paths[k]._mainPathIndex;
            auto iter = neighbourMainPathIndices.find(mainK);
            if (iter != neighbourMainPathIndices.end())
              _paths[k] = mainPath;
          }
        }

        // update winner
        if (mainPath._isConnectedBorder1 and mainPath._isConnectedBorder2)
          _winnerColor = _currentColor;
      }

      // end turn and prepare for next one
      _pathBoard[index] = mainPathIndex;
      _nbEmptyIndices--;
      _lastIndex = index;
      _currentColor = _currentColor == COLOR_BLACK ? COLOR_WHITE : COLOR_BLACK;
    }
  }
}

template <int SIZE, bool PIE>
bool Hex::Board<SIZE, PIE>::isValidCell(const Cell& refCell) const {
  return refCell.first >= 0 and refCell.first < SIZE 
    and refCell.second >= 0 and refCell.second < SIZE;
}

template <int SIZE, bool PIE> bool Hex::Board<SIZE, PIE>::isValidIndex(int index) const {
  return (index >= 0 and index < _nbFullIndices);
}

template <int SIZE, bool PIE>
std::vector<int> Hex::Board<SIZE, PIE>::findLegalIndices() const {
  std::vector<int> emptyIndices;
  emptyIndices.reserve(_nbEmptyIndices+1);
  for (int k = 0; k < _nbFullIndices; k++)
    if (_pathBoard[k] == 0)
      emptyIndices.push_back(k);
  if (canPie())
    emptyIndices.push_back(*_lastIndex);
  return emptyIndices;
}


template <int SIZE, bool PIE> std::vector<int> Hex::Board<SIZE, PIE>::findWinnerPath() const {
  assert(_winnerColor != COLOR_NONE);

  // find winning main path index
  int winPathIndex = 1;
  while (not _paths[winPathIndex]._isConnectedBorder1 or
         not _paths[winPathIndex]._isConnectedBorder2)
    winPathIndex++;

  assert(_paths[winPathIndex]._color == _winnerColor);

  // find all indices connected to main path
  std::vector<int> winIndices;
  winIndices.reserve(2 * SIZE);
  for (int k = 0; k < _nbFullIndices; k++) {
    int pathIndexOfK = _pathBoard[k];
    int mainPathIndexOfK = _paths[pathIndexOfK]._mainPathIndex;
    if (mainPathIndexOfK == winPathIndex)
      winIndices.push_back(k);
  }

  return winIndices;
}


template <int SIZE, bool PIE>
void Hex::Board<SIZE, PIE>::computeBorderConnection(int index,
                                                    Color color,
                                                    bool& isConnectedBorder1,
                                                    bool& isConnectedBorder2) const {
     
  if (color == COLOR_BLACK) {
    isConnectedBorder1 = (index < SIZE);
    isConnectedBorder2 = (index >= _nbFullIndices - SIZE);
  } else if (color == COLOR_WHITE) {
    int j = index % SIZE;
    isConnectedBorder1 = (j == 0);
    isConnectedBorder2 = (j == SIZE - 1);
  } else {
    isConnectedBorder1 = false;
    isConnectedBorder2 = false;
  }
}

