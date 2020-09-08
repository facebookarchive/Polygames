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
#include <cstring>
#include <optional>
#include <random>
#include <set>

namespace Havannah {

// For Havannah, SIZE is the edge size of the board. FULLSIZE is 2*SIZE - 1
// and the number of playable cells is FULLSIZE - SIZE*(SIZE-1).

enum Color { COLOR_BLACK, COLOR_WHITE, COLOR_NONE };

using Cell = std::pair<int, int>;  // (i, j) in [0, FULLSIZE) x [0, FULLSIZE)

constexpr int fullsize(int size) {
  return 2 * size - 1;
}

template <int SIZE> class Hash {
 protected:
  unsigned long long _array[2][fullsize(SIZE)][fullsize(SIZE)];
  unsigned long long _turn;
  unsigned long long _value;

 public:
  Hash();
  void init();
  void updateArray(int color, int j, int i);
  void updateTurn();
  unsigned long long getValue() const;
};

struct PathInfo {

  // Minimal index, in a path array, to a connected path.
  // Can index itself.
  int _mainPathIndex;

  Color _color;

  unsigned _borders;
  unsigned _corners;

  PathInfo() = default;

  PathInfo(int index, Color color, unsigned borders, unsigned corners)
    : _mainPathIndex(index)
    , _color(color)
    , _borders(borders)
    , _corners(corners) {
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
  std::array<std::array<int, 7>, fullsize(SIZE) * fullsize(SIZE)>
      _neighboursBoard;

  static inline Hash<SIZE> _hash;

 public: // TODO getter ?

  // PathInfo of the paths indexed from _pathBoard
  int _pathsEnd;
  std::array<PathInfo, fullsize(SIZE) * fullsize(SIZE)> _paths;

  // path of each cell (index in _paths)
  std::array<int, fullsize(SIZE) * fullsize(SIZE)> _pathBoard;

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

  Color getColorAtIndex(int index) const;

 protected:
  void getPathIndexAndColorAtIndex(int index,
                                    int& pathIndex,
                                    Color& color) const;

  ////////////////////////////////////////////////////////////
  // Havannah-specific
  ////////////////////////////////////////////////////////////

 protected:
  int _winningCycle;

 public:
  void reset();
  void play(int index);

  bool isValidCell(const Cell& refCell) const;
  bool isValidIndex(int index) const;

  std::vector<int> findLegalIndices() const;
  std::vector<int> findWinnerPath() const;

 protected:
  unsigned computeBorders(int index) const;
  unsigned computeCorners(int index) const;

  bool isWinningPath(const PathInfo& path, int pathIndex, int cellIndex);
  // isCycle: assume _paths[cellIndex] == pathIndex
  bool isCycle(int pathIndex, int cellIndex) const;
  int computeNbOnes(unsigned f) const;
  std::vector<int> findPathIndices(int pathIndex) const;
  int computeNbNeighbours(int cellIndex, Color color) const;
  bool detectHole(const std::vector<int>& indices) const;
};

}  // namespace Havannah

///////////////////////////////////////////////////////////////////////////////
// Havannah::Hash
///////////////////////////////////////////////////////////////////////////////

template <int SIZE> Havannah::Hash<SIZE>::Hash() {
  for (int color = 0; color < 2; color++)
    for (int j = 0; j < fullsize(SIZE); j++)
      for (int i = 0; i < fullsize(SIZE); i++) {
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

template <int SIZE> void Havannah::Hash<SIZE>::init() {
  _value = 0;
}

template <int SIZE>
void Havannah::Hash<SIZE>::updateArray(int color, int j, int i) {
  _value ^= _array[color][j][i];
}

template <int SIZE> void Havannah::Hash<SIZE>::updateTurn() {
  _value ^= _turn;
}

template <int SIZE> unsigned long long Havannah::Hash<SIZE>::getValue() const {
  return _value;
}

///////////////////////////////////////////////////////////////////////////////
// Havannah::Board
///////////////////////////////////////////////////////////////////////////////

template <int SIZE, bool PIE> Havannah::Board<SIZE, PIE>::Board() {
  _hash.init();
}


template <int SIZE, bool PIE> Havannah::Color Havannah::Board<SIZE, PIE>::getCurrentColor() const {
  return _currentColor;
}

template <int SIZE, bool PIE> Havannah::Color Havannah::Board<SIZE, PIE>::getWinnerColor() const {
  return _winnerColor;
}

template <int SIZE, bool PIE> PLAYER Havannah::Board<SIZE, PIE>::colorToPlayer(Color color) const {
  if (color == COLOR_NONE)
    return PLAYER_NULL;
  else if (color == COLOR_BLACK)
    return _hasPie ? PLAYER_1 : PLAYER_0;
  else 
    return _hasPie ? PLAYER_0 : PLAYER_1;
}

template <int SIZE, bool PIE> PLAYER Havannah::Board<SIZE, PIE>::getCurrentPlayer() const {
  return colorToPlayer(_currentColor);
}

template <int SIZE, bool PIE> PLAYER Havannah::Board<SIZE, PIE>::getWinnerPlayer() const {
  return colorToPlayer(_winnerColor);
}

template <int SIZE, bool PIE> bool Havannah::Board<SIZE, PIE>::isGameFinished() const {
  return _nbEmptyIndices == 0 or _winnerColor != COLOR_NONE;
}

template <int SIZE, bool PIE> std::optional<int> Havannah::Board<SIZE, PIE>::getLastIndex() const {
  return _lastIndex;
}

template <int SIZE, bool PIE> bool Havannah::Board<SIZE, PIE>::canPie() const {
  return PIE and _nbEmptyIndices == _nbIndices - 1 and not _hasPie;
}

template <int SIZE, bool PIE>
Havannah::Cell Havannah::Board<SIZE, PIE>::convertIndexToCell(int index) {
  int i = index / fullsize(SIZE);
  int j = index % fullsize(SIZE);
  return Cell(i, j);
}

template <int SIZE, bool PIE>
int Havannah::Board<SIZE, PIE>::convertCellToIndex(const Cell& refCell) {
  return refCell.first * fullsize(SIZE) + refCell.second;
}

template <int SIZE, bool PIE>
unsigned long long Havannah::Board<SIZE, PIE>::getHashValue() const {
  return _hash.getValue();
}

template <int SIZE, bool PIE>
void Havannah::Board<SIZE, PIE>::getPathIndexAndColorAtIndex(int index,
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

template <int SIZE, bool PIE>
Havannah::Color Havannah::Board<SIZE, PIE>::getColorAtIndex(int index) const {
  int pathIndex;
  Color color;
  getPathIndexAndColorAtIndex(index, pathIndex, color);
  return color;
}

////////////////////////////////////////////////////////////
// Havannah-specific
////////////////////////////////////////////////////////////

template <int SIZE, bool PIE> void Havannah::Board<SIZE, PIE>::reset() {

  _nbFullIndices =
      fullsize(SIZE) * fullsize(SIZE);  // nb indices for full hex board
  _nbIndices = _nbFullIndices - (SIZE - 1) * SIZE;
  _nbEmptyIndices = _nbIndices;  // nb indices for havannah board

  _winningCycle = 0;

  _currentColor = COLOR_BLACK;
  _winnerColor = COLOR_NONE;
  _hasPie = false;

  _lastIndex.reset();

  // _neighboursBoard
  // precompute all valid neighbours of each cell
  for (int i = 0; i < fullsize(SIZE); i++) {
    for (int j = 0; j < fullsize(SIZE); j++) {
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
  _paths[0] = PathInfo(0, COLOR_NONE, 0, 0);
  _pathsEnd = 1;

  // _pathBoard
  // set all cells to 0 (empty path)
  _pathBoard.fill(0);
}

template <int SIZE, bool PIE> void Havannah::Board<SIZE, PIE>::play(int index) {

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
      int borders = computeBorders(index);
      int corners = computeCorners(index);

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
          borders |= neighbourPath._borders;
          corners |= neighbourPath._corners;
          mainPathIndex = std::min(mainPathIndex, neighbourMain);
        }
      }

      // if the cell is not connected to any existing path, then create a
      // new path
      if (neighbourMainPathIndices.empty()) {
        _paths[_pathsEnd] =
            PathInfo(_pathsEnd, _currentColor, borders, corners);
        _pathsEnd++;
      _pathBoard[index] = mainPathIndex;
      }
      // if the cell is connected to an existing path, then update paths
      // and check end of game
      else {
        // update main path
        PathInfo& mainPath = _paths[mainPathIndex];
        mainPath._borders |= borders;
        mainPath._corners |= corners;

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
      _pathBoard[index] = mainPathIndex;
  
        // update winner
        if (isWinningPath(mainPath, mainPathIndex, index))
          _winnerColor = _currentColor;
      }

      // end turn and prepare for next one
      _nbEmptyIndices--;
      _lastIndex = index;
      _currentColor = _currentColor == COLOR_BLACK ? COLOR_WHITE : COLOR_BLACK;
    }
  }
}

template <int SIZE, bool PIE>
bool Havannah::Board<SIZE, PIE>::isValidCell(const Cell& refCell) const {
  int i = refCell.first;
  int j = refCell.second;
  return i >= 0 and i < fullsize(SIZE) and j >= 0 and j < fullsize(SIZE) and
         i + j >= SIZE - 1 and i + j <= 3 * SIZE - 3;
}

template <int SIZE, bool PIE> bool Havannah::Board<SIZE, PIE>::isValidIndex(int index) const {
  Cell c = convertIndexToCell(index);
  return isValidCell(c);
}

template <int SIZE, bool PIE>
std::vector<int> Havannah::Board<SIZE, PIE>::findLegalIndices() const {
  std::vector<int> emptyIndices;
  emptyIndices.reserve(_nbEmptyIndices+1);
  for (int k = 0; k < _nbFullIndices; k++)
    if (isValidIndex(k) and _pathBoard[k] == 0)
      emptyIndices.push_back(k);
  if (canPie())
    emptyIndices.push_back(*_lastIndex);
  return emptyIndices;
}

template <int SIZE, bool PIE>
std::vector<int> Havannah::Board<SIZE, PIE>::findWinnerPath() const {

  assert(_winnerColor != COLOR_NONE);

  // find winning path
  int winPathIndex;
  if (_winningCycle != 0)
    winPathIndex = _winningCycle;
  else {
    winPathIndex = 1;
    while (true) {
      const PathInfo& path = _paths[winPathIndex];
      if (computeNbOnes(path._borders) >= 3)
        break;
      if (computeNbOnes(path._corners) >= 2)
        break;
      winPathIndex++;
    }
  }

  assert(_paths[winPathIndex]._color == _winnerColor);

  // find all indices connected to winning path
  return findPathIndices(winPathIndex);
}

template <int SIZE, bool PIE>
bool Havannah::Board<SIZE, PIE>::isWinningPath(const PathInfo& path,
                                          int pathIndex,
                                          int cellIndex) {

  // test if path is connected to 3 borders
  if (computeNbOnes(path._borders) >= 3) {
    return true;
  }

  // test if path is connected to 2 corners
  if (computeNbOnes(path._corners) >= 2) {
    return true;
  }

  // test if path is a cycle
  if (isCycle(pathIndex, cellIndex)) {
    _winningCycle = pathIndex;
    return true;
  }

  return false;
}

template <int SIZE, bool PIE>
bool Havannah::Board<SIZE, PIE>::isCycle(int pathIndex, int cellIndex) const {

  Color currentColor = _paths[pathIndex]._color;

  // compute full path
  std::vector<int> indices = findPathIndices(pathIndex);

  // check if full path has 6 cells at least
  if (indices.size() < 6)
    return false;

  // check if cell is connected to two previous cells at least
  if (computeNbNeighbours(cellIndex, currentColor) < 2)
    return false;

  // detect interior point
  for (int index : indices)
    if (computeNbNeighbours(index, currentColor) == 6)
      return true;

  // detect hole
  return detectHole(indices);
}

template <int SIZE, bool PIE>
unsigned Havannah::Board<SIZE, PIE>::computeBorders(int index) const {
  unsigned borders = 0;
  Cell c = convertIndexToCell(index);
  if (isValidCell(c)) {
    int i = c.first;
    int j = c.second;
    int e1 = SIZE - 1;
    int s1 = fullsize(SIZE) - 1;
    if (i == 0 and e1 < j and j < s1)
      borders += 1;
    if (0 < i and i < e1 and j == s1)
      borders += 2;
    if (i + j == 3 * e1 and i < s1 and j < s1)
      borders += 4;
    if (i == s1 and 0 < j and j < e1)
      borders += 8;
    if (e1 < i and i < s1 and j == 0)
      borders += 16;
    if (i + j == e1 and i > 0 and j > 0)
      borders += 32;
  }
  return borders;
}

template <int SIZE, bool PIE>
unsigned Havannah::Board<SIZE, PIE>::computeCorners(int index) const {
  unsigned corners = 0;
  Cell c = convertIndexToCell(index);
  if (isValidCell(c)) {
    int i = c.first;
    int j = c.second;
    int e1 = SIZE - 1;
    int s1 = fullsize(SIZE) - 1;
    if (i == 0 and j == e1)
      corners += 1;
    if (i == 0 and j == s1)
      corners += 2;
    if (i == e1 and j == s1)
      corners += 4;
    if (i == s1 and j == e1)
      corners += 8;
    if (i == s1 and j == 0)
      corners += 16;
    if (i == e1 and j == 0)
      corners += 32;
  }
  return corners;
}

template <int SIZE, bool PIE> int Havannah::Board<SIZE, PIE>::computeNbOnes(unsigned f) const {
  int n = f & 1u;
  f >>= 1;
  n += f & 1u;
  f >>= 1;
  n += f & 1u;
  f >>= 1;
  n += f & 1u;
  f >>= 1;
  n += f & 1u;
  f >>= 1;
  n += f & 1u;
  return n;
}

template <int SIZE, bool PIE>
std::vector<int> Havannah::Board<SIZE, PIE>::findPathIndices(int pathIndex) const {
  std::vector<int> indices;
  indices.reserve(2 * fullsize(SIZE));
  for (int k = 0; k < _nbFullIndices; k++) {
    int pathIndexOfK = _pathBoard[k];
    int mainPathIndexOfK = _paths[pathIndexOfK]._mainPathIndex;
    if (mainPathIndexOfK == pathIndex)
      indices.push_back(k);
  }
  return indices;
}

template <int SIZE, bool PIE>
int Havannah::Board<SIZE, PIE>::computeNbNeighbours(int cellIndex,
                                               Color color) const {
  int nbNeighbours = 0;
  for (int neighbourIndex : _neighboursBoard[cellIndex]) {
    if (neighbourIndex == -1)
      break;
    if (getColorAtIndex(neighbourIndex) == color)
      nbNeighbours++;
  }
  return nbNeighbours;
}

template <int SIZE, bool PIE>
bool Havannah::Board<SIZE, PIE>::detectHole(const std::vector<int>& indices) const {

  std::vector<Cell> cells;
  cells.reserve(indices.size());

  for (int i : indices)
    cells.emplace_back(convertIndexToCell(i));

  // find bounding box
  int imin = fullsize(SIZE);
  int jmin = fullsize(SIZE);
  int imax = 0;
  int jmax = 0;
  for (const Cell& c : cells) {
    imin = std::min(imin, c.first);
    imax = std::max(imax, c.first);
    jmin = std::min(jmin, c.second);
    jmax = std::max(jmax, c.second);
  }

  // reset data
  int data[fullsize(SIZE) + 2][fullsize(SIZE) + 2];
  std::memset((void*)data, 0,
              sizeof(int) * (fullsize(SIZE) + 2) * (fullsize(SIZE) + 2));
  int di = imax - imin + 3;
  int dj = jmax - jmin + 3;
  for (int i = 0; i < di; i++) {
    data[i][0] = 1;
    data[i][dj - 1] = 1;
  }
  for (int j = 0; j < dj; j++) {
    data[0][j] = 1;
    data[di - 1][j] = 1;
  }

  // write object
  for (const Cell& c : cells) {
    int i = c.first - imin + 1;
    int j = c.second - jmin + 1;
    data[i][j] = -1;
  }

  // propagate background
  auto fMaxNeighbour = [&data](int i, int j) {
    int d = data[i][j];
    if (d >= 0) {
      d = std::max(d, data[i - 1][j]);
      d = std::max(d, data[i - 1][j + 1]);
      d = std::max(d, data[i][j - 1]);
      d = std::max(d, data[i][j + 1]);
      d = std::max(d, data[i + 1][j - 1]);
      d = std::max(d, data[i + 1][j]);
    }
    return d;
  };

  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;

    for (int i = 1; i < di - 1; i++) {
      for (int j = 1; j < dj - 1; j++) {
        int d = fMaxNeighbour(i, j);
        if (data[i][j] != d) {
          data[i][j] = d;
          hasChanged = true;
        }
      }
    }

    for (int i = di - 2; i > 0; i--) {
      for (int j = dj - 2; j > 0; j--) {
        int d = fMaxNeighbour(i, j);
        if (data[i][j] != d) {
          data[i][j] = d;
          hasChanged = true;
        }
      }
    }
  }

  // check initial background
  for (int i = 0; i < di; i++)
    for (int j = 0; j < dj; j++)
      if (data[i][j] == 0)
        return true;

  return false;
}

