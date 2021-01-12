/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../core/state.h"
#include "havannah.h"

#include <algorithm>
#include <chrono>

namespace Havannah {

template <int SIZE> class Action : public ::_Action {
 public:
  Action(int i, int j, int indexInActions);
};

template <int SIZE, bool PIE, bool EXTENDED> class State : public core::State {
 private:
  Board<SIZE, PIE> _board;

 public:
  State(int seed);
  // State(int seed, int history, bool turnFeatures);
  void findActions();
  void Initialize() override;
  void ApplyAction(const _Action& action) override;
  void DoGoodAction() override;
  std::unique_ptr<core::State> clone_() const override;
  std::string stateDescription() const override;
  std::string actionDescription(const _Action& action) const override;
  std::string actionsDescription() const override;
  int parseAction(const std::string& str) const override;
  virtual int getCurrentPlayerColor() const override;
};
}  // namespace Havannah

///////////////////////////////////////////////////////////////////////////////
// Havannah::Action
///////////////////////////////////////////////////////////////////////////////

template <int SIZE>
Havannah::Action<SIZE>::Action(int i, int j, int indexInActions) {
  _loc[0] = 0;
  _loc[1] = i;
  _loc[2] = j;
  _hash = uint32_t(i * fullsize(SIZE) + j);
  _i = indexInActions;  // (position in _legalActions)
}

///////////////////////////////////////////////////////////////////////////////
// Havannah::State
///////////////////////////////////////////////////////////////////////////////

template <int SIZE, bool PIE, bool EXTENDED>
Havannah::State<SIZE, PIE, EXTENDED>::State(int seed)
    : core::State(seed) {
}

template <int SIZE, bool PIE, bool EXTENDED>
void Havannah::State<SIZE, PIE, EXTENDED>::findActions() {
  auto legalIndices = _board.findLegalIndices();
  clearActions();
  for (unsigned k = 0; k < legalIndices.size(); ++k) {
    auto c = _board.convertIndexToCell(legalIndices[k]);
    addAction(0, c.first, c.second);
  }
}

template <int SIZE, bool PIE, bool EXTENDED>
void Havannah::State<SIZE, PIE, EXTENDED>::Initialize() {
  _board.reset();
  _moves.clear();
  _hash = 0;
  _status = GameStatus::player0Turn;

  // features
  _featSize = {EXTENDED ? 27 : 3, fullsize(SIZE), fullsize(SIZE)};
  _features =
      std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2], 0.f);

  for (int k = 0; k < fullsize(SIZE) * fullsize(SIZE); k++)
    _features[2 * fullsize(SIZE) * fullsize(SIZE) + k] = _board.isValidIndex(k);

  fillFullFeatures();

  // actions
  _actionSize = {1, fullsize(SIZE), fullsize(SIZE)};
  findActions();
}

template <int SIZE, bool PIE, bool EXTENDED>
void Havannah::State<SIZE, PIE, EXTENDED>::ApplyAction(const _Action& action) {

  assert(not _board.isGameFinished());

  // find board move from action
  int i = action.GetY();
  int j = action.GetZ();
  int index = _board.convertCellToIndex(Cell(i, j));
  std::optional<int> lastIndex = _board.getLastIndex();

  // update features
  if (not lastIndex or *lastIndex != index) {
    Color currentColor = _board.getCurrentColor();
    _features[((currentColor * fullsize(SIZE)) + i) * fullsize(SIZE) + j] = 1.f;
  }

  // add connections to borders/corners
  if (EXTENDED) {
    const int fs2 = fullsize(SIZE) * fullsize(SIZE);
    const unsigned mask = 1;
    for (int k = 0; k < fs2; k++) {
      if (_board.isValidIndex(k)) {

        int iPath = _board._pathBoard[k];
        if (iPath != 0) {
          assert(iPath < _board._pathsEnd);
          const auto& pathInfo = _board._paths[iPath];

          unsigned borders = pathInfo._borders;
          Color color = _board.getColorAtIndex(k);
          for (int iBorder = 0; iBorder < 6; iBorder++) {
            _features[(2 * iBorder + color + 3) * fs2 + k] =
                (borders >> iBorder) & mask;
          }

          unsigned corners = pathInfo._corners;
          for (int iCorner = 0; iCorner < 6; iCorner++) {
            _features[(2 * iCorner + 12 + color + 3) * fs2 + k] =
                (corners >> iCorner) & mask;
          }
        }
      }
    }
  }

  // play move
  _board.play(index);

  // update game status
  if (_board.isGameFinished()) {
    PLAYER winner = _board.getWinnerPlayer();
    if (winner == PLAYER_0)
      _status = GameStatus::player0Win;
    else if (winner == PLAYER_1)
      _status = GameStatus::player1Win;
    else
      _status = GameStatus::tie;
  } else {
    _status = _board.getCurrentPlayer() == PLAYER_0 ? GameStatus::player0Turn
                                                    : GameStatus::player1Turn;
  }

  fillFullFeatures();
  // update actions
  findActions();

  // update hash
  _hash = _board.getHashValue();
}

template <int SIZE, bool PIE, bool EXTENDED>
void Havannah::State<SIZE, PIE, EXTENDED>::DoGoodAction() {
  return DoRandomAction();
}

template <int SIZE, bool PIE, bool EXTENDED>
std::unique_ptr<core::State> Havannah::State<SIZE, PIE, EXTENDED>::clone_()
    const {
  return std::make_unique<Havannah::State<SIZE, PIE, EXTENDED>>(*this);
}

template <int SIZE, bool PIE, bool EXTENDED>
std::string Havannah::State<SIZE, PIE, EXTENDED>::stateDescription() const {

  const auto& feats = _features;
  const auto& sizes = _featSize;

  int ni = sizes[1];
  int nj = sizes[2];

  auto ind = [ni, nj](int i, int j, int k) { return (k * ni + i) * nj + j; };

  std::string str;

  str += "Havannah\n";

  str += "  ";
  for (int k = 0; k < ni; k++) {
    str += " ";
    if (k < 10)
      str += " ";
    str += std::to_string(k) + " ";
  }
  str += "\n";

  for (int i = 0; i < ni; i++) {

    str += "   ";
    for (int k = 0; k < i; k++)
      str += "  ";

    for (int k = 0; k < SIZE - i - 1; k++)
      str += "    ";
    for (int k = 0; k < SIZE + i and k < 3 * SIZE - i - 1; k++)
      str += "----";
    str += "\n";

    if (i < 10)
      str += " ";
    str += std::to_string(i) + " ";
    for (int k = 0; k < i; k++)
      str += "  ";
    for (int j = 0; j < nj; j++) {

      if (_board.isValidCell({i, j}))
        str += "\\ ";
      else if (j < SIZE)
        str += "  ";

      if (feats[ind(i, j, 0)] && feats[ind(i, j, 1)])
        str += "! ";
      else if (feats[ind(i, j, 0)])
        str += "X ";
      else if (feats[ind(i, j, 1)])
        str += "O ";
      else if (_board.isValidCell({i, j}))
        str += ". ";
      else if (j < SIZE)
        str += "  ";
      else
        continue;
    }

    str += "\\ \n";
  }

  str += "  ";
  for (int k = 0; k < ni; k++)
    str += "  ";
  for (int k = SIZE - 2; _board.isValidCell({SIZE, k}); k++)
    str += "----";
  str += "\n";

  str += "    ";
  for (int k = 0; k < SIZE - 1; k++)
    str += "    ";
  for (int k = 0; k < ni; k++) {
    str += " ";
    if (k < 10)
      str += " ";
    str += std::to_string(k) + " ";
  }
  str += "\n";

  return str;
}

template <int SIZE, bool PIE, bool EXTENDED>
std::string Havannah::State<SIZE, PIE, EXTENDED>::actionDescription(
    const _Action& action) const {
  return std::to_string(action.GetY()) + "," + std::to_string(action.GetZ());
}

template <int SIZE, bool PIE, bool EXTENDED>
std::string Havannah::State<SIZE, PIE, EXTENDED>::actionsDescription() const {
  std::ostringstream oss;
  for (const auto& a : _legalActions) {
    oss << a.GetY() << "," << a.GetZ() << " ";
  }
  oss << std::endl;
  return oss.str();
}

template <int SIZE, bool PIE, bool EXTENDED>
int Havannah::State<SIZE, PIE, EXTENDED>::parseAction(
    const std::string& str) const {
  std::istringstream iss(str);
  try {
    std::string token;
    if (not std::getline(iss, token, ','))
      throw - 1;
    int i = std::stoi(token);
    if (not std::getline(iss, token))
      throw - 1;
    int j = std::stoi(token);
    for (unsigned k = 0; k < _legalActions.size(); k++)
      if (_legalActions[k].GetY() == i and _legalActions[k].GetZ() == j)
        return k;
  } catch (...) {
    std::cout << "failed to parse action" << std::endl;
  }
  return -1;
}

template <int SIZE, bool PIE, bool EXTENDED>
int Havannah ::State<SIZE, PIE, EXTENDED>::getCurrentPlayerColor() const {
  return _board.getCurrentColor();
}
