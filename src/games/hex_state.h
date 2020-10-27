/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../core/state.h"
#include "hex.h"

#include <algorithm>
#include <chrono>

namespace Hex {

template <int SIZE, bool PIE> class State : public core::State {
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
  virtual int getNumPlayerColors() const override;
};
}  // namespace Hex

///////////////////////////////////////////////////////////////////////////////
// Hex::State
///////////////////////////////////////////////////////////////////////////////

template <int SIZE, bool PIE>
Hex::State<SIZE, PIE>::State(int seed)
    : core::State(seed) {
}

template <int SIZE, bool PIE> void Hex::State<SIZE, PIE>::findActions() {
  auto legalIndices = _board.findLegalIndices();
  clearActions();
  for (unsigned k = 0; k < legalIndices.size(); ++k) {
    auto c = _board.convertIndexToCell(legalIndices[k]);
    addAction(0, c.first, c.second);
  }
}

template <int SIZE, bool PIE> void Hex::State<SIZE, PIE>::Initialize() {
  _board.reset();
  _moves.clear();
  _hash = 0;
  _status = GameStatus::player0Turn;

  // features
  _featSize = {2, SIZE, SIZE};
  _features =
      std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2], 0.f);
  fillFullFeatures();

  // actions
  _actionSize = {1, SIZE, SIZE};
  findActions();
}

template <int SIZE, bool PIE>
void Hex::State<SIZE, PIE>::ApplyAction(const _Action& action) {

  assert(not _board.isGameFinished());

  // find board move from action
  int i = action.GetY();
  int j = action.GetZ();
  int index = _board.convertCellToIndex(Cell(i, j));
  std::optional<int> lastIndex = _board.getLastIndex();

  // update features
  // TODO assert action is in legal actions ?
  if (not lastIndex or *lastIndex != index) {
    Color currentColor = _board.getCurrentColor();
    _features[((currentColor * SIZE) + i) * SIZE + j] = 1.f;
  }

  // play move
  _board.play(index);

  // update game status
  if (_board.isGameFinished()) {
    PLAYER winner = _board.getWinnerPlayer();
    assert(winner == PLAYER_0 or winner == PLAYER_1);
    _status =
        winner == PLAYER_0 ? GameStatus::player0Win : GameStatus::player1Win;
  } else {
    PLAYER player = _board.getCurrentPlayer();
    assert(player == PLAYER_0 or player == PLAYER_1);
    _status =
        player == PLAYER_0 ? GameStatus::player0Turn : GameStatus::player1Turn;
  }

  fillFullFeatures();
  // update actions
  findActions();

  // update hash
  _hash = _board.getHashValue();
}

template <int SIZE, bool PIE> void Hex::State<SIZE, PIE>::DoGoodAction() {
  return DoRandomAction();
}

template <int SIZE, bool PIE>
std::unique_ptr<core::State> Hex::State<SIZE, PIE>::clone_() const {
  return std::make_unique<Hex::State<SIZE, PIE>>(*this);
}

template <int SIZE, bool PIE>
std::string Hex::State<SIZE, PIE>::stateDescription() const {

  const auto& feats = _features;
  const auto& sizes = _featSize;
  int ni = sizes[1];
  int nj = sizes[2];
  assert(ni <= 26);

  auto ind = [ni, nj](int i, int j, int k) { return (k * ni + i) * nj + j; };

  std::string str;

  str += "Hex\n";
  str += " ";
  for (int k = 0; k < nj; k++) {
    str += "   ";
    str += 'a' + k;
  }
  str += "\n";

  for (int i = 0; i < ni; i++) {

    str += "  ";
    for (int k = 0; k < i; k++)
      str += "  ";
    str += "-";
    for (int k = 0; k < nj; k++)
      str += "----";
    str += "\n";

    if (i < 9)
      str += " ";
    str += std::to_string(1 + i) + " ";
    for (int k = 0; k < i; k++)
      str += "  ";
    for (int j = 0; j < nj; j++) {
      str += "\\ ";
      if (feats[ind(i, j, 0)] && feats[ind(i, j, 1)])
        str += "! ";
      else if (feats[ind(i, j, 0)])
        str += "B ";
      else if (feats[ind(i, j, 1)])
        str += "W ";
      else
        str += ". ";
    }
    str += "\\ \n";
  }

  str += "  ";
  for (int k = 0; k < nj; k++)
    str += "  ";
  for (int k = 0; k < nj; k++)
    str += "----";
  str += "\n";

  str += "   ";
  for (int k = 0; k < SIZE - 1; k++)
    str += "  ";
  for (int k = 0; k < nj; k++) {
    str += "   ";
    str += 'a' + k;
  }
  str += "\n";

  return str;
}

template <int SIZE, bool PIE>
std::string Hex::State<SIZE, PIE>::actionDescription(
    const _Action& action) const {
  return char('a' + action.GetZ()) + std::to_string(1 + action.GetY());
}

template <int SIZE, bool PIE>
std::string Hex::State<SIZE, PIE>::actionsDescription() const {
  std::ostringstream oss;
  for (const auto& a : _legalActions) {
    oss << actionDescription(a) << " ";
  }
  oss << std::endl;
  return oss.str();
}

template <int SIZE, bool PIE>
int Hex::State<SIZE, PIE>::parseAction(const std::string& str) const {
  std::istringstream iss(str);
  try {
    char c;
    iss >> c;
    std::string token;
    int j = int(c) - 'a';
    if (not std::getline(iss, token))
      throw - 1;
    int i = std::stoi(token) - 1;
    for (unsigned k = 0; k < _legalActions.size(); k++)
      if (_legalActions[k].GetY() == i and _legalActions[k].GetZ() == j)
        return k;
  } catch (...) {
    std::cout << "failed to parse action" << std::endl;
  }
  return -1;
}

template <int SIZE, bool PIE>
int Hex::State<SIZE, PIE>::getCurrentPlayerColor() const {
  return _board.getCurrentColor();
}

template <int SIZE, bool PIE>
int Hex::State<SIZE, PIE>::getNumPlayerColors() const {
  return 2;
}
