/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: 林鈺錦 (Yù-Jǐn Lín)
// - Github: https://github.com/abc1236762
// - Email:  abc1236762@outlook.com
// Facilitator: 邱顯棟 (Xiǎn-Dòng Qiū)
// - Github: https://github.com/YumJelly
// - Email:  yumjelly@gmail.com

#pragma once

#include <algorithm>
#include <bitset>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

#include "../core/state.h"
#include "commons/chessboard.h"
#include "commons/player.h"

namespace Othello {

class ChessKind {
 public:
  static constexpr Chess empty = 0;
  static constexpr Chess black = 1;
  static constexpr Chess white = 2;
};

struct Move {
  Chess chess;
  int x, y;
};

template <int BR> class State : public ::State {
  static_assert(BR >= 4 && BR % 2 == 0,
                "radix of board must be greater than or equal to 4 and even");

 public:
  using Board = Chessboard<BR, BR, false>;

  State(int seed);
  void Initialize() override;
  std::unique_ptr<mcts::State> clone_() const override;
  void ApplyAction(const ::_Action& action) override;
  void DoGoodAction() override;
  void printCurrentBoard() const override;
  std::string stateDescription() const override;
  std::string actionDescription(const ::_Action& action) const override;
  std::string actionsDescription() override;
  int parseAction(const std::string& str) override;
  int humanInputAction(
      std::function<std::optional<int>(std::string)> specialAction) override;

 private:
  template <typename R> static void setupBoard(const R& re);
  static constexpr Player chessToPlayer(Chess chess);
  static constexpr Chess playerToChess(Player player);

  void setInitialChesses();
  void play(const Move& move);
  bool canGoNext(Player nextPlayer, bool isPassMove);
  Player findWinner();
  void findLegalActions(Player player);
  int countReverseChesses(const Move& move, int dx, int dy);
  bool canDoReverse(const Move& move);
  void doReverse(const Move& move);
  inline Player getCurrentPlayer();
  inline Player turnPlayer();
  inline void setTerminatedStatus(Player winner);
  void fillFeatures();

  static constexpr int players = 2;
  static constexpr int chessKinds = 2;
  static constexpr int maxLegalActionsCnt = Board::squares - 4;
  static constexpr std::tuple<int, int> directions[8] = {
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  static constexpr std::tuple<int, int> initialChessesPos[players][chessKinds] =
      {{{Board::rows / 2 - 1, Board::columns / 2},
        {Board::rows / 2, Board::columns / 2 - 1}},
       {{Board::rows / 2 - 1, Board::columns / 2 - 1},
        {Board::rows / 2, Board::columns / 2}}};
  static inline std::once_flag setupCalled;

  Board board;
  std::bitset<Board::squares> areEmpty, candi;
};

//template <int BR> class Action : public ::_Action {
// public:
//  Action(int i, int x, int y, bool isPassMove);
//};

template <int BR>
State<BR>::State(int seed)
    : ::State(seed) {
  std::call_once(setupCalled, [&] { setupBoard(_rng); });
}

template <int BR> void State<BR>::Initialize() {
  _moves.clear();
  _featSize = {chessKinds, Board::rows, Board::columns};
  _features.resize(chessKinds * Board::squares);
  _actionSize = {2, Board::rows, Board::columns};
  _NewlegalActions.reserve(maxLegalActionsCnt);
  _status = GameStatus::player0Turn;

  board.initialize();
  areEmpty.set();
  candi.reset();
  setInitialChesses();
  _hash = board.getHash();
  findLegalActions(Player::first);
  fillFeatures();
}

template <int BR> std::unique_ptr<mcts::State> State<BR>::clone_() const {
  return std::make_unique<State>(*this);
}

template <int BR> void State<BR>::ApplyAction(const ::_Action& action) {
  bool isPassMove = action.GetX();
  if (!isPassMove) {
    Move move{};
    move.chess = playerToChess(getCurrentPlayer());
    move.x = action.GetY();
    move.y = action.GetZ();
    play(move);
  }
  board.turnHash();
  _hash = board.getHash();
  Player nextPlayer = turnPlayer();
  if (canGoNext(nextPlayer, isPassMove)) {
    if (_NewlegalActions.size() == 0) {
      _NewlegalActions.emplace_back(_NewlegalActions.size(), 1, Board::rows / 2, Board::columns / 2);
    }
    fillFeatures();
  } else {
    Player winner = findWinner();
    setTerminatedStatus(winner);
    _NewlegalActions.clear();
  }
}

template <int BR> void State<BR>::DoGoodAction() {
  DoRandomAction();
}

template <int BR> void State<BR>::printCurrentBoard() const {
  std::cout << board.sprint("  ");
}

template <int BR> std::string State<BR>::stateDescription() const {
  return board.sprint("  ");
}

template <int BR>
std::string State<BR>::actionDescription(const ::_Action& action) const {
  bool isPassMove = (bool)action.GetX();
  if (isPassMove)
    return "passed";
  std::ostringstream oss;
  oss << board.getPosStr(action.GetY(), action.GetZ());
  return oss.str();
}

template <int BR> std::string State<BR>::actionsDescription() {
  std::set<std::tuple<int, int>> markedPos;
  if (_NewlegalActions.size() >= 1 && _NewlegalActions[0].GetX() == 0)
    for (auto& legalAction : _NewlegalActions)
      markedPos.insert({legalAction.GetY(), legalAction.GetZ()});
  return board.sprintBoard("  ", markedPos);
}

template <int BR> int State<BR>::parseAction(const std::string& str) {
  if (_NewlegalActions.size() == 1 && _NewlegalActions[0].GetX() == 1)
    return 0;
  auto result = board.parsePosStr(str);
  if (!result)
    return -1;
  auto [x, y] = result.value();
  int i = 0;
  for (auto& legalAction : _NewlegalActions) {
    if (legalAction.GetY() == x && legalAction.GetZ() == y)
      return i;
    i++;
  }
  return -1;
}

template <int BR>
int State<BR>::humanInputAction(
    std::function<std::optional<int>(std::string)> specialAction) {
  std::cout << "Current board:" << std::endl << stateDescription() << std::endl;
  if (_NewlegalActions.size() == 1 && _NewlegalActions[0].GetX() == 1) {
    std::cout << "No positions to play." << std::endl;
    std::cout << "Input nothing to pass." << std::endl;
  } else {
    std::cout << "Allowed positions: ('" << Board::getMarkSymbol()
              << "' means an allowed position)" << std::endl
              << actionsDescription() << std::endl;
    std::cout << "Input a position to play: (uses format <alphabet of x-axis>"
              << "<numbers of y-axis>, e.g. `A1`, `b2`, `C03`...)" << std::endl;
  }
  std::string str;
  int index = -1;
  while (index < 0) {
    std::cout << "Input action: ";
    std::getline(std::cin, str);
    index = parseAction(str);
    if (index < 0) {
      if (auto r = specialAction(str); r)
        return *r;
      std::cout << "invalid input, try again." << std::endl;
    }
  }
  return index;
}

template <int BR>
template <typename R>
void State<BR>::setupBoard(const R& re) {
  Board::setup({"Empty", "Black", "White"}, {" ", "●", "○"}, re);
}

template <int BR> constexpr Player State<BR>::chessToPlayer(Chess chess) {
  if (chess == ChessKind::black)
    return Player::first;
  else if (chess == ChessKind::white)
    return Player::second;
  assert(chess == ChessKind::black || chess == ChessKind::white);
  return Player::none;
}

template <int BR> constexpr Chess State<BR>::playerToChess(Player player) {
  if (player == Player::first)
    return ChessKind::black;
  else if (player == Player::second)
    return ChessKind::white;
  assert(player == Player::first || player == Player::second);
  return ChessKind::empty;
}

template <int BR> void State<BR>::setInitialChesses() {
  for (int p = 0; p < players; p++) {
    Chess chess = playerToChess(Player::set(p));
    for (auto [x, y] : initialChessesPos[p]) {
      board.setChess(x, y, chess);
      areEmpty.reset(Board::posTo1D(x, y));
    }
  }
  for (int y = Board::columns / 2 - 2; y < Board::columns / 2 + 2; y++)
    for (int x = Board::rows / 2 - 2; x < Board::rows / 2 + 2; x++)
      candi.set(Board::posTo1D(x, y));
}

template <int BR> void State<BR>::play(const Move& move) {
  board.setChess(move.x, move.y, move.chess);
  areEmpty.reset(Board::posTo1D(move.x, move.y));
  doReverse(move);
}

template <int BR>
bool State<BR>::canGoNext(Player nextPlayer, bool isPassMove) {
  if (areEmpty.none())
    return false;
  findLegalActions(nextPlayer);
  return _NewlegalActions.size() != 0 || !isPassMove;
}

template <int BR> Player State<BR>::findWinner() {
  auto counts = board.countChesses();
  if (counts[ChessKind::black] > counts[ChessKind::white])
    return Player::first;
  else if (counts[ChessKind::black] < counts[ChessKind::white])
    return Player::second;
  return Player::none;
}

template <int BR> void State<BR>::findLegalActions(Player player) {
  _NewlegalActions.clear();
  auto possibles = areEmpty & candi;
  Chess chess = playerToChess(player);
  int i = 0;
  for (int xy = 0; xy < Board::squares; xy++) {
    if (possibles[xy]) {
      auto [x, y] = Board::posTo2D(xy);
      Move legalMove = Move{chess, x, y};
      if (canDoReverse(legalMove)) {
        _NewlegalActions.emplace_back(i++, 0, legalMove.x, legalMove.y);
        assert(i <= maxLegalActionsCnt);
      }
    }
  }
}

template <int BR>
int State<BR>::countReverseChesses(const Move& move, int dx, int dy) {
  int x = move.x + dx, y = move.y + dy, chessCnt = 0;
  while (Board::isPosInBoard(x, y)) {
    if (Chess chess = board.getChess(x, y); chess == ChessKind::empty)
      return 0;
    else if (chess != move.chess)
      chessCnt++;
    else
      return chessCnt;
    x += dx, y += dy;
  }
  return 0;
}

template <int BR> bool State<BR>::canDoReverse(const Move& move) {
  for (auto [dx, dy] : directions)
    if (countReverseChesses(move, dx, dy) > 0)
      return true;
  return false;
}

template <int BR> void State<BR>::doReverse(const Move& move) {
  for (auto [dx, dy] : directions) {
    int x = move.x + dx, y = move.y + dy;
    if (Board::isPosInBoard(x, y)) {
      candi.set(Board::posTo1D(x, y));
      int chessCnt = countReverseChesses(move, dx, dy);
      for (int j = 0; j < chessCnt; j++) {
        board.setChess(x, y, move.chess);
        x += dx, y += dy;
      }
    }
  }
}

template <int BR> Player State<BR>::getCurrentPlayer() {
  if (_status == GameStatus::player0Turn)
    return Player::first;
  else if (_status == GameStatus::player1Turn)
    return Player::second;
  assert(_status == GameStatus::player0Turn ||
         _status == GameStatus::player1Turn);
  return Player::none;
}

template <int BR> Player State<BR>::turnPlayer() {
  if (_status == GameStatus::player0Turn) {
    _status = GameStatus::player1Turn;
    return Player::second;
  } else if (_status == GameStatus::player1Turn) {
    _status = GameStatus::player0Turn;
    return Player::first;
  }
  assert(_status == GameStatus::player0Turn ||
         _status == GameStatus::player1Turn);
  return Player::none;
}

template <int BR> void State<BR>::setTerminatedStatus(Player winner) {
  if (winner == Player::first)
    _status = GameStatus::player0Win;
  else if (winner == Player::second)
    _status = GameStatus::player1Win;
  else
    _status = GameStatus::tie;
}

template <int BR> void State<BR>::fillFeatures() {
  std::fill(_features.begin(), _features.end(), 0.0);
  auto* f = _features.data();
  for (int c = 0; c < chessKinds; c++) {
    Chess chess = static_cast<Chess>(c + 1);
    for (int xy = 0; xy < Board::squares; xy++, f++)
      if (board.getChess(xy) == chess)
        *f = 1.0;
  }
  fillFullFeatures();
}

//template <int BR>
//Action<BR>::Action(int i, int x, int y, bool isPassMove)
//    : ::_Action() {
//  _i = i;
//  _loc = {isPassMove, x, y};
//  _hash =
//      isPassMove ? State<BR>::Board::squares : State<BR>::Board::rows * y + x;
//}

}  // namespace Othello
