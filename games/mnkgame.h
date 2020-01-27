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

namespace MNKGame {

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

template <int M, int N, int K> class State : public ::State {
  static_assert(M > 0 && N > 0 && K > 0, "m, n, and k must be greater then 0");
  static_assert(K <= M || K <= N, "k must be less than or equal to m or n");

 public:
  using Board = Chessboard<M, N>;

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
  template <typename RE> static void setupBoard(const RE& re);
  static constexpr Player chessToPlayer(Chess chess);
  static constexpr Chess playerToChess(Player player);

  virtual void play(const Move& move);
  std::optional<Player> findWinner(const Move& move);
  bool isConnected(const Move& move);
  virtual void findLegalActions();
  inline Player getCurrentPlayer();
  inline void turnPlayer();
  inline void setTerminatedStatus(Player winner);
  void fillFeatures();

  static constexpr int chessKinds = 2;
  static constexpr int connections = K;
  static constexpr int maxLegalActionsCnt = Board::squares;
  static constexpr std::tuple<int, int> directions[4] = {
      {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  static inline std::once_flag setupCalled;

  Board board;
  std::bitset<Board::squares> areEmpty;
};

template <int M, int N, int K> class Action : public ::_Action {
 public:
  Action(int i, int x, int y);
};

template <int M, int N, int K>
State<M, N, K>::State(int seed)
    : ::State(seed) {
  std::call_once(setupCalled, [&] { setupBoard(_rng); });
}

template <int M, int N, int K> void State<M, N, K>::Initialize() {
  _moves.clear();
  _featSize = {chessKinds, Board::rows, Board::columns};
  _features.resize(chessKinds * Board::squares);
  _actionSize = {1, Board::rows, Board::columns};
  _legalActions.reserve(maxLegalActionsCnt);
  _status = GameStatus::player0Turn;

  board.initialize();
  areEmpty.set();
  _hash = board.getHash();
  findLegalActions();
  fillFeatures();
}

template <int M, int N, int K>
std::unique_ptr<mcts::State> State<M, N, K>::clone_() const {
  return std::make_unique<State>(*this);
}

template <int M, int N, int K>
void State<M, N, K>::ApplyAction(const ::_Action& action) {
  Move move{};
  move.chess = playerToChess(getCurrentPlayer());
  move.x = action.GetY();
  move.y = action.GetZ();
  play(move);
  board.turnHash();
  _hash = board.getHash();
  if (auto hasWinner = findWinner(move); !hasWinner) {
    turnPlayer();
    findLegalActions();
    fillFeatures();
  } else {
    setTerminatedStatus(hasWinner.value());
  }
}

template <int M, int N, int K> void State<M, N, K>::DoGoodAction() {
  DoRandomAction();
}

template <int M, int N, int K> void State<M, N, K>::printCurrentBoard() const {
  std::cout << board.sprint("  ");
}

template <int M, int N, int K>
std::string State<M, N, K>::stateDescription() const {
  return board.sprint("  ");
}

template <int M, int N, int K>
std::string State<M, N, K>::actionDescription(const ::_Action& action) const {
  std::ostringstream oss;
  oss << "put a chess at " << board.getPosStr(action.GetY(), action.GetZ());
  return oss.str();
}

template <int M, int N, int K>
std::string State<M, N, K>::actionsDescription() {
  std::set<std::tuple<int, int>> markedPos;
  for (auto& legalAction : _legalActions)
    markedPos.insert({legalAction->GetY(), legalAction->GetZ()});
  return board.sprintBoard("  ", markedPos);
}

template <int M, int N, int K>
int State<M, N, K>::parseAction(const std::string& str) {
  auto result = board.parsePosStr(str);
  if (!result)
    return -1;
  auto [x, y] = result.value();
  int i = 0;
  for (auto& legalAction : _legalActions) {
    if (legalAction->GetY() == x && legalAction->GetZ() == y)
      return i;
    i++;
  }
  return -1;
}

template <int M, int N, int K>
int State<M, N, K>::humanInputAction(
    std::function<std::optional<int>(std::string)> specialAction) {
  std::cout << "Current board:" << std::endl << stateDescription() << std::endl;
  std::cout << "Allowed positions: ('" << Board::getMarkSymbol()
            << "' means an allowed position)" << std::endl
            << actionsDescription() << std::endl;
  std::cout << "Input a position to play: (uses format <alphabet of x-axis>"
            << "<numbers of y-axis>, e.g. `A1`, `b2`, `C03`...)" << std::endl;
  std::string str;
  int index = -1;
  while (index < 0) {
    std::cout << "> ";
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

template <int M, int N, int K>
template <typename RE>
void State<M, N, K>::setupBoard(const RE& re) {
  Board::setup({"Empty", "Black", "White"}, {" ", "●", "○"}, re);
}

template <int M, int N, int K>
constexpr Player State<M, N, K>::chessToPlayer(Chess chess) {
  if (chess == ChessKind::black)
    return Player::first;
  else if (chess == ChessKind::white)
    return Player::second;
  assert(chess == ChessKind::black || chess == ChessKind::white);
  return Player::none;
}

template <int M, int N, int K>
constexpr Chess State<M, N, K>::playerToChess(Player player) {
  if (player == Player::first)
    return ChessKind::black;
  else if (player == Player::second)
    return ChessKind::white;
  assert(player == Player::first || player == Player::second);
  return ChessKind::empty;
}

template <int M, int N, int K> void State<M, N, K>::play(const Move& move) {
  board.setChess(move.x, move.y, move.chess);
  areEmpty.reset(Board::posTo1D(move.x, move.y));
}

template <int M, int N, int K>
std::optional<Player> State<M, N, K>::findWinner(const Move& move) {
  if (isConnected(move))
    return chessToPlayer(move.chess);
  else if (areEmpty.none())
    return Player::none;
  return std::nullopt;
}

template <int M, int N, int K>
bool State<M, N, K>::isConnected(const Move& move) {
  if (connections == 1)
    return true;
  auto areChessesEnough = [&](int& count, int dx, int dy) {
    int x = move.x + dx, y = move.y + dy;
    while (Board::isPosInBoard(x, y)) {
      if (board.getChess(x, y) == move.chess) {
        if (++count == connections)
          return true;
      } else {
        return false;
      }
      x += dx, y += dy;
    }
    return false;
  };
  for (auto [dx, dy] : directions) {
    int count = 1;
    if (areChessesEnough(count, dx, dy) || areChessesEnough(count, -dx, -dy))
      return true;
  }
  return false;
}

template <int M, int N, int K> void State<M, N, K>::findLegalActions() {
  _legalActions.clear();
  int i = 0;
  for (int xy = 0; xy < Board::squares; xy++) {
    if (areEmpty[xy]) {
      auto [x, y] = Board::posTo2D(xy);
      _legalActions.push_back(std::make_shared<Action<N, M, K>>(i++, x, y));
      assert(i <= maxLegalActionsCnt);
    }
  }
}

template <int M, int N, int K> Player State<M, N, K>::getCurrentPlayer() {
  if (_status == GameStatus::player0Turn)
    return Player::first;
  else if (_status == GameStatus::player1Turn)
    return Player::second;
  assert(_status == GameStatus::player0Turn ||
         _status == GameStatus::player1Turn);
  return Player::none;
}

template <int M, int N, int K> void State<M, N, K>::turnPlayer() {
  if (_status == GameStatus::player0Turn)
    _status = GameStatus::player1Turn;
  else if (_status == GameStatus::player1Turn)
    _status = GameStatus::player0Turn;
  assert(_status == GameStatus::player0Turn ||
         _status == GameStatus::player1Turn);
}

template <int M, int N, int K>
void State<M, N, K>::setTerminatedStatus(Player winner) {
  if (winner == Player::first)
    _status = GameStatus::player0Win;
  else if (winner == Player::second)
    _status = GameStatus::player1Win;
  else
    _status = GameStatus::tie;
}

template <int M, int N, int K> void State<M, N, K>::fillFeatures() {
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

template <int M, int N, int K>
Action<M, N, K>::Action(int i, int x, int y)
    : ::_Action() {
  _i = i;
  _loc = {0, x, y};
  _hash = State<M, N, K>::Board::rows * y + x;
}

}  // namespace MNKGame
