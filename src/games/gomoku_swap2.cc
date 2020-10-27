/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gomoku_swap2.h"

namespace GomokuSwap2 {

void Game::initGame() {
  board.initialize();
  hands = 0;
  winner = -1;
}

void Game::play(Move& m) {
  assert(hands < maxHands);
  hands += 1;

  if (!m.isColorChanged) {
    for (int x = 0; x < boardRadix; x++) {
      for (int y = 0; y < boardRadix; y++) {
        Chess chess = board.getChess(x, y);
        if (chess == Chesses::white) {
          board.setChess(x, y, Chesses::black);
        } else if (chess == Chesses::black) {
          board.setChess(x, y, Chesses::white);
        }
      }
    }
  }

  board.setChess(m.x, m.y, m.chess);
  board.turnHash();
}

bool Game::isWon(Move& m) {
  Chess chess = board.getChess(m.x, m.y);
  auto suck = [&](int dx, int dy) {
    int count = 1;
    int x = m.x + dx, y = m.y + dy;
    while (board.getChess(x, y) == chess) {
      count += 1;
      x += dx;
      y += dy;
      if (Board::isPosInBoard(x, y))
        break;
    }
    x = m.x - dx, y = m.y - dy;
    while (board.getChess(x, y) == chess) {
      count += 1;
      x -= dx;
      y -= dy;
      if (Board::isPosInBoard(x, y))
        break;
    }
    return count;
  };

  if (suck(0, 1) >= 5)
    return true;
  if (suck(1, 0) >= 5)
    return true;
  if (suck(1, 1) >= 5)
    return true;
  if (suck(1, -1) >= 5)
    return true;
  return false;
}

void Game::findLegalMoves(Player player) {
  legalMovesCnt = 0;
  Chess chess = playerToChess(player);
  for (int x = 0; x < boardRadix; x++) {
    for (int y = 0; y < boardRadix; y++) {
      Move m = Move{x, y, chess, false};

      if (board.getChess(x, y) == Chesses::empty) {
        assert(legalMovesCnt < maxLegalMovesCnt);
        legalMoves[legalMovesCnt] = m;
        legalMovesCnt += 1;
        if (hands == 3 || (hands == 5 && !isTurned)) {
          m.isColorChanged = true;
          assert(legalMovesCnt < maxLegalMovesCnt);
          legalMoves[legalMovesCnt] = m;
          legalMovesCnt += 1;
        }
      }
    }
  }
}

Player Game::chessToPlayer(Chess chess) {
  if (chess == Chesses::black)
    return isTurned ? Players::player1 : Players::player0;
  else if (chess == Chesses::white)
    return isTurned ? Players::player0 : Players::player1;
  else
    assert(chess == Chesses::black || chess == Chesses::white);
  return -1;
}

Chess Game::playerToChess(Player player) {
  if (player == Players::player0)
    return isTurned ? Chesses::white : Chesses::black;
  else if (player == Players::player1)
    return isTurned ? Chesses::black : Chesses::white;
  else
    assert(player == Players::player0 || player == Players::player1);
  return 0xFF;
}

template <typename R> void Game::setupBoard(const R& re) {
  Board::setup({"Empty", "Black", "White"}, {" ", "●", "○"}, re);
}

State::State(int seed)
    : core::State(seed)
    , Game() {
  std::call_once(setupCalled, [&] { setupBoard(_rng); });
}

void State::Initialize() {
  _moves.clear();

  _featSize[0] = featuresSizeX;
  _featSize[1] = featuresSizeY;
  _featSize[2] = featuresSizeZ;

  _actionSize[0] = 2;
  _actionSize[1] = boardRadix;
  _actionSize[2] = boardRadix;

  _status = GameStatus::player0Turn;
  _features.resize(featuresSize);

  initGame();
  findLegalMoves(Players::player0);
  findActions();
  findFeatures();
  _hash = board.getHash();
}

unique_ptr<core::State> State::clone_() const {
  return make_unique<State>(*this);
}

void State::ApplyAction(const _Action& action) {
  if (!terminated()) {
    Move m{};
    Player nextPlayer = -1;
    m.isColorChanged = action.GetX();
    if (m.isColorChanged)
      isTurned = true;
    if (_status == GameStatus::player0Turn) {
      m.chess = playerToChess(Players::player0);
      nextPlayer = Players::player1;
      _status = GameStatus::player1Turn;
    } else if (_status == GameStatus::player1Turn) {
      m.chess = playerToChess(Players::player1);
      nextPlayer = Players::player0;
      _status = GameStatus::player0Turn;
    }
    m.x = action.GetY();
    m.y = action.GetZ();
    play(m);
    if (canGoNext(m)) {
      findLegalMoves(nextPlayer);
      findActions();
      findFeatures();
    }
  }
  _hash = board.getHash();
}

bool State::canGoNext(Move& m) {
  if (isWon(m)) {
    Player winner = chessToPlayer(m.chess);
    if (winner == Players::player0)
      _status = GameStatus::player0Win;
    else if (winner == Players::player1)
      _status = GameStatus::player1Win;
    else
      assert(winner == Players::player0 || winner == Players::player1);
    return false;
  }
  return true;
}

void State::DoGoodAction() {
  DoRandomAction();
}

void State::findActions() {
  clearActions();
  for (int i = 0; i < legalMovesCnt; i++) {
    Move& m = legalMoves[i];
    addAction(m.isColorChanged, m.x, m.y);
  }
}

void State::findFeatures() {
  std::fill(_features.begin(), _features.end(), 0.0);
  auto* f = _features.data();
  for (int c = 0; c < this->chesses; c++) {
    Chess chess = static_cast<Chess>(c + 1);
    for (int i = 0; i < this->boardSize; i++) {
      if (this->board.getChess(i) == chess)
        *f = 1.0;
      f++;
    }
  }
  fillFullFeatures();
}

}  // namespace GomokuSwap2
