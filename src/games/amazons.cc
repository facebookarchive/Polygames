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

#include "amazons.h"

namespace Amazons {

State::State(int seed)
    : ::State(seed) {
  std::call_once(setupCalled, [&] { setupBoard(_rng); });
}

void State::Initialize() {
  _moves.clear();
  _featSize = {chessKinds, Board::rows, Board::columns};
  _features.resize(chessKinds * Board::squares);
  _actionSize = {Board::squares, Board::squares, Board::squares};
  _legalActions.reserve(maxLegalActionsCnt);
  _status = GameStatus::player0Turn;

  board.initialize();
  setInitialChesses();
  _hash = board.getHash();
  findLegalActions(Player::first);
  fillFeatures();
}

std::unique_ptr<mcts::State> State::clone_() const {
  return std::make_unique<State>(*this);
}

void State::ApplyAction(const ::_Action& action) {
  Move move{};
  std::tie(move.fromX, move.fromY) = Board::posTo2D(action.GetX());
  std::tie(move.toX, move.toY) = Board::posTo2D(action.GetY());
  std::tie(move.arrowX, move.arrowY) = Board::posTo2D(action.GetZ());
  play(move);
  board.turnHash();
  _hash = board.getHash();
  Player nextPlayer = turnPlayer();
  if (canGoNext(nextPlayer)) {
    fillFeatures();
  } else {
    setTerminatedStatus(nextPlayer);
    _legalActions.clear();
  }
}

void State::DoGoodAction() {
  DoRandomAction();
}

void State::printCurrentBoard() const {
  std::cout << board.sprint("  ");
}

std::string State::stateDescription() const {
  return board.sprint("  ");
}

std::string State::actionDescription(const ::_Action& action) const {
  std::ostringstream oss;
  oss << "moved the chess from " << board.getPosStr(action.GetX()) << " to "
      << board.getPosStr(action.GetY()) << " and shooted an arrow at "
      << board.getPosStr(action.GetZ());
  return oss.str();
}

std::string State::actionsDescription() {
  std::ostringstream oss;
  oss << "Position of queen chesses which are able to move: ";
  std::set<int> xySet;
  for (auto& legalAction : _legalActions)
    xySet.insert(legalAction->GetX());
  std::size_t i = 0;
  for (int xy : xySet) {
    oss << "`" << board.getPosStr(xy) << "`";
    if (++i < xySet.size())
      oss << ", ";
  }
  oss << std::endl;
  return oss.str();
}

int State::parseAction(const std::string& str) {
  auto getXY = [&](const std::set<int>& xySet, const std::string& content) {
    std::cout << content << std::endl
              << "Allowed positions: ('" << Board::getMarkSymbol()
              << "' means an allowed position)";
    std::set<std::tuple<int, int>> markedPos;
    for (auto xy : xySet)
      markedPos.insert(Board::posTo2D(xy));
    std::cout << std::endl << board.sprintBoard("  ", markedPos);
    std::cout << "> ";
    std::string str;
    std::getline(std::cin, str);
    auto xyResult = board.parsePosStr(str);
    if (!xyResult)
      return -1;
    auto [x, y] = xyResult.value();
    int xy = Board::posTo1D(x, y);
    if (xySet.find(xy) == xySet.end())
      return -1;
    return xy;
  };

  auto result = board.parsePosStr(str);
  if (!result)
    return -1;
  auto [fromX, fromY] = result.value();
  int fromXY = Board::posTo1D(fromX, fromY);
  std::set<int> toXYSet;
  std::unordered_map<int, std::set<int>> arrowXYSet;
  std::unordered_map<int, std::unordered_map<int, int>> iMap;
  for (std::size_t i = 0; i < _legalActions.size(); i++) {
    if (fromXY != _legalActions[i]->GetX())
      continue;
    int toXY = _legalActions[i]->GetY();
    int arrowXY = _legalActions[i]->GetZ();
    toXYSet.insert(toXY);
    arrowXYSet[toXY].insert(arrowXY);
    iMap[toXY][arrowXY] = i;
  }
  if (toXYSet.empty())
    return -1;

  int toXY =
      getXY(toXYSet, "Input the position of selected queen chess after moved:");
  if (toXY < 0)
    return -1;
  int arrowXY = getXY(
      arrowXYSet[toXY], "Input the position of the arrow that wants to shoot:");
  if (arrowXY < 0)
    return -1;
  return iMap[toXY][arrowXY];
}

int State::humanInputAction(
    std::function<std::optional<int>(std::string)> specialAction) {
  std::cout << "Current board:" << std::endl << stateDescription() << std::endl;
  std::cout << "Input three positions to play: (uses format <alphabet of x-axi"
            << "s><numbers of y-axis>, e.g. `A1`, `b2`, `C03`...)" << std::endl;
  std::string str;
  int index = -1;
  while (index < 0) {
    std::cout << actionsDescription() << std::endl;
    std::cout << "Input the position of the queen chess which wants to move:";
    std::cout << std::endl << "> ";
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

template <typename R> void State::setupBoard(const R& re) {
  Board::setup(
      {"Empty", "WhiteQueen", "BlackQueen", "WhiteArrow", "BlackArrow"},
      {" ", "○", "●", "□", "■"}, re);
}

constexpr Player State::chessToPlayer(Chess chess) {
  if (chess == ChessKind::whiteQueen || chess == ChessKind::whiteArrow)
    return Player::first;
  else if (chess == ChessKind::blackQueen || chess == ChessKind::blackArrow)
    return Player::second;
  assert(chess == ChessKind::whiteQueen || chess == ChessKind::whiteArrow ||
         chess == ChessKind::blackQueen || chess == ChessKind::blackArrow);
  return Player::none;
}

constexpr Chess State::playerToQueenChess(Player player) {
  if (player == Player::first)
    return ChessKind::whiteQueen;
  else if (player == Player::second)
    return ChessKind::blackQueen;
  assert(player == Player::first || player == Player::second);
  return ChessKind::empty;
}

constexpr Chess State::playerToArrowChess(Player player) {
  if (player == Player::first)
    return ChessKind::whiteArrow;
  else if (player == Player::second)
    return ChessKind::blackArrow;
  assert(player == Player::first || player == Player::second);
  return ChessKind::empty;
}

void State::setInitialChesses() {
  for (int p = 0; p < players; p++) {
    Chess queenChess = playerToQueenChess(Player::set(p));
    int i = 0;
    for (auto [x, y] : initialQueenChessesPos[p]) {
      board.setChess(x, y, queenChess);
      queenChessesPos[p][i++] = {x, y};
    }
  }
}

void State::play(const Move& move) {
  Chess queenChess = board.getChess(move.fromX, move.fromY);
  Player player = chessToPlayer(queenChess);
  assert(queenChess == ChessKind::whiteQueen ||
         queenChess == ChessKind::blackQueen);
  for (auto& [x, y] : queenChessesPos[player.index()]) {
    if (x == move.fromX && y == move.fromY) {
      x = move.toX, y = move.toY;
      break;
    }
  }
  board.setChess(move.fromX, move.fromY, ChessKind::empty);
  board.setChess(move.toX, move.toY, queenChess);
  board.setChess(move.arrowX, move.arrowY, playerToArrowChess(player));
}

bool State::canGoNext(Player nextPlayer) {
  findLegalActions(nextPlayer);
  return _legalActions.size() != 0;
}

void State::findLegalActions(Player player) {
  _legalActions.clear();
  int i = 0;

  auto addLegalActions = [&](Move move) {
    int fromXY = Board::posTo1D(move.fromX, move.fromY);
    int toXY = Board::posTo1D(move.toX, move.toY);
    int arrowXY = Board::posTo1D(move.arrowX, move.arrowY);
    _legalActions.push_back(
        std::make_shared<Action>(i++, fromXY, toXY, arrowXY));
    assert(i <= maxLegalActionsCnt);
  };

  auto findLegalArrowShoots = [&](int fromX, int fromY, int toX, int toY) {
    for (auto [dx, dy] : directions) {
      int arrowX = toX + dx, arrowY = toY + dy;
      while (Board::isPosInBoard(arrowX, arrowY)) {
        if (board.getChess(arrowX, arrowY) == ChessKind::empty ||
            (arrowX == fromX && arrowY == fromY)) {
          addLegalActions(Move{fromX, fromY, toX, toY, arrowX, arrowY});
        } else {
          break;
        }
        arrowX += dx, arrowY += dy;
      }
    }
  };

  auto findLegalQueenMoves = [&](int fromX, int fromY) {
    for (auto [dx, dy] : directions) {
      int toX = fromX + dx, toY = fromY + dy;
      while (Board::isPosInBoard(toX, toY) &&
             board.getChess(toX, toY) == ChessKind::empty) {
        findLegalArrowShoots(fromX, fromY, toX, toY);
        toX += dx, toY += dy;
      }
    }
  };

  for (auto [fromX, fromY] : queenChessesPos[player.index()])
    findLegalQueenMoves(fromX, fromY);
}

Player State::turnPlayer() {
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

void State::setTerminatedStatus(Player loser) {
  if (loser == Player::first)
    _status = GameStatus::player1Win;
  else if (loser == Player::second)
    _status = GameStatus::player0Win;
  assert(loser == Player::first || loser == Player::second);
}

void State::fillFeatures() {
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

Action::Action(int i, int fromXY, int toXY, int arrowXY)
    : ::_Action() {
  _i = i;
  _loc = {fromXY, toXY, arrowXY};
  _hash = State::Board::squares * State::Board::squares * fromXY +
          State::Board::squares * toXY + arrowXY;
}

}  // namespace Amazons
