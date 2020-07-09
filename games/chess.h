/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../core/state.h"

namespace chess {

struct ChessBoard {

  static const size_t boardSize = 8;
  static const size_t boardDim = boardSize + 4;

  std::array<uint8_t, boardDim * boardDim> board;

  std::vector<uint32_t> moves;
  uint_fast32_t moveflags = 0;
  std::array<std::vector<std::pair<uint64_t, char>>, 16> repetitions;
  uint64_t hash = 0;

  static const uint32_t castleleft = 1u << 28;
  static const uint32_t castleright = 1u << 30;

  int turn = 0;

  static const uint8_t EMPTY = 0;
  static const uint8_t PAWN = 1;
  static const uint8_t KNIGHT = 2;
  static const uint8_t BISHOP = 4;
  static const uint8_t ROOK = 8;
  static const uint8_t QUEEN = 12;
  static const uint8_t KING = 3;
  static const uint8_t OOB = 0x80;

  static const uint8_t WHITE = 1 << 5;
  static const uint8_t BLACK = 2 << 5;

  void init();

  void findMoves();

  void move(uint_fast32_t move);

  std::string moveString(uint_fast32_t move) const;

  bool done = false;
  int winner = -1;
  int fiftyMoveCounter = 0;
};

class State : public ::State {
 public:
  State(int seed)
      : ::State(seed) {
  }

  ChessBoard board;

  std::vector<size_t> moves;

  static const int boardSize = 8;

  virtual void Initialize() override {
    _moves.clear();
    _hash = 2166136261u;
    _status = GameStatus::player0Turn;
    _featSize[0] = 12;
    _featSize[1] = boardSize;
    _featSize[2] = boardSize;
    _actionSize[0] = 6;
    _actionSize[1] = boardSize;
    _actionSize[2] = boardSize;
    _features.clear();
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
    std::fill(_features.begin(), _features.end(), 0.0f);
    board.init();
    board.findMoves();
    featurize();
    findActions();
    fillFullFeatures();
  }

  virtual std::unique_ptr<mcts::State> clone_() const override {
    return std::make_unique<State>(*this);
  }

  virtual std::string stateDescription() const override {
    std::string str;
    const size_t boardDim = 8 + 4;
    int y = 8;
    for (size_t iy = 2 + boardDim * 2;;) {
      str += '0' + y;
      --y;
      str += ' ';
      size_t ii = (boardDim - 1 - iy / boardDim) * boardDim + 2;
      for (size_t i = 0; i != 8; ++i, ++ii) {
        int v = board.board[ii];
        if (v & ChessBoard::WHITE) {
          switch (v & 0xf) {
          case ChessBoard::PAWN:
            str += 'P';
            break;
          case ChessBoard::KNIGHT:
            str += 'N';
            break;
          case ChessBoard::BISHOP:
            str += 'B';
            break;
          case ChessBoard::ROOK:
            str += 'R';
            break;
          case ChessBoard::QUEEN:
            str += 'Q';
            break;
          case ChessBoard::KING:
            str += 'K';
            break;
          default:
            str += '.';
          }
        } else {
          switch (v & 0xf) {
          case ChessBoard::PAWN:
            str += 'p';
            break;
          case ChessBoard::KNIGHT:
            str += 'n';
            break;
          case ChessBoard::BISHOP:
            str += 'b';
            break;
          case ChessBoard::ROOK:
            str += 'r';
            break;
          case ChessBoard::QUEEN:
            str += 'q';
            break;
          case ChessBoard::KING:
            str += 'k';
            break;
          default:
            str += '.';
          }
        }
        str += ' ';
      }
      str += '\n';
      iy += 8 + 2;
      if (iy == 12 + boardDim * 9) {
        break;
      }
      iy += 2;
    }
    str += "  a b c d e f g h";
    return str;
  }

  virtual std::string actionDescription(const _Action& action) const override {
    auto move = board.moves.at(action.GetIndex());
    return board.moveString(move);
  }

  void featurize() {
    const size_t boardDim = 8 + 4;
    size_t findex = 0;
    std::fill(_features.begin(), _features.begin() + boardSize * boardSize * 12,
              0.0f);
    for (size_t ii = 2 + boardDim * 2;;) {
      for (size_t i = 0; i != 8; ++i, ++ii) {
        int v = board.board[ii];
        if (v) {
          size_t offset = 0;
          switch (v & 0xf) {
          case ChessBoard::PAWN:
            offset = 0;
            break;
          case ChessBoard::KNIGHT:
            offset = 1;
            break;
          case ChessBoard::BISHOP:
            offset = 2;
            break;
          case ChessBoard::ROOK:
            offset = 3;
            break;
          case ChessBoard::QUEEN:
            offset = 4;
            break;
          case ChessBoard::KING:
            offset = 5;
            break;
          }
          if (v & ChessBoard::BLACK) {
            offset += 6;
          }
          _features[boardSize * boardSize * offset + findex] = 1.0f;
        }
        ++findex;
      }
      ii += 2;
      if (ii == 12 + boardDim * 9) {
        break;
      }
      ii += 2;
    }
  }

  void findActions() {
    _NewlegalActions.clear();

    const size_t boardDim = 12;

    for (auto move : board.moves) {
      size_t to = move >> 17;
      size_t from = move & 0x7fff;

      int v = board.board[from];

      size_t offset = 0;
      switch (v & 0xf) {
      case ChessBoard::PAWN:
        offset = 0;
        break;
      case ChessBoard::KNIGHT:
        offset = 1;
        break;
      case ChessBoard::BISHOP:
        offset = 2;
        break;
      case ChessBoard::ROOK:
        offset = 3;
        break;
      case ChessBoard::QUEEN:
        offset = 4;
        break;
      case ChessBoard::KING:
        offset = 5;
        break;
      }

      size_t x = to % boardDim - 2;
      size_t y = to / boardDim - 2;

      _NewlegalActions.emplace_back(_NewlegalActions.size(), offset, y, x);
    }
  }

  virtual void ApplyAction(const _Action& action) override {
    auto move = board.moves.at(action.GetIndex());

    board.move(move);
    board.findMoves();
    findActions();

    if (board.done) {
      if (board.winner == 0) {
        _status = GameStatus::player0Win;
      } else if (board.winner == 1) {
        _status = GameStatus::player1Win;
      } else {
        _status = GameStatus::tie;
      }
    } else {
      _status =
          board.turn == 0 ? GameStatus::player0Turn : GameStatus::player1Turn;
      featurize();
    }
    fillFullFeatures();
  }

  virtual void DoGoodAction() override {
    return DoRandomAction();
  }
};

}  // namespace chess
