/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: Lin Hsin-I
// - Github: https://github.com/free00000000000
// - Email:   410521233@gms.ndhu.edu.tw
// Facilitator: 邱顯棟 (Xiǎn-Dòng Qiū)
// - Github: https://github.com/YumJelly
// - Email:  yumjelly@gmail.com

#include "../core/state.h"

typedef unsigned short Coord;

#include "time.h"
#include <iostream>
#include <random>
#include <string>
#include <vector>

const int StateForDiceshogiX = 225;
const int StateForDiceshogiY = 5;
const int StateForDiceshogiZ = 5;

#include "diceshogi.h"

class ActionForDiceshogi : public _Action {
 public:
  // each action has a position (_x[0], _x[1], _x[2])
  // here for Diceshogi, there is (0, 0, 0) and (1, 0, 0),
  // corresponding to steps 2 and 3 respectively.
  ActionForDiceshogi(int x, int y, int piece)
      : _Action() {
    _loc[0] = piece;
    _loc[1] = x;
    _loc[2] = y;
    _hash = (x + y * 5) * 19 + piece;
  }  // step is 2 or 3.
};

class StateForDiceshogi : public core::State {
 public:
  StateForDiceshogi(int seed)
      : State(seed) {
    _stochasticReset = true;
  }

  DSPiece board[DSDx][DSDy];
  unsigned long long hash;

  DSMove rollout[DSMaxPlayoutLength];
  int length, turn;

  int repeat;
  std::queue<unsigned long long> situation;

  // 0 = DiceWhite, 1 = DiceBlack
  std::vector<std::vector<DSPiece>> chess;
  // 0~5
  short dice;

  void init() {
    chess.clear();
    chess.resize(2);
    for (int i = 0; i < DSDx; ++i) {
      for (int j = 0; j < DSDy; ++j) {
        board[i][j] = DSPiece(DiceEmpty, DSPieceType::None, false);
      }
    }

    for (int i = 1; i <= DSDx; ++i) {
      board[i - 1][0].addDSPiece(
          DiceWhite, DSPieceType(i), false, DSPosition(i - 1, 0));
      chess[DiceWhite].push_back(board[i - 1][0]);
    }
    board[0][1].addDSPiece(
        DiceWhite, DSPieceType::Pawn, false, DSPosition(0, 1));
    chess[DiceWhite].push_back(board[0][1]);
    for (int i = 1; i <= DSDx; ++i) {
      board[DSDx - i][4].addDSPiece(
          DiceBlack, DSPieceType(i), false, DSPosition(DSDx - i, 4));
      chess[DiceBlack].push_back(board[DSDx - i][4]);
    }
    board[4][3].addDSPiece(
        DiceBlack, DSPieceType::Pawn, false, DSPosition(4, 3));
    chess[DiceBlack].push_back(board[4][3]);

    turn = DiceBlack;  // black first
    if (forcedDice > 0) {
      assert(forcedDice > 0);
      assert(forcedDice < 7);
      dice = forcedDice - 1;
      forcedDice = -1;
    } else {
      dice = _rng() % 6;
    }
    _hash = dice + 1;
    hash = 0;
    length = 0;
    repeat = 0;
    situation.push(hash);
  }

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<StateForDiceshogi>(*this);
  }

  bool fourfold() {
    if (repeat < 9)
      return false;
    return true;
  }

  bool won(int color) {
    if (chess[color].back().type == DSPieceType::King)
      return true;
    if (_NewlegalActions.empty())
      return true;
    if (fourfold() && opponent(turn) == color)
      return true;
    return false;
  }

  virtual std::string stateDescription() const override {
    std::string str;
    str += "   A| B| C| D| E\n";
    for (int i = DSDy - 1; i >= 0; --i) {
      str += to_string(i + 1) + ' ';
      for (int j = 0; j < DSDx; ++j) {
        if (j > 0)
          str += '|';
        str += board[j][i].print();
      }
      str += '\n';
    }

    return str;
  }

  virtual std::string actionsDescription() override {
    std::stringstream ss;
    char x1, y1;
    for (int i = 0; i < (int)_NewlegalActions.size(); i++) {
      _Action& action = *(_NewlegalActions[i]);
      int color = (GameStatus)_status == GameStatus::player1Turn ? DiceWhite
                                                                 : DiceBlack;
      DSPieceType type = z_to_type(action.GetX());
      bool promote = z_promoted(action.GetX());
      DSPiece piece = DSPiece(color, type, promote);

      x1 = static_cast<char>(action.GetY() + 'A');
      y1 = static_cast<char>(action.GetZ() + '1');
      ss << "Action " << i << ": " << piece.print() << "-" << x1 << y1
         << std::endl;
    }
    ss << "\nInput format : action index e.g. 0\n";
    return ss.str();
  }

  virtual std::string actionDescription(const _Action& action) const {
    std::stringstream ss;
    char x1, y1;
    int color = (turn + 1) % 2;
    DSPieceType type = z_to_type(action.GetX());
    bool promote = z_promoted(action.GetX());
    DSPiece piece = DSPiece(color, type, promote);

    x1 = static_cast<char>(action.GetY() + 'A');
    y1 = static_cast<char>(action.GetZ() + '1');
    ss << piece.print() << "-" << x1 << y1;

    return ss.str();
  }

  void print_chess(int color, FILE* fp) {
    if (color == DiceWhite)
      fprintf(fp, "DiceWhite ");
    else
      fprintf(fp, "DiceBlack ");
    fprintf(fp, "%lu\n", chess[color].size());
    std::vector<DSPiece>::iterator it;
    for (it = chess[color].begin(); it != chess[color].end(); ++it) {
      if (!(*it).pos.on_board()) {
        fprintf(fp, "(%s)", (*it).print().c_str());
      } else
        fprintf(fp, "%s", (*it).print().c_str());
    }
    fprintf(fp, "\n");
  }

  void legal_king_moves(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    short dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    short dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int i = 0; i < 8; ++i) {
      origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
      if (origin.next.on_board() &&
          board[origin.next.x][origin.next.y].color != origin.piece.color)
        moves.push_back(origin);
    }
  }

  void legal_gold_moves(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    if (origin.piece.color == DiceWhite) {
      short dx[] = {1, 1, 0, -1, -1, 0};
      short dy[] = {0, 1, 1, 1, 0, -1};
      for (int i = 0; i < 6; ++i) {
        origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
        if (origin.next.on_board() &&
            board[origin.next.x][origin.next.y].color != origin.piece.color)
          moves.push_back(origin);
      }
    } else {
      short dx[] = {1, 0, -1, -1, 0, 1};
      short dy[] = {0, 1, 0, -1, -1, -1};
      for (int i = 0; i < 6; ++i) {
        origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
        if (origin.next.on_board() &&
            board[origin.next.x][origin.next.y].color != origin.piece.color)
          moves.push_back(origin);
      }
    }
  }

  void legal_silver_moves(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    if (origin.piece.promoted) {
      legal_gold_moves(origin, moves);
      return;
    }
    if (origin.piece.color == DiceWhite) {
      short dx[] = {1, 0, -1, -1, 1};
      short dy[] = {1, 1, 1, -1, -1};
      for (int i = 0; i < 5; ++i) {
        origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
        if (origin.next.on_board() &&
            board[origin.next.x][origin.next.y].color != origin.piece.color) {
          moves.push_back(origin);
          if (origin.next.y == 4) {
            origin.promote = true;
            moves.push_back(origin);
            origin.promote = false;
          }
        }
      }
    } else {
      short dx[] = {1, -1, -1, 0, 1};
      short dy[] = {1, 1, -1, -1, -1};
      for (int i = 0; i < 5; ++i) {
        origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
        if (origin.next.on_board() &&
            board[origin.next.x][origin.next.y].color != origin.piece.color) {
          moves.push_back(origin);
          if (origin.next.y == 0) {
            origin.promote = true;
            moves.push_back(origin);
            origin.promote = false;
          }
        }
      }
    }
  }

  void legal_bishop_moves(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    short dx[] = {1, -1, -1, 1};
    short dy[] = {1, 1, -1, -1};
    for (int i = 0; i < 4; ++i) {
      origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
      while (origin.next.on_board() &&
             board[origin.next.x][origin.next.y].color != origin.piece.color) {
        moves.push_back(origin);
        if (origin.piece.color == DiceWhite) {
          if (origin.next.y == 4 && !origin.piece.promoted) {
            origin.promote = true;
            moves.push_back(origin);
            origin.promote = false;
          }
        } else {  // DiceBlack
          if (origin.next.y == 0 && !origin.piece.promoted) {
            origin.promote = true;
            moves.push_back(origin);
            origin.promote = false;
          }
        }
        if (board[origin.next.x][origin.next.y].color != DiceEmpty)
          break;
        origin.next = origin.next + DSPosition(dx[i], dy[i]);
      }
    }
    if (origin.piece.promoted) {
      short dx[] = {1, 0, -1, 0};
      short dy[] = {0, 1, 0, -1};
      for (int i = 0; i < 4; ++i) {
        origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
        if (origin.next.on_board() &&
            board[origin.next.x][origin.next.y].color != origin.piece.color)
          moves.push_back(origin);
      }
    }
  }

  void legal_rook_moves(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    short dx[] = {1, 0, -1, 0};
    short dy[] = {0, 1, 0, -1};
    for (int i = 0; i < 4; ++i) {
      origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
      while (origin.next.on_board() &&
             board[origin.next.x][origin.next.y].color != origin.piece.color) {
        moves.push_back(origin);
        if (origin.piece.color == DiceWhite) {
          if (origin.next.y == 4 && !origin.piece.promoted) {
            origin.promote = true;
            moves.push_back(origin);
            origin.promote = false;
          }
        } else {  // DiceBlack
          if (origin.next.y == 0 && !origin.piece.promoted) {
            origin.promote = true;
            moves.push_back(origin);
            origin.promote = false;
          }
        }
        if (board[origin.next.x][origin.next.y].color != DiceEmpty)
          break;
        origin.next = origin.next + DSPosition(dx[i], dy[i]);
      }
    }
    if (origin.piece.promoted) {
      short dx[] = {1, -1, -1, 1};
      short dy[] = {1, 1, -1, -1};
      for (int i = 0; i < 4; ++i) {
        origin.next = origin.piece.pos + DSPosition(dx[i], dy[i]);
        if (origin.next.on_board() &&
            board[origin.next.x][origin.next.y].color != origin.piece.color)
          moves.push_back(origin);
      }
    }
  }

  void legal_pawn_moves(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    if (origin.piece.promoted) {
      legal_gold_moves(origin, moves);
      return;
    }
    if (origin.piece.color == DiceWhite) {
      origin.next = origin.piece.pos + DSPosition(0, 1);
      if (origin.next.on_board() &&
          board[origin.next.x][origin.next.y].color != origin.piece.color) {
        if (origin.next.y != 5)
          moves.push_back(origin);
        else {
          origin.promote = true;
          moves.push_back(origin);
          origin.promote = false;
        }
      }

    } else {
      origin.next = origin.piece.pos + DSPosition(0, -1);
      if (origin.next.on_board() &&
          board[origin.next.x][origin.next.y].color != origin.piece.color) {
        if (origin.next.y != 0)
          moves.push_back(origin);
        else {
          origin.promote = true;
          moves.push_back(origin);
          origin.promote = false;
        }
      }
    }
  }

  void legal_drop(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    for (int i = 0; i < DSDx; ++i) {
      for (int j = 0; j < DSDy; ++j) {
        if (board[i][j].color == DiceEmpty) {
          origin.next = DSPosition(i, j);
          moves.push_back(origin);
        }
      }
    }
  }

  void legal_drop_pawn(DSMove origin, std::vector<DSMove>& moves) {
    origin.promote = false;
    // find another pawn
    std::vector<DSPiece>::iterator it;
    int cannotdrop = DSDx;
    DSPieceType t = new_type(origin.piece.type);
    for (it = chess[origin.piece.color].begin();
         it != chess[origin.piece.color].end(); ++it) {
      if ((*it).type == t)
        cannotdrop = (*it).pos.x;
    }

    if (origin.piece.color == DiceWhite) {
      for (int i = 0; i < DSDx; ++i) {
        if (i == cannotdrop)
          continue;
        for (int j = 0; j < DSDy - 1; ++j) {
          if (board[i][j].color == DiceEmpty) {
            if ((j - 1) >= 0 && board[i][j - 1].type == DSPieceType::King &&
                board[i][j - 1].color != origin.piece.color) {
              if (checkmate(board[i][j - 1]))
                continue;
            }
            origin.next = DSPosition(i, j);
            moves.push_back(origin);
          }
        }
      }
    } else {
      for (int i = 0; i < DSDx; ++i) {
        if (i == cannotdrop)
          continue;
        for (int j = 1; j < DSDy; ++j) {
          if (board[i][j].color == DiceEmpty) {
            if (board[i][j - 1].type == DSPieceType::King &&
                board[i][j - 1].color != origin.piece.color) {
              if (checkmate(board[i][j - 1]))
                continue;
            }
            origin.next = DSPosition(i, j);
            moves.push_back(origin);
          }
        }
      }
    }
  }

  bool can_eat(DSPosition tar, int color) {
    std::vector<DSMove> moves;
    legalDSMoves_onboard(opponent(color), moves);

    std::vector<DSMove>::iterator it;
    for (it = moves.begin(); it != moves.end(); ++it)
      if ((*it).next == tar)
        return true;

    return false;
  }

  bool checkmate(DSPiece king) {
    std::vector<DSMove> king_moves;
    DSMove m;
    m.piece = king;
    legal_king_moves(m, king_moves);
    if (king_moves.empty())
      return true;

    std::vector<DSMove>::iterator it;
    for (it = king_moves.begin(); it != king_moves.end(); ++it)
      if (!can_eat((*it).next, king.color))
        return false;

    return true;
  }

  void legalDSMoves(int color, std::vector<DSMove>& moves) {
    legalDSMoves_onboard(color, moves);
    std::vector<DSPiece>::iterator it;
    for (it = chess[color].begin(); it != chess[color].end(); ++it) {
      DSPiece p = *it;

      if (!p.pos.on_board()) {
        DSMove m;
        m.piece = p;
        switch (m.piece.type) {
        case DSPieceType::Gold:
        case DSPieceType::Gold2:
        case DSPieceType::Silver:
        case DSPieceType::Silver2:
        case DSPieceType::Bishop:
        case DSPieceType::Bishop2:
        case DSPieceType::Rook:
        case DSPieceType::Rook2:
          legal_drop(m, moves);
          break;

        case DSPieceType::Pawn:
        case DSPieceType::Pawn2:
          legal_drop_pawn(m, moves);
          break;

        default:
          break;
        }
      }
    }
  }

  void legalDSMoves_onboard(int color, std::vector<DSMove>& moves) {
    std::vector<DSPiece>::iterator it;
    for (it = chess[color].begin(); it != chess[color].end(); ++it) {
      DSPiece p = *it;
      DSMove m;
      m.piece = p;
      if (m.piece.pos.on_board()) {
        switch (m.piece.type) {
        case DSPieceType::King:
          legal_king_moves(m, moves);
          break;

        case DSPieceType::Gold:
        case DSPieceType::Gold2:
          legal_gold_moves(m, moves);
          break;

        case DSPieceType::Silver:
        case DSPieceType::Silver2:
          legal_silver_moves(m, moves);
          break;

        case DSPieceType::Bishop:
        case DSPieceType::Bishop2:
          legal_bishop_moves(m, moves);
          break;

        case DSPieceType::Rook:
        case DSPieceType::Rook2:
          legal_rook_moves(m, moves);
          break;

        case DSPieceType::Pawn:
        case DSPieceType::Pawn2:
          legal_pawn_moves(m, moves);
          break;

        default:
          break;
        }
      }
    }
  }

  int opponent(int player) {
    if (player == DiceWhite)
      return DiceBlack;
    return DiceWhite;
  }

  DSPieceType new_type(DSPieceType p) {
    DSPieceType t = p;
    switch (p) {
    case DSPieceType::Gold:
      t = DSPieceType::Gold2;
      break;
    case DSPieceType::Gold2:
      t = DSPieceType::Gold;
      break;
    case DSPieceType::Silver:
      t = DSPieceType::Silver2;
      break;
    case DSPieceType::Silver2:
      t = DSPieceType::Silver;
      break;
    case DSPieceType::Bishop:
      t = DSPieceType::Bishop2;
      break;
    case DSPieceType::Bishop2:
      t = DSPieceType::Bishop;
      break;
    case DSPieceType::Rook:
      t = DSPieceType::Rook2;
      break;
    case DSPieceType::Rook2:
      t = DSPieceType::Rook;
      break;
    case DSPieceType::Pawn:
      t = DSPieceType::Pawn2;
      break;
    case DSPieceType::Pawn2:
      t = DSPieceType::Pawn;
      break;
    default:
      break;
    }
    return t;
  }

  void play(DSMove m) {
    m.piece.promoted |= m.promote;

    turn = opponent(turn);
    if (m.piece.pos.on_board()) {
      hash ^= DSHashArray[m.piece.color][getHashNum(m.piece)][m.piece.pos.x]
                         [m.piece.pos.y];

      // eat
      if (board[m.next.x][m.next.y].color != DiceEmpty) {
        hash ^= DSHashArray[turn][getHashNum(board[m.next.x][m.next.y])]
                           [m.next.x][m.next.y];
        hash ^= DSHashArrayE[getHashNumE(m.piece)];

        DSPiece tmp(
            m.piece.color, new_type(board[m.next.x][m.next.y].type), false);
        chess[m.piece.color].push_back(tmp);

        std::vector<DSPiece>::iterator it;
        for (it = chess[turn].begin(); it != chess[turn].end(); ++it) {
          if ((*it).type == board[m.next.x][m.next.y].type) {
            chess[turn].erase(it);
            break;
          }
        }
      }

      std::vector<DSPiece>::iterator it;
      for (it = chess[m.piece.color].begin(); it != chess[m.piece.color].end();
           ++it) {
        if ((*it).type == m.piece.type) {
          (*it).pos = m.next;
          // decide promoted
          if ((m.piece.color == DiceWhite && m.next.y == DSDy - 1) ||
              (m.piece.color == DiceBlack && m.next.y == 0)) {
            if (m.piece.promoted || (*it).type == DSPieceType::Pawn ||
                (*it).type == DSPieceType::Pawn2)
              (*it).promoted = true;
          }
          board[m.next.x][m.next.y] = (*it);
          board[m.piece.pos.x][m.piece.pos.y] =
              DSPiece(DiceEmpty, DSPieceType::None, false);
          break;
        }
      }

    } else {
      hash ^= DSHashArrayE[getHashNumE(m.piece)];

      std::vector<DSPiece>::iterator it;
      for (it = chess[m.piece.color].begin(); it != chess[m.piece.color].end();
           ++it) {
        if ((*it).type == m.piece.type) {
          (*it).pos = m.next;
          board[m.next.x][m.next.y] = (*it);
          break;
        }
      }
    }

    hash ^= DSHashArray[m.piece.color][getHashNum(board[m.next.x][m.next.y])]
                       [m.next.x][m.next.y];
    hash ^= DSHashTurn;

    if (length < DSMaxPlayoutLength) {
      rollout[length] = m;
      length++;
    } else {
      _status = GameStatus::tie;
    }

    // find repeat
    if (hash != situation.front())
      repeat = 0;
    else
      repeat += 1;
  }

  int getHashNum(DSPiece p) {
    int num = static_cast<int>(p.type);
    if (num >= 7)
      num -= 5;
    if (p.promoted)
      num += 5;
    //   7 8 9 10 11
    // 1 2 3 4  5  6 | 7 8 9 10
    num -= 1;
    return num;
  }

  int getHashNumE(DSPiece p) {
    // 0~19
    return static_cast<int>(p.type) - 2 + 10 * p.color;
  }

  // ############### board

  virtual void Initialize() override {
    // People implementing classes should not have much to do in _moves; just
    // _moves.clear().
    _stochastic = true;
    _moves.clear();
    // std::cout << "OTGDiceshogi initialize" << std::endl;

    // the features are just one number between 0 and 1 (the distance,
    // normalized).
    _featSize[0] = StateForDiceshogiX;
    _featSize[1] = StateForDiceshogiY;
    _featSize[2] = StateForDiceshogiZ;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).

    _actionSize[0] = 19;
    _actionSize[1] = 5;
    _actionSize[2] = 5;

    // _hash is an unsigned int, it has to be *unique*.
    _hash = 0;

    // std::cout << "restart!" << std::endl;
    // _features is a vector representing the current state. It can
    // (must...) be large for complex games; here just one number
    // between 0 and 1. trivial case in dimension 1.
    _features.resize(StateForDiceshogiX * StateForDiceshogiY *
                     StateForDiceshogiZ);
    std::fill(_features.begin(), _features.end(), 0);
    /*
        // _features[:_hash] = 1
        for (int i = 0; i < DISTANCE; i++) {
          _features[i] = (float(_hash) > float(i)) ? 1. : 0.;
        }
    */
    init();
    _status = (GameStatus)opponent(turn);
    findFeatures();
    findActions(turn);
    fillFullFeatures();
  }

  int type_to_z(DSPiece p) {
    if (!p.promoted)
      return (int)p.type - 1;
    switch (p.type) {
    case DSPieceType::Silver:
      return (int)DSPieceZ::PSilver;
    case DSPieceType::Bishop:
      return (int)DSPieceZ::PBishop;
    case DSPieceType::Rook:
      return (int)DSPieceZ::PRook;
    case DSPieceType::Pawn:
      return (int)DSPieceZ::PPawn;
    case DSPieceType::Silver2:
      return (int)DSPieceZ::PSilver2;
    case DSPieceType::Bishop2:
      return (int)DSPieceZ::PBishop2;
    case DSPieceType::Rook2:
      return (int)DSPieceZ::PRook2;
    case DSPieceType::Pawn2:
      return (int)DSPieceZ::PPawn2;

    default:
      // fprintf(stderr);
      fprintf(
          stderr, "%s type to z error %d\n", p.print().c_str(), (int)p.type);
      break;
    }
    return -1;
  }

  DSPieceType z_to_type(int z) const {
    switch ((DSPieceZ)z) {
    case DSPieceZ::K:
      return DSPieceType::King;
    case DSPieceZ::G:
      return DSPieceType::Gold;
    case DSPieceZ::G2:
      return DSPieceType::Gold2;
    case DSPieceZ::PSilver:
    case DSPieceZ::S:
      return DSPieceType::Silver;
    case DSPieceZ::PBishop:
    case DSPieceZ::B:
      return DSPieceType::Bishop;
    case DSPieceZ::PRook:
    case DSPieceZ::R:
      return DSPieceType::Rook;
    case DSPieceZ::PPawn:
    case DSPieceZ::P:
      return DSPieceType::Pawn;
    case DSPieceZ::PSilver2:
    case DSPieceZ::S2:
      return DSPieceType::Silver2;
    case DSPieceZ::PBishop2:
    case DSPieceZ::B2:
      return DSPieceType::Bishop2;
    case DSPieceZ::PRook2:
    case DSPieceZ::R2:
      return DSPieceType::Rook2;
    case DSPieceZ::PPawn2:
    case DSPieceZ::P2:
      return DSPieceType::Pawn2;

    default:
      // print(stderr);
      fprintf(stderr, "%d", z);
      fprintf(stderr, "z to type error\n");
      break;
    }
    return DSPieceType::None;
  }

  bool z_promoted(int z) const {
    return z >= 11;
  }

  void findActions(int color) {
    std::vector<DSMove> moves;
    std::vector<DSMove> dice_moves;
    legalDSMoves(color, moves);

    // diece limit
    if (dice != 5) {
      std::vector<DSMove>::iterator it;
      for (it = moves.begin(); it != moves.end(); ++it) {
        if ((*it).next.x == dice)
          dice_moves.push_back((*it));
      }
    }
    if (dice_moves.empty())
      dice_moves = moves;

    int nb = dice_moves.size();

    clearActions();
    for (int i = 0; i < nb; ++i) {
      int x = dice_moves[i].next.x;
      int y = dice_moves[i].next.y;
      dice_moves[i].piece.promoted |= dice_moves[i].promote;
      int z = type_to_z(dice_moves[i].piece);

      addAction(z, x, y);
    }
  }

  void findFeatures() {
    // fprintf(stderr, "rrrrrrrrrrrr%d\n", turn);
    std::vector<float> old(_features);
    for (int i = 0; i < 5425; ++i)
      _features[i] = 0;
    // 0 ~ 500
    for (int i = 0; i < 25; ++i) {
      DSPiece p = board[i % 5][i / 5];
      if (p.color == DiceWhite) {
        switch (p.type) {
        case DSPieceType::King:
          _features[i] = 1;
          break;

        case DSPieceType::Gold:
        case DSPieceType::Gold2:
          _features[25 + i] = 1;
          break;

        case DSPieceType::Silver:
        case DSPieceType::Silver2:
          if (p.promoted)
            _features[50 + i] = 1;
          else
            _features[75 + i] = 1;
          break;

        case DSPieceType::Bishop:
        case DSPieceType::Bishop2:
          if (p.promoted)
            _features[100 + i] = 1;
          else
            _features[125 + i] = 1;
          break;

        case DSPieceType::Rook:
        case DSPieceType::Rook2:
          if (p.promoted)
            _features[150 + i] = 1;
          else
            _features[175 + i] = 1;
          break;

        case DSPieceType::Pawn:
        case DSPieceType::Pawn2:
          if (p.promoted)
            _features[200 + i] = 1;
          else
            _features[225 + i] = 1;
          break;

        default:
          break;
        }
      } else {
        switch (p.type) {
        case DSPieceType::King:
          _features[250 + i] = 1;
          break;

        case DSPieceType::Gold:
        case DSPieceType::Gold2:
          _features[275 + i] = 1;
          break;

        case DSPieceType::Silver:
        case DSPieceType::Silver2:
          if (p.promoted)
            _features[300 + i] = 1;
          else
            _features[325 + i] = 1;
          break;

        case DSPieceType::Bishop:
        case DSPieceType::Bishop2:
          if (p.promoted)
            _features[350 + i] = 1;
          else
            _features[375 + i] = 1;
          break;

        case DSPieceType::Rook:
        case DSPieceType::Rook2:
          if (p.promoted)
            _features[400 + i] = 1;
          else
            _features[425 + i] = 1;
          break;

        case DSPieceType::Pawn:
        case DSPieceType::Pawn2:
          if (p.promoted)
            _features[450 + i] = 1;
          else
            _features[475 + i] = 1;
          break;

        default:
          break;
        }
      }
    }

    // 500 ~ 575
    switch (repeat) {
    case 1:
      std::fill(_features.begin() + 500, _features.begin() + 525, 1);
      break;
    case 5:
      std::fill(_features.begin() + 525, _features.begin() + 550, 1);
      break;
    case 9:
      std::fill(_features.begin() + 550, _features.begin() + 575, 1);
      break;
    default:
      break;
    }

    // prison w 575 ~ 625
    // prison b 625 ~ 675
    int tmp = 575;
    for (int i = 0; i < 2; ++i) {
      std::vector<DSPiece>::iterator it;
      for (it = chess[i].begin(); it != chess[i].end(); ++it) {
        if (!(*it).pos.on_board()) {
          switch ((*it).type) {
          case DSPieceType::Gold:
            std::fill(_features.begin() + tmp, _features.begin() + tmp + 5, 1);
            break;
          case DSPieceType::Silver:
            std::fill(
                _features.begin() + tmp + 5, _features.begin() + tmp + 10, 1);
            break;
          case DSPieceType::Bishop:
            std::fill(
                _features.begin() + tmp + 10, _features.begin() + tmp + 15, 1);
            break;
          case DSPieceType::Rook:
            std::fill(
                _features.begin() + tmp + 15, _features.begin() + tmp + 20, 1);
            break;
          case DSPieceType::Pawn:
            std::fill(
                _features.begin() + tmp + 20, _features.begin() + tmp + 25, 1);
            break;
          case DSPieceType::Gold2:
            std::fill(
                _features.begin() + tmp + 25, _features.begin() + tmp + 30, 1);
            break;
          case DSPieceType::Silver2:
            std::fill(
                _features.begin() + tmp + 30, _features.begin() + tmp + 35, 1);
            break;
          case DSPieceType::Bishop2:
            std::fill(
                _features.begin() + tmp + 35, _features.begin() + tmp + 40, 1);
            break;
          case DSPieceType::Rook2:
            std::fill(
                _features.begin() + tmp + 40, _features.begin() + tmp + 45, 1);
            break;
          case DSPieceType::Pawn2:
            std::fill(
                _features.begin() + tmp + 45, _features.begin() + tmp + 50, 1);
            break;
          default:
            break;
          }
        }
      }
      tmp += 50;
    }

    // dice 675 ~ 700
    if (dice == 5)
      std::fill(_features.begin() + 675, _features.begin() + 700, 1);
    else
      std::fill(
          _features.begin() + dice * 5, _features.begin() + dice * 5 + 5, 1);

    // history 700 ~ 4900+700
    std::copy(old.begin(), old.begin() + 4900, _features.begin() + 700);

    // 5600 ~ 5625
    std::fill(_features.begin() + 5600, _features.end(), turn);
  }
  // The action just decreases the distance and swaps the turn to play.
  virtual void ApplyAction(const _Action& action) override {
    DSMove m;
    if ((GameStatus)_status ==
        GameStatus::player1Turn) {  // 1 DiceWhite to move
      // fprintf(stderr, "DiceWhite ");
      m.piece.color = DiceWhite;
      m.next = DSPosition(action.GetY(), action.GetZ());
      m.piece.type = z_to_type(action.GetX());
      m.promote = z_promoted(action.GetX());

      std::vector<DSPiece>::iterator it;
      for (it = chess[DiceWhite].begin(); it != chess[DiceWhite].end(); ++it) {
        if ((*it).type == m.piece.type)
          m.piece.pos = (*it).pos;
      }

      play(m);

      if ((GameStatus)_status == GameStatus::tie) {
      }  // fprintf(stderr, "draw ");  // draw
      else if (!won(DiceWhite))
        _status = GameStatus::player0Turn;  // DiceBlack turn
      else
        _status = GameStatus::player1Win;  // DiceWhite won
    } else {                               // DiceBlack
      // fprintf(stderr, "DiceBlack ");
      m.piece.color = DiceBlack;
      m.next = DSPosition(action.GetY(), action.GetZ());
      m.piece.type = z_to_type(action.GetX());
      m.promote = z_promoted(action.GetX());

      std::vector<DSPiece>::iterator it;
      for (it = chess[DiceBlack].begin(); it != chess[DiceBlack].end(); ++it) {
        if ((*it).type == m.piece.type)
          m.piece.pos = (*it).pos;
      }

      play(m);

      if ((GameStatus)_status == GameStatus::tie)
        fprintf(stderr, "draw ");  // draw
      else if (!won(DiceBlack))
        _status = GameStatus::player1Turn;  // DiceWhite turn
      else
        _status = GameStatus::player0Win;  // DiceBlack won
    }
    findFeatures();
    if (situation.size() == 4) {
      situation.pop();
      situation.push(hash);
    } else
      situation.push(hash);
    if (forcedDice >= 0) {
      assert(forcedDice > 0);
      assert(forcedDice < 7);
      dice = forcedDice - 1;
      forcedDice = -1;
    } else {
      dice = _rng() % 6;
    }
    // TODO useless now.
    _hash = dice + 1;  // This is useful for the interaction with human. From
                       // now on, _hash represented the random dice.
    findActions(turn);
    fillFullFeatures();
  }

  // For this trivial example we just compare to random play
  virtual void DoGoodAction() override {
    DoRandomAction();
  }
};
