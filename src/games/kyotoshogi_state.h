/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "../core/state.h"

typedef unsigned short Coord;

#include "time.h"
#include <iostream>
#include <random>
#include <string>
#include <vector>

const int StateForKyotoshogiNumActions = 64 * 3;
const int StateForKyotoshogiX = 201;  //(9+9+3+2+2)*8
const int StateForKyotoshogiY = 5;
const int StateForKyotoshogiZ = 5;

#include "kyotoshogi.h"

class ActionForKyotoshogi : public _Action {
 public:
  // each action has a position (_x[0], _x[1], _x[2])
  // here for Kyotoshogi, there is (0, 0, 0) and (1, 0, 0),
  // corresponding to steps 2 and 3 respectively.
  ActionForKyotoshogi(int x, int y, int piece)
      : _Action() {
    _loc[0] = piece;
    _loc[1] = x;
    _loc[2] = y;
    _hash = (x + y * 5) * 17 + piece;
  }
};

class StateForKyotoshogi : public core::State {
 public:
  StateForKyotoshogi(int seed) : State(seed) {}
  KSPiece board[KSDx][KSDy];
  unsigned long long hash;
  KSMove rollout[KSMaxPlayoutLength];
  int length, turn;

  int Repetition;
  std::queue<unsigned long long> situation;

  // 0 = White, 1 = Black
  std::vector<std::vector<KSPiece>> chess;
  std::queue<unsigned long long> repet;

  void init() {
    chess.clear();
    chess.resize(2);
    for (int i = 0; i < KSDx; ++i) {
      for (int j = 0; j < KSDy; ++j) {
        board[i][j] = KSPiece(KyotoEmpty, KSNone, false);
      }
    }

    for (int i = 1; i <= KSDx; ++i) {
      board[i - 1][0].addKSPiece(
          KyotoWhite, KSPieceType(i), false, KSPosition(i - 1, 0));
      chess[KyotoWhite].push_back(board[i - 1][0]);
    }

    for (int i = 1; i <= KSDx; ++i) {
      board[KSDx - i][4].addKSPiece(
          KyotoBlack, KSPieceType(i), false, KSPosition(KSDx - i, 4));
      chess[KyotoBlack].push_back(board[KSDx - i][4]);
    }

    turn = KyotoBlack;  // black first
    hash = 0;
    length = 0;
    Repetition = 0;
    repet.push(_hash);
  }

  bool fourfold() {
    if (Repetition < 9)
      return false;
    return true;
  }

  bool won(int color) {
    if (chess[color].back().type == KSKing)
      return true;
    if (_legalActions.empty())
      return true;
    
    return false;
  }

  virtual std::string stateDescription() const override {
    std::string str;
      str += "   A| B| C| D| E\n";
      for (int i = KSDy - 1; i >= 0; --i) {
        str += std::to_string(i+1) + ' ';
        for (int j = 0; j < KSDx; ++j) {
          if(j > 0)
            str += '|';
          str += board[j][i].print();
        }
        str += '\n';
      }

    return str;
  }

  virtual std::string actionsDescription() const override {
    std::stringstream ss;
    char x1, y1;
    for (int i = 0; i < (int)_legalActions.size(); i++) {
      _Action & action = *(_legalActions[i]);
      int color = (GameStatus)_status == GameStatus::player1Turn ? KyotoWhite : KyotoBlack;
      KSPieceType type = z_to_type(action.GetX());
      bool promote = z_promoted(action.GetX());
      KSPiece piece = KSPiece(color, type, promote);
      KSPiece flip = KSPiece(color, type, !promote);

      x1 = static_cast<char>(action.GetY() + 'A');
      y1 = static_cast<char>(action.GetZ() + '1');
      ss << "Action " << i << ": " << piece.print() << " -> " << flip.print() << "-" << x1 << y1 << std::endl;
    }
    ss << "\nInput format : action index e.g. 0\n";
    return ss.str();
  }

  virtual std::string actionDescription(const _Action & action) const {
    std::stringstream ss;
    char x1, y1;
    int color = (turn + 1) % 2;
    KSPieceType type = z_to_type(action.GetX());
    bool promote = z_promoted(action.GetX());
    KSPiece piece = KSPiece(color, type, promote);
    KSPiece flip = KSPiece(color, type, !promote);

    x1 = static_cast<char>(action.GetY() + 'A');
    y1 = static_cast<char>(action.GetZ() + '1');
    ss << piece.print() << " -> " << flip.print() << "-" << x1 << y1;

    return ss.str();
  }

  void print_chess(int color, FILE* fp) {
    if (color == KyotoWhite)
      fprintf(fp, "KyotoWhite ");
    else
      fprintf(fp, "KyotoBlack ");
    fprintf(fp, "%lu\n", chess[color].size());
    std::vector<KSPiece>::iterator it;
    for (it = chess[color].begin(); it != chess[color].end(); ++it) {
      if (!(*it).pos.on_board()) {
        fprintf(fp, "(%s)", (*it).print().c_str());
      } else
        fprintf(fp, "%s", (*it).print().c_str());
    }
    fprintf(fp, "\n");
  }

  void legal_king_moves(KSMove origin, std::vector<KSMove>& moves) {
    origin.piece.promoted = false;
    short dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    short dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int i = 0; i < 8; ++i) {
      origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
      if (origin.pos1.on_board() &&
          board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
        moves.push_back(origin);
    }
  }

  void legal_gold_knight_moves(KSMove origin, std::vector<KSMove>& moves) {
    if (origin.piece.promoted) {
      if (origin.piece.color == KyotoWhite) {
        short dx[] = {-1, 1};
        short dy[] = {2, 2};
        for (int i = 0; i < 2; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      } else {
        short dx[] = {-1, 1};
        short dy[] = {-2, -2};
        for (int i = 0; i < 2; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      }
    } else {
      if (origin.piece.color == KyotoWhite) {
        short dx[] = {1, 1, 0, -1, -1, 0};
        short dy[] = {0, 1, 1, 1, 0, -1};
        for (int i = 0; i < 6; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      } else {
        short dx[] = {1, 0, -1, -1, 0, 1};
        short dy[] = {0, 1, 0, -1, -1, -1};
        for (int i = 0; i < 6; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      }
    }
  }

  void legal_silver_bishop_moves(KSMove origin, std::vector<KSMove>& moves) {
    if (origin.piece.promoted) {
      short dx[] = {1, -1, -1, 1};
      short dy[] = {1, 1, -1, -1};
      for (int i = 0; i < 4; ++i) {
        origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
        while (origin.pos1.on_board() &&
               board[origin.pos1.x][origin.pos1.y].color !=
                   origin.piece.color) {
          moves.push_back(origin);
          if (board[origin.pos1.x][origin.pos1.y].color != KyotoEmpty)
            break;
          origin.pos1 = origin.pos1 + KSPosition(dx[i], dy[i]);
        }
      }
    } else {
      if (origin.piece.color == KyotoWhite) {
        short dx[] = {1, 0, -1, -1, 1};
        short dy[] = {1, 1, 1, -1, -1};
        for (int i = 0; i < 5; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      } else {
        short dx[] = {1, -1, -1, 0, 1};
        short dy[] = {1, 1, -1, -1, -1};
        for (int i = 0; i < 5; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      }
    }
  }

  void legal_pawn_rook_moves(KSMove origin, std::vector<KSMove>& moves) {
    if (origin.piece.promoted) {
      short dx[] = {1, 0, -1, 0};
      short dy[] = {0, 1, 0, -1};
      for (int i = 0; i < 4; ++i) {
        origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
        while (origin.pos1.on_board() &&
               board[origin.pos1.x][origin.pos1.y].color !=
                   origin.piece.color) {
          moves.push_back(origin);
          if (board[origin.pos1.x][origin.pos1.y].color != KyotoEmpty)
            break;
          origin.pos1 = origin.pos1 + KSPosition(dx[i], dy[i]);
        }
      }
    } else {
      if (origin.piece.color == KyotoWhite) {
        origin.pos1 = origin.piece.pos + KSPosition(0, 1);
        if (origin.pos1.on_board() &&
            board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
          moves.push_back(origin);
      } else {
        origin.pos1 = origin.piece.pos + KSPosition(0, -1);
        if (origin.pos1.on_board() &&
            board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
          moves.push_back(origin);
      }
    }
  }

  void legal_tokin_lance_moves(KSMove origin, std::vector<KSMove>& moves) {
    if (origin.piece.promoted) {
      if (origin.piece.color == KyotoWhite) {
        origin.pos1 = origin.piece.pos + KSPosition(0, 1);
        while (origin.pos1.on_board() &&
               board[origin.pos1.x][origin.pos1.y].color !=
                   origin.piece.color) {
          moves.push_back(origin);
          if (board[origin.pos1.x][origin.pos1.y].color != KyotoEmpty)
            break;
          origin.pos1 = origin.pos1 + KSPosition(0, 1);
        }
      } else {
        origin.pos1 = origin.piece.pos + KSPosition(0, -1);
        while (origin.pos1.on_board() &&
               board[origin.pos1.x][origin.pos1.y].color !=
                   origin.piece.color) {
          moves.push_back(origin);
          if (board[origin.pos1.x][origin.pos1.y].color != KyotoEmpty)
            break;
          origin.pos1 = origin.pos1 + KSPosition(0, -1);
        }
      }
    } else {
      if (origin.piece.color == KyotoWhite) {
        short dx[] = {1, 1, 0, -1, -1, 0};
        short dy[] = {0, 1, 1, 1, 0, -1};
        for (int i = 0; i < 6; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      } else {
        short dx[] = {1, 0, -1, -1, 0, 1};
        short dy[] = {0, 1, 0, -1, -1, -1};
        for (int i = 0; i < 6; ++i) {
          origin.pos1 = origin.piece.pos + KSPosition(dx[i], dy[i]);
          if (origin.pos1.on_board() &&
              board[origin.pos1.x][origin.pos1.y].color != origin.piece.color)
            moves.push_back(origin);
        }
      }
    }
  }

  void legal_drop(KSMove origin, std::vector<KSMove>& moves) {
    origin.piece.promoted = false;
    for (int i = 0; i < KSDx; ++i) {
      for (int j = 0; j < KSDy; ++j) {
        if (board[i][j].color == KyotoEmpty) {
          origin.pos1 = KSPosition(i, j);
          origin.piece.promoted = false;
          moves.push_back(origin);
          origin.piece.promoted = true;
          moves.push_back(origin);
        }
      }
    }
  }

  bool can_eat(KSPosition tar, int color) {
    std::vector<KSMove> moves;
    legalKSMoves_onboard(opponent(color), moves);

    std::vector<KSMove>::iterator it;
    for (it = moves.begin(); it != moves.end(); ++it)
      if ((*it).pos1 == tar)
        return true;

    return false;
  }

  bool checkmate(KSPiece king) {
    std::vector<KSMove> king_moves;
    KSMove m;
    m.piece = king;
    legal_king_moves(m, king_moves);
    if (king_moves.empty())
      return true;

    std::vector<KSMove>::iterator it;
    for (it = king_moves.begin(); it != king_moves.end(); ++it)
      if (!can_eat((*it).pos1, king.color))
        return false;
    
    return true;
  }

  void legalKSMoves(int color, std::vector<KSMove>& moves) {
    legalKSMoves_onboard(color, moves);
    std::vector<KSPiece>::iterator it;
    for (it = chess[color].begin(); it != chess[color].end(); ++it) {
      KSPiece p = *it;

      if (!p.pos.on_board()) {
        KSMove m;
        m.piece = p;
        legal_drop(m, moves);
      }
    }
  }

  void legalKSMoves_onboard(int color, std::vector<KSMove>& moves) {
    std::vector<KSPiece>::iterator it;
    for (it = chess[color].begin(); it != chess[color].end(); ++it) {
      KSPiece p = *it;
      KSMove m;
      m.piece = p;

      if (m.piece.pos.on_board()) {
        switch (m.piece.type) {
        case KSKing:
          legal_king_moves(m, moves);
          break;

        case Gold_Knight:
        case Gold_Knight2:
          legal_gold_knight_moves(m, moves);
          break;

        case Silver_Bishop:
        case Silver_Bishop2:
          legal_silver_bishop_moves(m, moves);
          break;

        case Tokin_Lance:
        case Tokin_Lance2:
          legal_tokin_lance_moves(m, moves);
          break;

        case Pawn_Rook:
        case Pawn_Rook2:
          legal_pawn_rook_moves(m, moves);
          break;

        default:
          break;
        }
      }
    }
  }

  int opponent(int player) {
    if (player == KyotoWhite)
      return KyotoBlack;
    return KyotoWhite;
  }

  KSPieceType new_type(KSPieceType p) {
    KSPieceType t = p;
    switch (p) {
    case Gold_Knight:
      t = Gold_Knight2;
      break;
    case Gold_Knight2:
      t = Gold_Knight;
      break;
    case Silver_Bishop:
      t = Silver_Bishop2;
      break;
    case Silver_Bishop2:
      t = Silver_Bishop;
      break;
    case Tokin_Lance:
      t = Tokin_Lance2;
      break;
    case Tokin_Lance2:
      t = Tokin_Lance;
      break;
    case Pawn_Rook:
      t = Pawn_Rook2;
      break;
    case Pawn_Rook2:
      t = Pawn_Rook;
      break;
    default:
      break;
    }
    return t;
  }

  void play(KSMove m) {
    turn = opponent(turn);
    if (m.piece.pos.on_board()) {
      hash ^= KSHashArray[m.piece.color][getHashNum(m.piece)][m.piece.pos.x][m.piece.pos.y];

      if (board[m.pos1.x][m.pos1.y].color != KyotoEmpty) {
        assert(m.pos1.on_board());
        hash ^= KSHashArray[turn][getHashNum(board[m.pos1.x][m.pos1.y])][m.pos1.x][m.pos1.y];
        hash ^= KSHashArrayE[getHashNumE(m.piece)];

        KSPiece tmp(m.piece.color, new_type(board[m.pos1.x][m.pos1.y].type), false);
        chess[m.piece.color].push_back(tmp);

        std::vector<KSPiece>::iterator it;
        for (it = chess[turn].begin(); it != chess[turn].end(); ++it)
          if ((*it).type == board[m.pos1.x][m.pos1.y].type) {
            chess[turn].erase(it);
            break;
          }
      }

      board[m.pos1.x][m.pos1.y] = board[m.piece.pos.x][m.piece.pos.y];
      board[m.pos1.x][m.pos1.y].pos = KSPosition(m.pos1.x, m.pos1.y);
      // decide promoted
      if (m.piece.type != KSKing) {
        board[m.pos1.x][m.pos1.y].promoted =
            !board[m.pos1.x][m.pos1.y].promoted;
      }

      board[m.piece.pos.x][m.piece.pos.y] = KSPiece(KyotoEmpty, KSNone, false);
    } else {
      hash ^= KSHashArrayE[getHashNumE(m.piece)];
      board[m.pos1.x][m.pos1.y] = KSPiece(m.piece.color, m.piece.type, m.piece.promoted);
      board[m.pos1.x][m.pos1.y].pos = KSPosition(m.pos1.x, m.pos1.y);
    }

    std::vector<KSPiece>::iterator it;
    for (it = chess[m.piece.color].begin(); it != chess[m.piece.color].end();
         ++it) {
      if ((*it).type == m.piece.type) {
        (*it).pos = m.pos1;
        if (m.piece.pos.on_board()) {
          if (m.piece.type != KSKing)
            (*it).promoted = !(*it).promoted;
        } else {
          (*it).promoted = m.piece.promoted;
        }
        break;
      }
    }

    hash ^= KSHashArray[m.piece.color][getHashNum(board[m.pos1.x][m.pos1.y])][m.pos1.x][m.pos1.y];
    hash ^= KSHashTurn;

    if (length < KSMaxPlayoutLength) {
      rollout[length] = m;
      length++;
    } else {
      _status = GameStatus::tie;
    }

    if (hash == repet.front()) {
      Repetition += 1;
    } else {
      Repetition = 0;
    }
  }

  int getHashNum(KSPiece p) {
    int num = p.type;
    if (num == 3) {
      num = 1;
    } else if (num == 1 || num == 2) {
      num = num + 1;
    } else if (num >= 6) {
      num -= 4;
    }
    if (p.promoted)
      num += 4;
    num = num - 1;
    return num;
  }

  int getHashNumE(KSPiece p) {
    int num = p.type;
    if (num == 3) {
      num = 1;
    } else if (num == 1 || num == 2) {
      num = num + 1;
    }

    return num - 2 + 10 * p.color;
  }

  virtual void Initialize() override {
    _moves.clear();

    // the features are just one number between 0 and 1 (the distance,
    // normalized).
    _featSize[0] = StateForKyotoshogiX;
    _featSize[1] = StateForKyotoshogiY;
    _featSize[2] = StateForKyotoshogiZ;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).
    _actionSize[0] = 17;
    _actionSize[1] = 5;
    _actionSize[2] = 5;

    // _hash is an unsigned int, it has to be *unique*.
    _hash = 0;

    _features.resize(StateForKyotoshogiX * StateForKyotoshogiY * StateForKyotoshogiZ);
    std::fill(_features.begin(), _features.end(), 0);

    init();
    _status = (GameStatus)opponent(turn);
    findFeatures();
    findActions(turn);
    fillFullFeatures();
  }

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<StateForKyotoshogi>(*this);
  }

  int type_to_z(KSPiece p) {
    if (!p.promoted)
      return (int)p.type - 1;
    switch (p.type) {
    case Tokin_Lance:
      return PTokin_Lance;
    case Silver_Bishop:
      return PSilver_Bishop;
    case Gold_Knight:
      return PGold_Knight;
    case Pawn_Rook:
      return PPawn_Rook;
    case Tokin_Lance2:
      return PTokin_Lance2;
    case Silver_Bishop2:
      return PSilver_Bishop2;
    case Gold_Knight2:
      return PGold_Knight2;
    case Pawn_Rook2:
      return PPawn_Rook2;

    default:
      fprintf(stderr, "%s type to z error %d\n", p.print().c_str(), (int)p.type);
      break;
    }
    return -1;
  }

  KSPieceType z_to_type(int z) const {
    switch ((KSPieceZ)z) {
    case KSK:
      return KSKing;
    case GK:
    case PGold_Knight:
      return Gold_Knight;
    case GK2:
    case PGold_Knight2:
      return Gold_Knight2;
    case PSilver_Bishop:
    case SB:
      return Silver_Bishop;
    case PSilver_Bishop2:
    case SB2:
      return Silver_Bishop2;
    case PPawn_Rook:
    case PR:
      return Pawn_Rook;
    case PPawn_Rook2:
    case PR2:
      return Pawn_Rook2;
    case PTokin_Lance2:
    case TL2:
      return Tokin_Lance2;
    case PTokin_Lance:
    case TL:
      return Tokin_Lance;

    default:
      fprintf(stderr, "z %d to type error\n", z);
      break;
    }
    return KSNone;
  }

  bool z_promoted(int z) const {
    return z >= 9;
  }

  void findActions(int color) {
    std::vector<KSMove> moves;

    legalKSMoves(color, moves);

    int nb = moves.size();
    _legalActions.clear();

    for (int i = 0; i < nb; ++i) {
      int x = moves[i].pos1.x;
      int y = moves[i].pos1.y;
      int z = type_to_z(moves[i].piece);

      _legalActions.push_back(std::make_shared<ActionForKyotoshogi>(x, y, z));
      _legalActions[i]->SetIndex(i);
    }
  }

  void findFeatures() {
    std::vector<float> old(_features);
    for (int i = 0; i < 4865; ++i)
      _features[i] = 0;
    // 0 ~ 500
    for (int i = 0; i < 25; ++i) {
      KSPiece p = board[i % 5][i / 5];
      if (p.color == KyotoWhite) {
        switch (p.type) {
        case KSKing:
          _features[i] = 1;
          break;

        case Gold_Knight:
        case Gold_Knight2:
          if (p.promoted)
            _features[25 + i] = 1;
          else
            _features[50 + i] = 1;
          break;

        case Silver_Bishop:
        case Silver_Bishop2:
          if (p.promoted)
            _features[75 + i] = 1;
          else
            _features[100 + i] = 1;
          break;

        case Pawn_Rook:
        case Pawn_Rook2:
          if (p.promoted)
            _features[125 + i] = 1;
          else
            _features[150 + i] = 1;
          break;

        case Tokin_Lance:
        case Tokin_Lance2:
          if (p.promoted)
            _features[175 + i] = 1;
          else
            _features[200 + i] = 1;
          break;

        default:
          break;
        }
      } else {
        switch (p.type) {
        case KSKing:
          _features[225 + i] = 1;
          break;

        case Gold_Knight:
        case Gold_Knight2:
          if (p.promoted)
            _features[250 + i] = 1;
          else
            _features[275 + i] = 1;
          break;

        case Silver_Bishop:
        case Silver_Bishop2:
          if (p.promoted)
            _features[300 + i] = 1;
          else
            _features[325 + i] = 1;
          break;

        case Pawn_Rook:
        case Pawn_Rook2:
          if (p.promoted)
            _features[350 + i] = 1;
          else
            _features[375 + i] = 1;
          break;

        case Tokin_Lance:
        case Tokin_Lance2:
          if (p.promoted)
            _features[400 + i] = 1;
          else
            _features[425 + i] = 1;
          break;

        default:
          break;
        }
      }
    }

    // 450 ~ 525
    switch (Repetition) {
    case 1:
      std::fill(_features.begin() + 450, _features.begin() + 475, 1);
      break;
    case 5:
      std::fill(_features.begin() + 475, _features.begin() + 500, 1);
      break;
    case 9:
      std::fill(_features.begin() + 500, _features.begin() + 525, 1);
      break;
    default:
      break;
    }

    int tmp = 525;
    for (int i = 0; i < 2; ++i) {
      std::vector<KSPiece>::iterator it;
      for (it = chess[i].begin(); it != chess[i].end(); ++it) {
        if (!(*it).pos.on_board()) {
          switch ((*it).type) {
          case Gold_Knight:
            std::fill(_features.begin() + tmp, _features.begin() + tmp + 5, 1);
            break;
          case Silver_Bishop:
            std::fill(
                _features.begin() + tmp + 5, _features.begin() + tmp + 10, 1);
            break;
          case Pawn_Rook:
            std::fill(
                _features.begin() + tmp + 10, _features.begin() + tmp + 15, 1);
            break;
          case Tokin_Lance:
            std::fill(
                _features.begin() + tmp + 15, _features.begin() + tmp + 20, 1);
            break;
          case Gold_Knight2:
            std::fill(
                _features.begin() + tmp + 25, _features.begin() + tmp + 30, 1);
            break;
          case Silver_Bishop2:
            std::fill(
                _features.begin() + tmp + 30, _features.begin() + tmp + 35, 1);
            break;
          case Pawn_Rook2:
            std::fill(
                _features.begin() + tmp + 35, _features.begin() + tmp + 40, 1);
            break;
          case Tokin_Lance2:
            std::fill(
                _features.begin() + tmp + 40, _features.begin() + tmp + 45, 1);
            break;
          default:
            break;
          }
        }
      }
      std::fill(_features.begin() + tmp + 20, _features.begin() + tmp + 25, 0);
      std::fill(_features.begin() + tmp + 45, _features.begin() + tmp + 50, 0);
      tmp += 50;
    }

    // history 625 ~ 4375+625
    std::copy(old.begin(), old.begin() + 4375, _features.begin() + 625);
    // 5000~5025
    std::fill(_features.begin() + 5000, _features.end(), turn);
  }
  // The action just decreases the distance and swaps the turn to play.
  virtual void ApplyAction(const _Action& action) override {
    KSMove m;
    if ((GameStatus)_status == GameStatus::player1Turn) {  // KyotoWhite to move
      m.piece.color = KyotoWhite;
      m.pos1 = KSPosition(action.GetY(), action.GetZ());
      m.piece.type = z_to_type(action.GetX());
      m.piece.promoted = z_promoted(action.GetX());

      std::vector<KSPiece>::iterator it;
      for (it = chess[KyotoWhite].begin(); it != chess[KyotoWhite].end();
           ++it) {
        if ((*it).type == m.piece.type)
          m.piece.pos = (*it).pos;
      }

      play(m);
      findActions(KyotoBlack);
      if ((GameStatus)_status == GameStatus::tie || fourfold()) {
        _status = GameStatus::tie;
        _legalActions.clear();
      } else if (won(KyotoWhite) || _legalActions.empty())
        _status = GameStatus::player1Win;  // KyotoWhite win
      else
        _status = GameStatus::player0Turn;  // KyotoBlack turn
    } else {                               // KyotoBlack
      m.piece.color = KyotoBlack;
      m.pos1 = KSPosition(action.GetY(), action.GetZ());
      m.piece.type = z_to_type(action.GetX());
      m.piece.promoted = z_promoted(action.GetX());

      std::vector<KSPiece>::iterator it;
      for (it = chess[KyotoBlack].begin(); it != chess[KyotoBlack].end();
           ++it) {
        if ((*it).type == m.piece.type)
          m.piece.pos = (*it).pos;
      }

      play(m);
      findActions(KyotoWhite);
      if ((GameStatus)_status == GameStatus::tie || fourfold()) {
        _status = GameStatus::tie;
        _legalActions.clear();
      } else if (won(KyotoBlack) || _legalActions.empty())
        _status = GameStatus::player0Win;  // KyotoBlack won
      else
        _status = GameStatus::player1Turn;  // KyotoWhite turn
    }
    findFeatures();
    _hash = hash;

    if (repet.size() == 4) {
      repet.pop();
      repet.push(_hash);
    } else
      repet.push(_hash);
    
    fillFullFeatures();
  }

  // For this trivial example we just compare to random play.
  virtual void DoGoodAction() override {
    DoRandomAction();
  }
};
