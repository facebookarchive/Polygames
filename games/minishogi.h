/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author1: Maria Elsa
// - Github: https://github.com/melsaa
// - Email:  m_elsa@ymail.com
// Author2: Lin Hsin-I
// - Github: https://github.com/free00000000000
// - Email:  410521233@gms.ndhu.edu.tw
// Facilitator: 邱顯棟 (Xiǎn-Dòng Qiū)
// - Github: https://github.com/YumJelly
// - Email:  yumjelly@gmail.com

#pragma once
#include "../core/state.h"
#include "shogi.h"
#include <queue>
#include <sstream>
#include <vector>

#include <mutex>

#include <fmt/printf.h>

// class ActionForMinishogi : public _Action {
//  public:
//    ActionForMinishogi(int x, int y, int p, size_t index) : _Action() {
//        _loc[0] = p;
//        _loc[1] = x;
//        _loc[2] = y;
//        _hash = (x + y * 5) * 19 + p;
//        _i = (int)index;
//    }
//};

template <int version = 2>
class StateForMinishogi : public State, public Shogi {
 public:
  static inline unsigned long long HashArray[2][10][Dx][Dy];
  static inline unsigned long long HashArrayJail[20];
  static inline unsigned long long HashTurn;
  int length;

  bool sennichite;
  std::array<int, 2> checkCount;
  std::array<std::vector<std::pair<uint64_t, char>>, 16> repetitions;

  StateForMinishogi(int seed)
      : State(seed)
      , Shogi() {
  }

  virtual void Initialize() override {
    _moves.clear();
    // _hash = 2166136261u;
    _hash = 0;
    _status = GameStatus::player0Turn;
    _featSize[0] = 217;
    if (version == 2) {
      _featSize[0] = (6 + 4 + 6) * 2;  // 6 pieces + 4 promoted + 6 off board (counts)
    }
    _featSize[1] = Dy;
    _featSize[2] = Dx;
    _actionSize[0] = 19;  // 11 pieces + 8 promoted
    if (version == 2) {
      _actionSize[0] = 6;
    }
    _actionSize[1] = Dy;
    _actionSize[2] = Dx;
    _features.clear();
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
    // setFeatures(false, false, false, 0, 0, false);

    gameInit();
    static std::once_flag initFlag;
    std::call_once(initFlag, initHash);
    // printCurrentBoard();

    findFeature();
    findActions();
    // fixxx
    fillFullFeatures();
  }

  void gameInit() {
    chess.clear();
    chess.resize(2);
    for (int i = 0; i < Dx; ++i)
      for (int j = 0; j < Dy; ++j)
        board[i][j] = Piece();

    for (int i = 0; i < Dx; ++i) {
      board[i][0] = Piece(White, PieceType(i + 1), false, Position(i, 0));
      chess[White].push_back(board[i][0]);
    }
    board[0][1] = Piece(White, PieceType::Pawn, false, Position(0, 1));
    chess[White].push_back(board[0][1]);

    for (int i = 1; i <= Dx; ++i) {
      int x = Dx - i;
      board[x][4] = Piece(Black, PieceType(i), false, Position(x, 4));
      chess[Black].push_back(board[x][4]);
    }
    board[4][3] = Piece(Black, PieceType::Pawn, false, Position(4, 3));
    chess[Black].push_back(board[4][3]);

    length = 0;
    _hash = 0;
    sennichite = false;
    checkCount = {0, 0};
    for (auto& v : repetitions) {
      v.clear();
    }
    repetitions[_hash % repetitions.size()].emplace_back(_hash, 0);
  }

  static void initHash() {
    static std::minstd_rand _rng(
        std::chrono::steady_clock::now().time_since_epoch().count());
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 10; ++b)
        for (int c = 0; c < 5; ++c)
          for (int d = 0; d < 5; ++d) {
            HashArray[a][b][c][d] = 0;
            for (int k = 0; k < 64; ++k)
              if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
                HashArray[a][b][c][d] |= (1ULL << k);
          }
    for (int a = 0; a < 20; ++a) {
      for (int k = 0; k < 64; ++k)
        if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
          HashArrayJail[a] |= (1ULL << k);
    }

    HashTurn = 0;
    for (int k = 0; k < 64; k++)
      if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
        HashTurn |= (1ULL << k);
  }

  void findFeature() {
    if (version == 2) {
      std::fill(_features.begin(), _features.end(), 0);
      for (auto& v : chess) {
        for (const Piece& p : v) {
          if (p.pos.on_board()) {
            int x = p.pos.x;
            int y = p.pos.y;
            size_t offset = y * Dx + x;
            size_t index = (int)p.type - 1;
            if (p.color == Black) {
              index += 6 + 4;
            }
            _features[offset + Dx * Dy * index] = 1.0f;
            if (p.promoted) {
              index = 6 + (int)p.type - 3;
              if (p.color == Black) {
                index += 6 + 4;
              }
              _features[offset + Dx * Dy * index] = 1.0f;
            }
          } else {
            size_t index = (6 + 4) * 2 + (int)p.type - 1;
            if (p.color == Black) {
              index += 6;
            }
            size_t begin = Dx * Dy * index;
            size_t end = Dx * Dy * (index + 1);
            for (size_t i = begin; i != end; ++i) {
              _features[i] += 1.0f;
            }
          }
        }
      }
      return;
    }

    std::vector<float> old(_features);
    for (int i = 0; i < 5425; ++i)
      _features[i] = 0;
    // 0 ~ 500
    for (int i = 0; i < 25; ++i) {
      Piece p = board[i % 5][i / 5];
      if (p.color == White) {
        switch (p.type) {
        case PieceType::King:
          _features[i] = 1;
          break;

        case PieceType::Gold:
        case PieceType::Gold2:
          _features[25 + i] = 1;
          break;

        case PieceType::Silver:
        case PieceType::Silver2:
          if (p.promoted)
            _features[50 + i] = 1;
          else
            _features[75 + i] = 1;
          break;

        case PieceType::Bishop:
        case PieceType::Bishop2:
          if (p.promoted)
            _features[100 + i] = 1;
          else
            _features[125 + i] = 1;
          break;

        case PieceType::Rook:
        case PieceType::Rook2:
          if (p.promoted)
            _features[150 + i] = 1;
          else
            _features[175 + i] = 1;
          break;

        case PieceType::Pawn:
        case PieceType::Pawn2:
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
        case PieceType::King:
          _features[250 + i] = 1;
          break;

        case PieceType::Gold:
        case PieceType::Gold2:
          _features[275 + i] = 1;
          break;

        case PieceType::Silver:
        case PieceType::Silver2:
          if (p.promoted)
            _features[300 + i] = 1;
          else
            _features[325 + i] = 1;
          break;

        case PieceType::Bishop:
        case PieceType::Bishop2:
          if (p.promoted)
            _features[350 + i] = 1;
          else
            _features[375 + i] = 1;
          break;

        case PieceType::Rook:
        case PieceType::Rook2:
          if (p.promoted)
            _features[400 + i] = 1;
          else
            _features[425 + i] = 1;
          break;

        case PieceType::Pawn:
        case PieceType::Pawn2:
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

//    // 500 ~ 575
//    switch (repeat) {
//    case 1:
//      std::fill(_features.begin() + 500, _features.begin() + 525, 1);
//      break;
//    case 5:
//      std::fill(_features.begin() + 525, _features.begin() + 550, 1);
//      break;
//    case 9:
//      std::fill(_features.begin() + 550, _features.begin() + 575, 1);
//      break;
//    default:
//      break;
//    }

    // prison w 575 ~ 625
    // prison b 625 ~ 675
    int tmp = 575;
    for (int i = 0; i < 2; ++i) {
      std::vector<Piece>::iterator it;
      for (it = chess[i].begin(); it != chess[i].end(); ++it) {
        if (!(*it).pos.on_board()) {
          switch ((*it).type) {
          case PieceType::Gold:
            std::fill(_features.begin() + tmp, _features.begin() + tmp + 5, 1);
            break;
          case PieceType::Silver:
            std::fill(
                _features.begin() + tmp + 5, _features.begin() + tmp + 10, 1);
            break;
          case PieceType::Bishop:
            std::fill(
                _features.begin() + tmp + 10, _features.begin() + tmp + 15, 1);
            break;
          case PieceType::Rook:
            std::fill(
                _features.begin() + tmp + 15, _features.begin() + tmp + 20, 1);
            break;
          case PieceType::Pawn:
            std::fill(
                _features.begin() + tmp + 20, _features.begin() + tmp + 25, 1);
            break;
          case PieceType::Gold2:
            std::fill(
                _features.begin() + tmp + 25, _features.begin() + tmp + 30, 1);
            break;
          case PieceType::Silver2:
            std::fill(
                _features.begin() + tmp + 30, _features.begin() + tmp + 35, 1);
            break;
          case PieceType::Bishop2:
            std::fill(
                _features.begin() + tmp + 35, _features.begin() + tmp + 40, 1);
            break;
          case PieceType::Rook2:
            std::fill(
                _features.begin() + tmp + 40, _features.begin() + tmp + 45, 1);
            break;
          case PieceType::Pawn2:
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

    // history 675 ~ 4725+675
    std::copy(old.begin(), old.begin() + 4725, _features.begin() + 675);

    // 5400 ~ 5425
    std::fill(_features.begin() + 5400, _features.end(), (int)_status);
  }

  std::vector<Move> moves;

  void findActions() {
    // fprintf(stderr, "find action\n");
    // fprintf(stderr, "color: %d\n", (int)_status);

    moves.clear();

    auto& list = chess[(int)_status];
    for (size_t i = 0; i != list.size(); ++i) {
      const Piece& p = list[i];
      if (!p.pos.on_board()) {
        bool duplicate = false;
        for (size_t i2 = 0; i2 != i; ++i2) {
          const Piece& p2 = list[i2];
          if (p2.type == p.type && !p2.pos.on_board()) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) {
          continue;
        }
      }
      legalMoves(p, moves);
    }

    int i = 0;
    _NewlegalActions.clear();
    for (auto m : moves) {
      m.piece.promoted = m.promote;
      // std::cerr << m.piece.print();
      // fprintf(stderr, " (%c, %d) to (%c, %d) ---%d\n", m.piece.pos.x+'A',
      // m.piece.pos.y, m.next.x+'A', m.next.y, i);

      if (version == 2) {
        _NewlegalActions.emplace_back(
            _NewlegalActions.size(), (int)m.piece.type - 1, m.next.y, m.next.x);
        //fmt::printf("FindAction found action %s\n", actionDescription(_NewlegalActions.back()));
      } else {

        int x = m.next.x;
        int y = m.next.y;
        int z = type_to_z(m.piece);

        _NewlegalActions.emplace_back(_NewlegalActions.size(), z, x, y);
      }
      //            _legalActions.push_back(
      //                std::make_shared<ActionForMinishogi>(x, y, z,
      //                _legalActions.size())
      //            );
      i++;
    }

    // fprintf(stderr, "end find action\n");
  }

  virtual void printCurrentBoard() const override {
    std::cerr << stateDescription();
    // for(int i=0; i<2; ++i) {
    //     for(auto j : chess[i]) {
    //         fprintf(stderr, "(%c,%d) ", j.pos.x+'A', j.pos.y);
    //     }
    //     std::cerr << std::endl;
    // }
  }

  std::string print_chess(const int color) const {
    std::string str;
    if (color == White)
      str += "MiniWhite: ";
    else
      str += "MiniBlack: ";
    for (auto i : chess[color]) {
      if (!i.pos.on_board()) {
        str += '(';
        str += i.print();
        str += ')';
      } else
        str += i.print();
      str += ' ';
    }
    str += '\n';
    return str;
  }

  virtual std::string stateDescription() const override {
    std::string str;
    str += "   A| B| C| D| E\n";
    for (int i = Dy - 1; i >= 0; --i) {
      str += std::to_string(i + 1) + ' ';
      for (int j = 0; j < Dx; ++j) {
        if (j > 0) {
          str += '|';
        }
        auto x = board[j][i].print();
        if (x.size() == 1) {
          str += ' ';
        }
        str += x;
      }
      str += '\n';
    }
    str += print_chess(White);
    str += print_chess(Black);

    return str;
  }


  virtual std::string actionDescription(const _Action& action) const {
    const Move& move = moves.at(action.GetIndex());

    const Piece& p = move.piece;

    bool disy = false;
    bool disx = false;
    if (p.pos.on_board()) {
      for (const Move& m : moves) {
        if ((m.piece.type == p.type || new_type(m.piece.type) == p.type) && m.piece.pos.on_board() && m.piece.pos != p.pos && m.next == move.next) {
          if (m.piece.pos.x == p.pos.x) {
            disy = true;
          } else {
            disx = true;
          }
        }
      }
    }

    std::string s = p.print();
    for (auto& v : s) {
      v = std::toupper(v);
    }
    if (disx) {
      s += char('a' + p.pos.x);
    }
    if (disy) {
      s += std::to_string(1 + p.pos.y);
    }
    if (board[move.next.x][move.next.y].color != Empty) {
      s += 'x';
    }
    if (!p.pos.on_board()) {
      s += '@';
    }
    s += 'a' + move.next.x;
    s += std::to_string(1 + move.next.y);
    if (move.promote) {
      s += '+';
    }
    return s;
  }

  virtual std::unique_ptr<mcts::State> clone_() const override {
    return std::make_unique<StateForMinishogi>(*this);
  }

  int getHashNum(Piece p) {
    int num = (int)p.type;
    if (num >= 7)
      num -= 5;
    if (p.promoted)
      num += 5;
    //   7 8 9 10 11
    // 1 2 3 4  5  6 | 7 8 9 10
    num -= 1;
    return num;
  }

  int getHashNumjail(Piece p) {
    // 0~19
    return (int)p.type - 2 + 10 * p.color;
  }

  void test(std::string str) {
    bool bad = false;
    for (auto& v : chess) {
      for (Piece& p : v) {
        if (p.pos.on_board()) {
          Piece& x = board[p.pos.x][p.pos.y];
          if (x.color != p.color || x.type != p.type || x.promoted != p.promoted || !(x.pos == p.pos)) {
            fmt::printf("piece not found on board %d %d %d %d %d\n", p.color, (int)p.type, p.promoted, p.pos.x, p.pos.y);
            bad = true;
          }
        }
      }
    }
    for (int y = 0; y != 5; ++y) {
      for (int xx = 0; xx != 5; ++xx) {
        Piece& x = board[xx][y];
        if (x.color != Empty) {
          bool found = false;
          for (auto& p : chess[x.color]) {
            if (x.color == p.color && x.type == p.type && x.promoted == p.promoted && x.pos == p.pos) {
              found = true;
            }
          }
          if (!found) {
            fmt::printf("board piece not found %d %d %d %d %d\n", x.color, (int)x.type, x.promoted, x.pos.x, x.pos.y);
            bad = true;
          }
        }
      }
    }
    if (bad) {
      for (auto& v : chess) {
        for (Piece& x : v) {
          fmt::printf("piece %d %d %d %d %d\n", x.color, (int)x.type, x.promoted, x.pos.x, x.pos.y);
        }
      }
      fmt::printf("%s\n", stateDescription());
      throw std::runtime_error("bad " + str);
    }
    fmt::printf("%s passed\n", str);
  }

  void play(Move m) {
    // std::cerr << m.piece.print();
    // fprintf(stderr, " play (%c, %d) to (%c, %d)\n\n", m.piece.pos.x+'A',
    // m.piece.pos.y, m.next.x+'A', m.next.y);
    //test("play enter");
    m.piece.promoted |= m.promote;

    //printf("play %d %d %d %d   %d %d\n", m.piece.color, (int)m.piece.type, m.piece.pos.x, m.piece.pos.y, m.next.x, m.next.y);

    if (m.piece.pos.on_board()) {
      _hash ^= HashArray[m.piece.color][getHashNum(m.piece)][m.piece.pos.x]
                        [m.piece.pos.y];
      // eat
      if (board[m.next.x][m.next.y].color != Empty) {
        //test("enter eat");
        //auto& x = board[m.next.x][m.next.y];
        //printf("eat %d %d %d %d\n", x.color, (int)x.type, x.pos.x, x.pos.y);
        int opp = opponent(m.piece.color);
        _hash ^= HashArray[opp][getHashNum(board[m.next.x][m.next.y])][m.next.x]
                          [m.next.y];
        _hash ^= HashArrayJail[getHashNumjail(board[m.next.x][m.next.y])];

        auto type = board[m.next.x][m.next.y].type;
        if (version == 1) {
          type = new_type(type);
        }
        //test("eat pre push");
        Piece tmp(m.piece.color, type, false);
        chess[m.piece.color].push_back(tmp);

        bool found = false;
        std::vector<Piece>::iterator it;
        for (it = chess[opp].begin(); it != chess[opp].end(); ++it) {
          if (it->pos == m.next) {
            chess[opp].erase(it);
            found = true;
            break;
          }
        }
        if (!found) throw std::runtime_error("Could not find piece to erase");
      }

      std::vector<Piece>::iterator it;
      bool found = false;
      for (it = chess[m.piece.color].begin(); it != chess[m.piece.color].end();
           ++it) {
        if ((*it).pos == m.piece.pos) {
          (*it).pos = m.next;
          // decide promoted
          if (m.piece.promoted) {
            (*it).promoted = true;
          }

//          if (it->type != m.piece.type || it->color != m.piece.color) {
//            fmt::printf("it is %d %d %d %d\n", it->color, (int)it->type, it->pos.x, it->pos.y);
//            fmt::printf("m.piece is %d %d %d %d\n", m.piece.color, (int)m.piece.type, m.piece.pos.x, m.piece.pos.y);
//            throw std::runtime_error("piece mismatch");
//          }

          //printf("piece moved\n");
          found = true;

          board[m.next.x][m.next.y] = (*it);
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          break;
        }
      }
      if (!found) {
        throw std::runtime_error("could not find piece to move");
      }
      //test("post move");
    } else {  // Drop move
      //test("pre drop");
      _hash ^= HashArrayJail[getHashNumjail(m.piece)];
      std::vector<Piece>::iterator it;
      for (it = chess[m.piece.color].begin(); it != chess[m.piece.color].end();
           ++it) {
        if ((*it).type == m.piece.type && !it->pos.on_board()) {
          (*it).pos = m.next;
          board[m.next.x][m.next.y] = (*it);
          break;
        }
      }
      //test("post drop");
    }
    _hash ^= HashArray[m.piece.color][getHashNum(board[m.next.x][m.next.y])]
                      [m.next.x][m.next.y];
    _hash ^= HashTurn;

    if (length < MaxPlayoutLength) {
      // rollout[length] = m;
      length++;
    } else {
      // set draw when the moves bigger than 1000
      _status = GameStatus::tie;
    }

    //test("pre repeat");

    // find repeat
    bool found = false;
    size_t index = _hash % repetitions.size();
    for (auto& v : repetitions[index]) {
      if (v.first == _hash) {
        ++v.second;
        if (v.second >= 4) {
          sennichite = true;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      repetitions[index].emplace_back(_hash, 1);
    }

    for (auto i : chess[opponent(m.piece.color)]) {
      if (i.type == PieceType::King) {
        if (check(i.pos, m.piece.color)) {
          ++checkCount.at(m.piece.color);
        } else {
          checkCount.at(m.piece.color) = 0;
        }
        break;
      }
    }

  }

  virtual void ApplyAction(const _Action& action) override {
    // fprintf(stderr, "\nApply Action %d\n", (int)_status);

    //fmt::printf("%s\n", stateDescription());

    //fmt::printf(" ApplyAction %d index %d, %s\n", (int)_status, action.GetIndex(), actionDescription(action));

    play(moves.at(action.GetIndex()));
    if (_status == GameStatus::player0Turn ||
        _status == GameStatus::player1Turn) {
      _status = _status == GameStatus::player0Turn ? GameStatus::player1Turn
                                                   : GameStatus::player0Turn;
      findActions();
      //printf("there are %d moves yey\n", moves.size());
      if (moves.empty()) {
        _status = _status == GameStatus::player1Turn ? GameStatus::player0Win
                                                     : GameStatus::player1Win;
      } else if (sennichite) {
        if (checkCount[White] >= 4) {
          _status = GameStatus::player1Win;
        } else if (checkCount[Black] >= 4) {
          _status = GameStatus::player0Win;
        } else {
          _status = GameStatus::player1Win;
        }
      }
    }
    if (_status == GameStatus::player0Turn ||
        _status == GameStatus::player1Turn) {
      findFeature();
      //    fixxx
      fillFullFeatures();
    } else {
      _NewlegalActions.clear();
      // if(_status == GameStatus::player0Win)
      //     fprintf(stderr, "white win\n");
      // else if(_status == GameStatus::player1Win)
      //     fprintf(stderr, "black win\n");
      // else fprintf(stderr, "tie\n");
    }
    // fprintf(stderr, "end apply action\n");
  }

  virtual void DoGoodAction() override {
    // int i;
    // printCurrentBoard();
    // std::cout << actionsDescription();
    // std::cin >> i;
    // _Action a = *(_legalActions[i].get());
    // ApplyAction(a);
    // std::cout << actionDescription(a);

    return DoRandomAction();
  }
};
