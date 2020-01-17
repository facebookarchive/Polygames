/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../core/state.h"
#include <algorithm>

class ActionForBlockGo : public _Action {
 public:
  ActionForBlockGo(int x, int y, int p, size_t index)
      : _Action() {
    _loc[0] = p;
    _loc[1] = x;
    _loc[2] = y;
    _hash = (uint32_t)(x + y * 13) * 36 + p;
    _i = (int)index;
  }
};

class StateForBlockGo : public State {
 public:
  class Point {
   public:
    int x, y;

    Point(int X, int Y) {
      x = X;
      y = Y;
    }

    Point() {
      x = y = 0;
    }

    void setxy(int X, int Y) {
      x = X, y = Y;
    }

    void turn90() {
      int tmp = x;
      x = -y;
      y = tmp;
    }
  };

  class Piece {
   public:
    friend class StateForBlockGo;
    short count;  // 1~4
    short turn;
    bool onboard;
    Point s;
    Point tail[4];

    Piece() {
      count = 1;
      turn = 4;  // 0 90 180 270
      onboard = false;
    }

    void turn90() {
      for (int i = 1; i < count; ++i)
        tail[i].turn90();
    }
  };

  class Move {
   public:
    int x, y;
    int piece;
    int dir;
    Move() {
      x = y = -1;
      piece = -1;
      dir = 0;
    }
    Move(int X, int Y, int p, int d) {
      x = X;
      y = Y;
      piece = p;
      dir = d;
    }
  };

 public:
  const int boardWidth = 13;
  const int boardHeight = 13;
  int board[13][13];
  int territory[13][13];
  int round;
  unsigned long long HashArray[20][13][13];  // 20 different types
  unsigned long long HashTurn;
  std::vector<Piece> player0;
  std::vector<Piece> player1;
  std::vector<Move> moves;

  StateForBlockGo(int seed)
      : State(seed) {
    Initialize();
  }

  virtual void Initialize() override {
    _moves.clear();
    // _hash = 2166136261u;
    _hash = 0;
    _status = GameStatus::player0Turn;
    _featSize[0] = 2;
    _featSize[1] = boardHeight;
    _featSize[2] = boardWidth;
    _actionSize[0] = 36;  // 9 pieces * 4 directions
    _actionSize[1] = boardHeight;
    _actionSize[2] = boardWidth;
    _features.clear();
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);

    gameInit();
    initHash();
    // printCurrentBoard();

    findFeature();
    findActions();
    fillFullFeatures();
  }

  void initHash() {
    for (int x = 0; x < 20; ++x)
      for (int y = 0; y < boardHeight; ++y)
        for (int z = 0; z < boardWidth; ++z) {
          HashArray[x][y][z] = 0;
          for (int k = 0; k < 64; ++k)
            if ((rand() / (RAND_MAX + 1.0)) > 0.5)
              HashArray[x][y][z] |= (1ULL << k);
        }
    HashTurn = 0;
    for (int k = 0; k < 64; k++)
      if ((rand() / (RAND_MAX + 1.0)) > 0.5)
        HashTurn |= (1ULL << k);
  }

  void gameInit() {
    round = 1;
    std::fill(&board[0][0], &board[0][0] + 169, 2);
    player0.clear();
    player0.resize(9);
    player1.clear();
    player1.resize(9);

    // s
    player0[0].turn = 1;
    player0[0].count = 1;
    // s
    player0[1].turn = 1;
    player0[1].count = 1;
    // s+
    // ++
    player0[2].turn = 1;
    player0[2].count = 4;
    player0[2].tail[1].setxy(1, 0);
    player0[2].tail[2].setxy(0, 1);
    player0[2].tail[3].setxy(1, 1);
    // s+++
    player0[3].turn = 2;
    player0[3].count = 4;
    player0[3].tail[1].setxy(1, 0);
    player0[3].tail[2].setxy(2, 0);
    player0[3].tail[3].setxy(3, 0);
    // s+
    //  ++
    player0[4].turn = 2;
    player0[4].count = 4;
    player0[4].tail[1].setxy(1, 0);
    player0[4].tail[2].setxy(1, 1);
    player0[4].tail[3].setxy(2, 1);
    //  ++
    // s+
    player0[5].turn = 2;
    player0[5].count = 4;
    player0[5].tail[1].setxy(1, 0);
    player0[5].tail[2].setxy(1, -1);
    player0[5].tail[3].setxy(2, -1);
    // s++
    //  +
    player0[6].turn = 4;
    player0[6].count = 4;
    player0[6].tail[1].setxy(1, 0);
    player0[6].tail[2].setxy(2, 0);
    player0[6].tail[3].setxy(1, 1);
    // s++
    //   +
    player0[7].turn = 4;
    player0[7].count = 4;
    player0[7].tail[1].setxy(1, 0);
    player0[7].tail[2].setxy(2, 0);
    player0[7].tail[3].setxy(2, 1);
    // s++
    // +
    player0[8].turn = 4;
    player0[8].count = 4;
    player0[8].tail[1].setxy(1, 0);
    player0[8].tail[2].setxy(2, 0);
    player0[8].tail[3].setxy(0, 1);

    // ============= //

    // s
    player1[0].turn = 1;
    player1[0].count = 1;
    // s
    player1[1].turn = 1;
    player1[1].count = 1;
    // s+
    // ++
    player1[2].turn = 1;
    player1[2].count = 4;
    player1[2].tail[1].setxy(1, 0);
    player1[2].tail[2].setxy(0, 1);
    player1[2].tail[3].setxy(1, 1);
    // s+++
    player1[3].turn = 2;
    player1[3].count = 4;
    player1[3].tail[1].setxy(1, 0);
    player1[3].tail[2].setxy(2, 0);
    player1[3].tail[3].setxy(3, 0);
    // s+
    //  ++
    player1[4].turn = 2;
    player1[4].count = 4;
    player1[4].tail[1].setxy(1, 0);
    player1[4].tail[2].setxy(1, 1);
    player1[4].tail[3].setxy(2, 1);
    //  ++
    // s+
    player1[5].turn = 2;
    player1[5].count = 4;
    player1[5].tail[1].setxy(1, 0);
    player1[5].tail[2].setxy(1, -1);
    player1[5].tail[3].setxy(2, -1);
    // s++
    //  +
    player1[6].turn = 4;
    player1[6].count = 4;
    player1[6].tail[1].setxy(1, 0);
    player1[6].tail[2].setxy(2, 0);
    player1[6].tail[3].setxy(1, 1);
    // s++
    //   +
    player1[7].turn = 4;
    player1[7].count = 4;
    player1[7].tail[1].setxy(1, 0);
    player1[7].tail[2].setxy(2, 0);
    player1[7].tail[3].setxy(2, 1);
    // s++
    // +
    player1[8].turn = 4;
    player1[8].count = 4;
    player1[8].tail[1].setxy(1, 0);
    player1[8].tail[2].setxy(2, 0);
    player1[8].tail[3].setxy(0, 1);
  }

  virtual std::unique_ptr<mcts::State> clone_() const override {
    return std::make_unique<StateForBlockGo>(*this);
  }

  virtual void printCurrentBoard() const override {
    fprintf(stderr, "   0  1  2  3  4  5  6  7  8  9  10 11 12\n");
    for (int i = 0; i < boardHeight; ++i) {
      fprintf(stderr, "%2d", i);
      for (int j = 0; j < boardWidth; ++j) {
        switch (board[i][j]) {
        case 0:
          std::cerr << " O ";
          break;
        case 1:
          std::cerr << " X ";
          break;
        default:
          std::cerr << " - ";
          break;
        }
      }
      std::cerr << std::endl;
    }
  }

  void findFeature() {
    std::fill(_features.begin(), _features.end(), 0);
    for (int i = 0; i < 169; ++i) {
      switch (board[i / 13][i % 13]) {
      case 0:
        _features[i] = 1;
        break;
      case 1:
        _features[169 + i] = 1;
        break;
      default:
        break;
      }
    }
    // std::fill(_features.begin()+338, _features.end(), getCurrentPlayer());
  }

  bool canDrop(int x, int y) {
    return (x >= 0 && y >= 0 && x < boardWidth && y < boardHeight &&
            board[y][x] == 2);
  }

  void legalMoves(std::vector<Piece> player, std::vector<Move>& moves) {
    if (round <= 4) {
      int dx[] = {3, 9, 3, 9};
      int dy[] = {3, 3, 9, 9};
      for (int d = 0; d < 4; ++d) {
        if (board[dy[d]][dx[d]] == 2) {  // empty
          for (size_t p = 0; p < player.size(); ++p) {
            if (!player[p].onboard) {
              for (int t = 0; t < player[p].turn; ++t) {
                for (int i = 0; i < player[p].count; ++i) {
                  Move* m = new Move();
                  m->dir = t;
                  m->piece = p;
                  m->x = dx[d] - player[p].tail[i].x;
                  m->y = dy[d] - player[p].tail[i].y;
                  // fprintf(stderr, "%d %d || %d %d\n", m->x, m->y,
                  // player[p].tail[i].x, player[p].tail[i].y);
                  moves.push_back(*m);
                  delete m;
                }
                // fprintf(stderr, "\n\n");
                player[p].turn90();
              }

              if (p >= 3 && p <= 5) {
                player[p].turn90();
                player[p].turn90();
              } else if (p == 2) {
                player[p].turn90();
                player[p].turn90();
                player[p].turn90();
              }

            } else
              continue;
          }
        }
      }
    } else {
      bool visited[boardHeight][boardWidth];
      for (int j = 0; j < boardHeight; ++j)
        for (int i = 0; i < boardWidth; ++i)
          visited[j][i] = false;

      for (int j = 0; j < boardHeight; ++j) {
        for (int i = 0; i < boardWidth; ++i) {
          if ((int)_status == board[j][i]) {
            int dx[] = {1, 0, -1, 0};
            int dy[] = {0, 1, 0, -1};

            for (int k = 0; k < 4; ++k) {
              int x = i + dx[k];
              int y = j + dy[k];
              if (canDrop(x, y))
                visited[y][x] = true;
            }
          }
        }
      }

      for (size_t p = 0; p < player.size(); ++p) {
        if (player[p].onboard)
          continue;
        for (int t = 0; t < player[p].turn; ++t) {
          for (int j = 0; j < boardHeight; ++j) {
            for (int i = 0; i < boardWidth; ++i) {
              if (board[j][i] != 2)
                continue;
              bool legal = false;
              for (int c = 0; c < player[p].count; ++c) {
                int x = i + player[p].tail[c].x;
                int y = j + player[p].tail[c].y;
                if (!canDrop(x, y)) {
                  legal = false;
                  break;
                } else if (visited[y][x]) {
                  legal = true;
                }
              }
              if (legal) {
                Move* m = new Move();
                m->dir = t;
                m->piece = p;
                m->x = i;
                m->y = j;
                moves.push_back(*m);
                delete m;
              }
            }
          }
          player[p].turn90();
        }
        if (p >= 3 && p <= 5) {
          player[p].turn90();
          player[p].turn90();
        } else if (p == 2) {
          player[p].turn90();
          player[p].turn90();
          player[p].turn90();
        }
      }
    }
  }

  void findActions() {
    moves.clear();
    if (_status == GameStatus::player0Turn) {
      legalMoves(player0, moves);

    } else if (_status == GameStatus::player1Turn) {
      legalMoves(player1, moves);
    }
    // fprintf(stderr, "moves: %d\n", moves.size());
    _legalActions.clear();
    for (auto m : moves) {
      int x = m.x;
      int y = m.y;
      int p = m.piece * 4 + m.dir;

      _legalActions.push_back(
          std::make_shared<ActionForBlockGo>(x, y, p, _legalActions.size()));
    }
  }

  void track(int x, int y) {
    if (!canDrop(x, y))
      return;
    if (territory[y][x] == 4)
      return;
    territory[y][x] = 4;
    // fprintf(stderr, "xy: %d %d\n", x, y);
    track(x + 1, y);
    track(x, y - 1);
    track(x - 1, y);
    track(x, y + 1);
  }

  int edge() {
    int p[2] = {0};
    for (int j = 0; j < boardHeight; ++j) {
      for (int i = 0; i < boardWidth; ++i) {
        if (territory[j][i] == 4) {
          if (i + 1 < boardWidth && board[j][i + 1] < 2)
            p[board[j][i + 1]]++;
          if (j - 1 >= 0 && board[j - 1][i] < 2)
            p[board[j - 1][i]]++;
          if (i - 1 >= 0 && board[j][i - 1] < 2)
            p[board[j][i - 1]]++;
          if (j + 1 < boardHeight && board[j + 1][i] < 2)
            p[board[j + 1][i]]++;
        }
      }
    }
    // fprintf(stderr, "p: %d %d\n", p[0], p[1]);
    if (p[0] && p[1])
      return 2;
    else if (p[0])
      return 0;
    else
      return 1;
  }

  void setTerritory(int a) {
    for (int j = 0; j < boardHeight; ++j)
      for (int i = 0; i < boardWidth; ++i)
        if (territory[j][i] == 4)
          territory[j][i] = a;
  }

  void printTerritory() {
    for (int i = 0; i < 13; ++i) {
      for (int j = 0; j < 13; ++j)
        fprintf(stderr, " %d ", territory[i][j]);
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
  }

  void findTerritory() {
    // fprintf(stderr, "findTerritory\n");
    for (int j = 0; j < boardHeight; ++j)
      for (int i = 0; i < boardWidth; ++i)
        territory[j][i] = 3;

    for (int j = 0; j < boardHeight; ++j) {
      for (int i = 0; i < boardWidth; ++i) {
        if (canDrop(i, j) && territory[j][i] == 3) {
          // fprintf(stderr, "x: %d y: %d\n", i, j);
          track(i, j);
          // printTerritory();
          setTerritory(edge());
        }
      }
    }
  }

  GameStatus findWinner() {
    findTerritory();
    int p[4] = {0};
    for (int j = 0; j < boardHeight; ++j)
      for (int i = 0; i < boardWidth; ++i)
        p[territory[j][i]]++;

    // fprintf(stderr, "score: %d %d %d %d\n", p[0], p[1], p[2], p[3]);

    if (p[0] == p[1])
      return GameStatus::tie;
    else if (p[0] > p[1])
      return GameStatus::player0Win;
    else
      return GameStatus::player1Win;
  }

  int findType(int piece, int dir) {
    switch (piece) {
    case 0:
    case 1:
      return 0;
    case 2:
      return 1;
    case 3:
      return 2 + dir;
    case 4:
      return 4 + dir;
    case 5:
      return 6 + dir;
    case 6:
      return 8 + dir;
    case 7:
      return 12 + dir;
    case 8:
      return 16 + dir;

    default:
      fprintf(stderr, "piece error!!!!!\n");
      return -1;
    }
  }

  virtual void ApplyAction(const _Action& action) override {
    _hash ^= HashTurn;
    // fprintf(stderr, "ApplyAction round %d\n", round);
    int x = action.GetY();
    int y = action.GetZ();
    int piece = action.GetX() >> 2;
    int dir = action.GetX() & 3;
    _hash ^= HashArray[findType(piece, dir)][x][y];
    // fprintf(stderr, "(%d, %d) %d %d\n", x, y, piece, dir);
    // fprintf(stderr, "hash: %llu\n", _hash);
    if (_status == GameStatus::player0Turn) {
      for (int i = 0; i < dir; ++i)
        player0[piece].turn90();
      for (int i = 0; i < player0[piece].count; ++i)
        board[y + player0[piece].tail[i].y][x + player0[piece].tail[i].x] = 0;
      player0[piece].onboard = true;
      _status = GameStatus::player1Turn;

    } else {
      for (int i = 0; i < dir; ++i)
        player1[piece].turn90();
      for (int i = 0; i < player1[piece].count; ++i)
        board[y + player1[piece].tail[i].y][x + player1[piece].tail[i].x] = 1;
      player1[piece].onboard = true;
      _status = GameStatus::player0Turn;
    }

    if (round >= 18) {
      _status = findWinner();
      // printTerritory();
    } else {
      round += 1;
      findFeature();
      findActions();
    }

    // printCurrentBoard();
    // fprintf(stderr, "end applyAction =======\n\n");
    fillFullFeatures();
  }

  virtual std::string stateDescription() const override {
    std::stringstream ss;
    ss << "    0  1  2  3  4  5  6  7  8  9  10 11 12\n";
    for(int i=0; i<boardHeight; ++i) {
      char buff[12];
      sprintf(buff, "%2d ", i);
      ss << buff;
      for(int j=0; j<boardWidth; ++j) {
        switch (board[i][j]) {
          case 0: ss << " O "; break;
          case 1: ss << " X "; break;
          default: ss << " - "; break;
        }
      }
      ss << std::endl;
    }
    ss << std::endl;
    return ss.str();
  }

  struct Actions {
    int x, y, z, i;
  };

  static bool compareAction(Actions a, Actions b) {
    if (a.z != b.z) return a.z < b.z; 
    if (a.x != b.x) return a.x < b.x;
    return a.y < b.y;
  };

  virtual std::string actionsDescription() override {
    std::stringstream ss;
    Actions a[_legalActions.size()];
    int i=0;
    for(auto action : _legalActions) {
      a[i].x = (*action).GetY();
      a[i].y = (*action).GetZ();
      a[i].z = (*action).GetX();
      a[i].i = i;
      i++;
    }
    std::sort(a, a+_legalActions.size(), compareAction);
    for(auto action : a) {
      char buff[54];
      sprintf(buff, "%d: (%d %d) ---%d\n", action.z, action.x, action.y, action.i);
      ss << buff;
    }
    ss << "\nInput format: action index e.g. 0\n";
    return ss.str();
  }

  virtual std::string actionDescription(const _Action & action) const {
    std::stringstream ss;
    int z = action.GetX();
    int x = action.GetY();
    int y = action.GetZ();
    ss << z << ": " << '(' << x << ' ' << y << ")\n";
    return ss.str();
  }

  virtual void DoGoodAction() override {
//    std::cerr << "DoGoodAction" << std::endl;
    return DoRandomAction();
  }
};
