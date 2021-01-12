/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../core/state.h"

class StateForEinstein : public core::State {
 public:
  class Piece {
   public:
    int color;
    int type;  // 1~6
    bool onboard;
    int x, y;

    Piece() {
      color = 2;
      type = 0;
      x = y = -1;
    }

    Piece(int c, int t) {
      color = c;
      type = t;
      onboard = false;
      x = y = -1;
    }

    void setPiece(int c, int t, bool isonboard) {
      color = c;
      type = t;
      onboard = isonboard;
    }

    void setPosition(int X, int Y) {
      x = X;
      y = Y;
    }
  };

  class Move {
   public:
    int x, y;
    int type;
    Move() {
      x = y = -1;
      type = 0;
    }
    Move(int X, int Y, int p) {
      x = X;
      y = Y;
      type = p;
    }
  };

 public:
  const static int boardWidth = 5;
  const static int boardHeight = 5;
  Move p0_drop;
  Piece player[2][6];
  Piece board[5][5];
  int dice;
  int round;
  std::vector<Move> moves;
  //  unsigned long long HashArray[2][6][5][5];
  unsigned long long HashTurn;

  StateForEinstein(int seed)
      : State(seed) {
    _stochasticReset = true;
  }
  /*~StateForEinstein() {
  }*/
  virtual void Initialize() override {
    _moves.clear();
    // _hash = 2166136261u;
    _hash = 0;
    _status = GameStatus::player0Turn;
    _featSize[0] = 14;  // 2 players * 6 pieces + 1 dice + 1 turn
    _featSize[1] = boardHeight;
    _featSize[2] = boardWidth;
    _actionSize[0] = 6;  // 6 pieces
    _actionSize[1] = boardHeight;
    _actionSize[2] = boardWidth;
    _features.clear();
    _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
    _stochastic = true;
    // setFeatures(false, false, false, 0, 0, false);

    gameInit();
    // printCurrentBoard();

    findFeature();
    findActions();
  }

  void initHash() {
    /*    for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 6; ++j)
            for (int x = 0; x < boardWidth; ++x)
              for (int y = 0; y < boardHeight; ++y) {
                HashArray[i][j][x][y] = 0;
                for (int k = 0; k < 64; ++k)
                  if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
                    HashArray[i][j][x][y] |= (1ULL << k);
              }
        HashTurn = 0;
        for (int k = 0; k < 64; ++k)
          if ((_rng() / (RAND_MAX + 1.0)) > 0.5)
            HashTurn |= (1ULL << k);*/
  }

  void gameInit() {
    for (int j = 0; j < boardHeight; ++j)
      for (int i = 0; i < boardWidth; ++i)
        board[j][i].setPiece(2, 0, false);  // 2 = Empty

    for (int i = 0; i < 6; ++i) {
      player[0][i].setPiece(0, i + 1, false);
      player[1][i].setPiece(1, i + 1, false);
    }
    if (forcedDice > 0) {
      dice = forcedDice - 1;
    } else {
      dice = _rng() % 6;
    }
    _hash = dice;
    round = 1;
  }

  virtual void setStateFromStr(const std::string& str) override {
    // example: ABCDEF0000000000000abcdef
    /* -> x1 x2 x3 x4 x5
          x6 0 0 0 0
          0 0 0 0 0
          0 0 0 0 o1
          o2 o3 o4 o5 o6
    */
    assert(str.length() == 26);
    char turn = str[25];
    _status = turn == '0' ? GameStatus::player0Turn : GameStatus::player1Turn;
    int color = -1;
    for (int i = 0; i < 25; i++) {
      int t = -1;
      int y = i / 5;
      int x = i % 5;
      char c = str[i];
      if (c >= 'A' && c <= 'F') {
        t = int(c) - int('A');
        color = 1;
      }
      if (c >= 'a' && c <= 'f') {
        t = int(c) - int('a');
        color = 0;
      }
      if (t == -1) {
        continue;
      }
      player[color][t].onboard = true;
      player[color][t].setPosition(x, y);
      board[y][x] = player[color][t];
    }
    if (forcedDice > 0) {
      dice = forcedDice - 1;
    } else {
      dice = _rng() % 6;
    }
    _hash = dice;
    round = 13;
    findActions();
  }

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<StateForEinstein>(*this);
  }

  virtual std::string stateDescription() const override {
    std::string str;
    str += "  A |B |C |D |E \n";
    for (int j = 0; j < boardHeight; j++) {
      str += to_string(j + 1) + ' ';
      for (int i = 0; i < boardWidth; i++) {
        if (i > 0)
          str += '|';
        if (board[j][i].color == 0) {
          str += 'x';
          str += static_cast<char>(board[j][i].type + '0');
        } else if (board[j][i].color == 1) {
          str += 'o';
          str += static_cast<char>(board[j][i].type + '0');
        } else
          str += "  ";
      }
      str += '\n';
    }

    return str;
  }

  virtual std::string actionsDescription() const override {
    std::stringstream ss;
    char c, p, x1, y1;
    for (int i = 0; i < (int)_legalActions.size(); i++) {
      const _Action& action = _legalActions[i];
      c = (_status == GameStatus::player0Turn) ? 'x' : 'o';
      p = static_cast<char>(action.GetX() + ((round <= 12) ? '0' : '1'));
      x1 = static_cast<char>(action.GetY() + 'A');
      y1 = static_cast<char>(action.GetZ() + '1');
      ss << "Action " << i << ": " << c << p << " to " << x1 << y1 << std::endl;
    }
    ss << "\nInput format : action index e.g. 0\n";

    return ss.str();
  }

  virtual std::string actionDescription(const _Action& action) const override {
    std::stringstream ss;
    char c, p, x1, y1;
    c = (_status == GameStatus::player0Turn) ? 'o' : 'x';
    p = static_cast<char>(action.GetX() + ((round <= 12) ? '0' : '1'));
    x1 = static_cast<char>(action.GetY() + 'A');
    y1 = static_cast<char>(action.GetZ() + '1');
    ss << c << p << " to " << x1 << y1;

    return ss.str();
  }

  void findFeature() {
    std::fill(_features.begin(), _features.end(), 0);
    if (_status == GameStatus::player1Turn) {
      // 0 ~ 150
      for (int i = 0; i < 6; ++i) {
        Piece p = player[0][i];
        if (p.onboard)
          _features[25 * p.type + p.y * 5 + p.x] = 1;
      }
      // 150 ~ 300
      for (int i = 0; i < 6; ++i) {
        Piece p = player[1][i];
        if (p.onboard)
          _features[25 * p.type + p.y * 5 + p.x + 150] = 1;
      }
    } else {
      // 0 ~ 150
      for (int i = 0; i < 6; ++i) {
        Piece p = player[0][i];
        if (p.onboard)
          _features[25 * p.type + (4 - p.y) * 5 + (4 - p.x)] = 1;
      }
      // 150 ~ 300
      for (int i = 0; i < 6; ++i) {
        Piece p = player[1][i];
        if (p.onboard)
          _features[25 * p.type + (4 - p.y) * 5 + (4 - p.x) + 150] = 1;
      }
    }
    // 300 ~ 325
    if (dice == 5)
      std::fill(_features.begin() + 300, _features.begin() + 325, 1);
    else
      std::fill(_features.begin() + 300 + dice * 5,
                _features.begin() + 300 + dice * 5 + 5, 1);

    // 325 ~
    std::fill(_features.begin() + 325, _features.end(), (float)_status);
    fillFullFeatures();
  }

  void legalMoves(int color, std::vector<Move>& moves) {
    // fprintf(stderr, "dice: %d\n", dice+1);
    if (round <= 12) {
      if (color == 0) {  // player0
        int p = round >> 1;

        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 3 - j; ++i) {
            if (board[j][i].type == 0) {  // if empty
              Move m(i, j, p + 1);        // p+1 = piece type
              moves.push_back(m);
            }
          }
        }

      } else {
        Move m(-p0_drop.y + 4, -p0_drop.x + 4, p0_drop.type);
        moves.push_back(m);
      }
    } else {
      if (player[color][dice].onboard) {
        if (color == 0) {
          short dx[] = {1, 1, 0};
          short dy[] = {0, 1, 1};
          for (int i = 0; i < 3; ++i) {
            int x = dx[i] + player[color][dice].x;
            int y = dy[i] + player[color][dice].y;
            if (x < 5 && y < 5) {
              Move m(x, y, player[color][dice].type);
              moves.push_back(m);
            }
          }
        } else {
          short dx[] = {0, -1, -1};
          short dy[] = {-1, -1, 0};
          for (int i = 0; i < 3; ++i) {
            int x = dx[i] + player[color][dice].x;
            int y = dy[i] + player[color][dice].y;
            if (x >= 0 && y >= 0) {
              Move m(x, y, player[color][dice].type);
              moves.push_back(m);
            }
          }
        }
      } else {
        bool find = false;
        for (int i = 1; i < 6; ++i) {
          for (int j = 0, k = 1; j < 2; ++j, k *= -1) {
            int closest = dice + i * k;
            if (closest < 6 && closest >= 0 && player[color][closest].onboard) {
              find = true;
              if (color == 0) {
                short dx[] = {1, 1, 0};
                short dy[] = {0, 1, 1};
                for (int ii = 0; ii < 3; ++ii) {
                  int x = dx[ii] + player[color][closest].x;
                  int y = dy[ii] + player[color][closest].y;
                  if (x < 5 && y < 5) {
                    Move m(x, y, player[color][closest].type);
                    moves.push_back(m);
                  }
                }
              } else {
                short dx[] = {0, -1, -1};
                short dy[] = {-1, -1, 0};
                for (int ii = 0; ii < 3; ++ii) {
                  int x = dx[ii] + player[color][closest].x;
                  int y = dy[ii] + player[color][closest].y;
                  if (x >= 0 && y >= 0) {
                    Move m(x, y, player[color][closest].type);
                    moves.push_back(m);
                  }
                }
              }
            }
          }
          if (find)
            break;
        }
      }
    }
  }

  void findActions() {
    moves.clear();
    if (_status == GameStatus::player0Turn) {
      legalMoves(0, moves);

    } else if (_status == GameStatus::player1Turn) {
      legalMoves(1, moves);
    }
    // fprintf(stderr, "round %d moves: %d\n", round, moves.size());

    clearActions();
    for (auto m : moves) {
      // fprintf(stderr, "%d: (%d, %d)\n", m.type, m.x, m.y);
      addAction(m.type - 1, m.x, m.y);
    }
  }

  virtual void ApplyAction(const _Action& action) override {
    // fprintf(stderr, "Apply action round: %d\n", round);
    int color;
    if (_status == GameStatus::player0Turn) {
      color = 0;
      _status = GameStatus::player1Turn;
    } else {
      color = 1;
      _status = GameStatus::player0Turn;
    }

    int t = action.GetX();
    assert(t < 6);
    int x = action.GetY();
    int y = action.GetZ();
    assert(y < 6);
    assert(x < 6);
    // fprintf(stderr, "get: %d %d %d\n", t, x, y);
    if (round <= 12) {
      p0_drop.type = t + 1;
      p0_drop.x = x;
      p0_drop.y = y;
      player[color][t].onboard = true;
    } else {
      board[player[color][t].y][player[color][t].x] = Piece();
    }

    if (board[y][x].type != 0) {  // eat
      player[board[y][x].color][board[y][x].type - 1].onboard = false;
      player[board[y][x].color][board[y][x].type - 1].setPosition(-1, -1);
      //      _hash ^= HashArray[board[y][x].color][board[y][x].type - 1][x][y];
    }
    player[color][t].onboard = true;
    player[color][t].setPosition(x, y);
    board[y][x] = player[color][t];
    //    _hash ^= HashArray[color][t][x][y];
    //    _hash ^= HashTurn;
    // fprintf(stderr, "(%d %d) t:%d c:%d\n", x, y, board[y][x].type,
    // board[y][x].color);

    if (color == 0 && x == 4 && y == 4)
      _status = GameStatus::player0Win;
    else if (color == 1 && x == 0 && y == 0)
      _status = GameStatus::player1Win;
    else {
      round += 1;
      if (forcedDice > 0) {
        assert(forcedDice > 0);
        assert(forcedDice < 7);
        dice = forcedDice - 1;
        forcedDice = -1;
      } else {
        dice = _rng() % 6;
      }
      _hash = dice;
      findActions();
    }
    findFeature();
    if (_legalActions.size() <= 0)
      _status = (GameStatus)(3 + color);
    // printCurrentBoard();
    // fprintf(stderr, "end Apply Action\n\n");
  }

  virtual void DoGoodAction() override {
    std::cerr << "DoGoodAction" << std::endl;
    DoRandomAction();
  }
};
