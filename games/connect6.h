/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author 1. CHEN,SHIH-YU leo03164@gmail.com
// Author 2. CHIU,HSIEN-TUNG yumjelly@gmail.com

#include <list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

using namespace std;
namespace Connect6 {


const int C6White = 0;
const int C6Black = 1;
const int C6Empty = 2;

const int C6Dx = 19;
const int C6Dy = 19;

const int C6MaxLegalMoves = C6Dy * C6Dx;
const int C6MaxPlayoutLength = C6Dx * C6Dy;

// TODO: 原本沒有 BEGIN

class C6Player {
 public:
  int player;

  bool operator==(C6Player p) {
    return (p.player == player);
  }
};
// END TODO

// const int MaxMoveNumber = 2 * 2 * (3 * C6Dx * C6Dy) + 1;
// const int MaxMoveNumber = 80 * 2 * 2 * (3 * C6Dx * C6Dy) + 1;
const int MaxMoveNumber = C6Dx * C6Dy;

class C6Move {
 public:
  int x, y, color;
};

class C6Board {

 public:
  int nb;
  char board[C6Dx][C6Dy];
  unsigned long long hash;

  void init() {
    for (int i = 0; i < C6Dx; i++)
      for (int j = 0; j < C6Dy; j++)
        board[i][j] = C6Empty;

    hash = 0;
    // printf("init\n");
  }
  bool won(C6Move m) {
    int Max_connect = 6;
    int current_coun = 1;

    int x = m.x;
    int y = m.y;
    int color = m.color;

    bool opsite = true;

    for (int i = 1; i <= Max_connect; i++) {
      if (x - i >= 0 && board[x - i][y] == color && opsite) {
        current_coun++;
        continue;
      } else {
        i = 7;
        opsite = false;
        for (int j = 1; j <= Max_connect; j++) {
          if (x + j < C6Dx && board[x + j][y] == color) {
            current_coun++;
            continue;
          } else
            break;
        }
      }
    }
    if (current_coun >= Max_connect)
      return true;
    current_coun = 1;
    opsite = true;
    //------------------------------------------------------------------------------
    for (int i = 1; i <= Max_connect; i++) {
      if (y - i >= 0 && board[x][y - i] == color && opsite) {
        current_coun++;
        continue;
      } else {
        i = 7;
        opsite = false;
        for (int j = 1; j <= Max_connect; j++) {
          if (y + j < C6Dx && board[x][y + j] == color) {
            current_coun++;
            continue;
          } else
            break;
        }
      }
    }
    if (current_coun >= Max_connect)
      return true;
    current_coun = 1;
    opsite = true;
    //------------------------------------------------------------------------------
    for (int i = 1; i <= Max_connect; i++) {
      if ((y - i >= 0) && (x + i < C6Dx) && board[x + i][y - i] == color &&
          opsite) {
        current_coun++;
        continue;
      } else {
        i = 7;
        opsite = false;
        for (int j = 1; j <= Max_connect; j++) {
          if ((x - j >= 0) && (y + j < C6Dx) && board[x - j][y + j] == color) {
            current_coun++;
            continue;
          } else
            break;
        }
      }
    }
    if (current_coun >= Max_connect)
      return true;
    current_coun = 1;
    opsite = true;
    //------------------------------------------------------------------------------
    for (int i = 1; i <= Max_connect; i++) {
      if ((y - i >= 0) && (x - i >= 0) && board[x - i][y - i] == color &&
          opsite) {
        current_coun++;
        continue;
      } else {
        i = 7;
        opsite = false;
        for (int j = 1; j <= Max_connect; j++) {
          if ((y + j < C6Dx) && (x + j < C6Dy) &&
              board[x + j][y + j] == color) {
            current_coun++;
            continue;
          } else
            break;
        }
      }
    }
    if (current_coun >= Max_connect)
      return true;
    else
      return false;
  }

  int opponent(int joueur) {
    // printf("opponent\n");
    if (joueur == C6White)
      return C6Black;
    return C6White;
  }

  //找合法步
  bool legalMove(C6Move m) {
    // printf("legalMove\n");
    if (board[m.x][m.y] != C6Empty)
      return false;
    return true;
  }

  void play(C6Move m) {

    board[m.x][m.y] = m.color;
  }

  int legalMoves(C6Move moves[C6MaxLegalMoves]) {
    // printf("legalMoves\n");
    nb = 0;

    for (int i = 0; i < C6Dx; i++) {
      for (int j = 0; j < C6Dy; j++) {
        if (board[i][j] == C6Empty) {
          moves[nb].x = i;
          moves[nb].y = j;
          nb++;
        }
      }
    }
    return nb;
  }

};
}
