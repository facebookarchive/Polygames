/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

const int White = 0;
const int Black = 1;
const int Empty = 2;

const int BTDx = 8;
const int BTDy = 8;

const int BTMaxLegalBTMoves = 3 * BTDx * 2;
const int BTMaxPlayoutLength = 1000;

// const int SizeTable = 1048575;  // une puissance de 2 moins 1

extern unsigned long long BTHashArray[2][BTDx][BTDy];
extern unsigned long long BTHashTurn;

// int level = 1;

// unsigned long long nbPlay = 0;

// timeval stop, start;
// unsigned long long previousTime = 0;

// extern bool BTinitHashCalled = false;
extern std::once_flag BTinitHashCalled;

extern void BTinitHash();

class BTPlayer {
 public:
  int player;

  bool operator==(BTPlayer p) {
    return (p.player == player);
  }
};

// const int MaxBTMoveNumber = 2 * 2 * (3 * BTDx * BTDy) + 1;
const int MaxBTMoveNumber = 80 * 2 * 2 * (3 * BTDx * BTDy) + 1;

// bool BTuseCode = true;

class BTMove {
 public:
  int x, y, x1, y1, color;
  int code;
  int codePrevious;

  BTMove() {
    x = -1;
    y = -1;
    x1 = -1;
    y1 = -1;
    color = -1;
    code = -1;
    codePrevious = -1;
  }

  int numberPrevious() {
    int c = 0;
    c = code;
    if (color == White)
      return c + 3 * (x + BTDx * y) + x1 - x + 1;
    else
      return c + 3 * BTDx * BTDy + 3 * (x + BTDx * y) + x1 - x + 1;
  }

  int number() {
    int c = 0;
    c = code + codePrevious;
    if (color == White)
      return c + 3 * (x + BTDx * y) + x1 - x + 1;
    else
      return c + 3 * BTDx * BTDy + 3 * (x + BTDx * y) + x1 - x + 1;
  }
};

class BTBoard {
 public:
  int board[BTDx][BTDy];
  unsigned long long hash;
  BTMove rollout[BTMaxPlayoutLength];
  int length, turn;
  int orderBTMove[BTMaxLegalBTMoves];

  void init() {
    for (int i = 0; i < BTDx; i++)
      for (int j = 0; j < BTDy; j++)
        board[i][j] = Empty;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < BTDx; j++)
        board[j][i] = Black;
    for (int i = BTDy - 2; i < BTDy; i++)
      for (int j = 0; j < BTDx; j++)
        board[j][i] = White;
    hash = 0;
    length = 0;
    turn = White;
    /*if (BTinitHashCalled == false) {
      BTinitHash();
      BTinitHashCalled = true;
    }*/
    std::call_once(BTinitHashCalled, BTinitHash);
  }

  int countPieces(int color) const {
    int r = 0;
    for (int i = 0; i < BTDx; i++) {
      for (int j = 0; j < BTDy; j++) {
        if (board[i][j] == color) {
          ++r;
        }
      }
    }
    return r;
  }

  bool won(int color) {
    if (color == White) {
      for (int j = 0; j < BTDx; j++)
        if (board[j][0] == White)
          return true;
      BTMove listeCoups[BTMaxLegalBTMoves];
      int nb = legalBTMoves(Black, listeCoups);
      if (nb == 0)
        return true;
    } else {
      for (int j = 0; j < BTDx; j++)
        if (board[j][BTDy - 1] == Black)
          return true;
      BTMove listeCoups[BTMaxLegalBTMoves];
      int nb = legalBTMoves(White, listeCoups);
      if (nb == 0)
        return true;
    }
    return false;
  }

  bool terminal() {
    // return won (Black) || won (White);
    for (int j = 0; j < BTDx; j++)
      if (board[j][0] == White)
        return true;
    for (int j = 0; j < BTDx; j++)
      if (board[j][BTDy - 1] == Black)
        return true;
    BTMove listeCoups[BTMaxLegalBTMoves];
    int nb = legalBTMoves(turn, listeCoups);
    if (nb == 0)
      return true;
    return false;
  }

  int score() {
    if (won(White))
      return 1;
    return 0;
  }

  float evaluation(int color) {
    if (won(color))
      return 1000000.0;
    if (won(opponent(color)))
      return -1000000.0;
    BTMove moves[BTMaxLegalBTMoves];
    int nb = legalBTMoves(turn, moves);
    if (nb == 0) {
      if (color == turn)
        return -1000000.0;
      else
        return 1000000.0;
    }
    int nbOpponent = legalBTMoves(opponent(turn), moves);
    if (color == turn)
      return (float)(nb - nbOpponent);
    return (float)(nbOpponent - nb);
  }

  int opponent(int joueur) const {
    if (joueur == White)
      return Black;
    return White;
  }

  bool losingBTMove(BTMove m) {
    if (m.color == Black) {
      for (int j = 0; j < BTDx; j++)
        if (board[j][BTDy - 2] == Black)
          if (m.y1 != BTDy - 1)
            return true;
      for (int j = 0; j < BTDx; j++)
        if (board[j][1] == White)
          if ((m.y1 != 1) || (m.x1 != j))
            return true;
    }
    if (m.color == White) {
      for (int j = 0; j < BTDx; j++)
        if (board[j][1] == White)
          if (m.y1 != 0)
            return true;
      for (int j = 0; j < BTDx; j++)
        if (board[j][BTDy - 2] == Black)
          if ((m.y1 != BTDy - 2) || (m.x1 != j))
            return true;
    }
    return false;
  }

  int order(BTMove m) {
    if (m.color == Black) {
      if (m.y1 == BTDy - 1)
        return 0;
    }
    if (m.color == White) {
      if (m.y1 == 0)
        return 0;
    }
    if (board[m.x1][m.y1] == opponent(m.color))
      return 1;
    return 2;
  }

  bool legalBTMove(BTMove m) {
    if (board[m.x][m.y] != m.color)
      return false;
    if (board[m.x1][m.y1] == m.color)
      return false;
    if (m.color == White)
      if ((m.y1 == m.y - 1) && (m.x == m.x1))
        if (board[m.x1][m.y1] == Black)
          return false;
    if (m.color == Black)
      if ((m.y1 == m.y + 1) && (m.x == m.x1))
        if (board[m.x1][m.y1] == White)
          return false;
    return true;
  }

  void play(BTMove m) {
    board[m.x][m.y] = Empty;
    hash ^= BTHashArray[m.color][m.x][m.y];
    if (board[m.x1][m.y1] != Empty)
      hash ^= BTHashArray[board[m.x1][m.y1]][m.x1][m.y1];
    board[m.x1][m.y1] = m.color;
    hash ^= BTHashArray[m.color][m.x1][m.y1];
    hash ^= BTHashTurn;
    if (length < BTMaxPlayoutLength) {
      rollout[length] = m;
      length++;
    } else
      fprintf(stderr, "Pb play,");
    turn = opponent(turn);
    // nbPlay++;
  }

  void print(FILE* fp) {
    for (int i = 0; i < BTDy; i++) {
      for (int j = 0; j < BTDx; j++)
        if (board[j][i] == Empty)
          fprintf(fp, " +");
        else if (board[j][i] == Black)
          fprintf(fp, " @");
        else
          fprintf(fp, " O");
      fprintf(fp, " \n");
    }
    fprintf(fp, " \n");
  }

  int legalBTMoves(int color, BTMove moves[BTMaxLegalBTMoves]) {
    int nb = 0;
    for (int i = 0; i < BTDx; i++)
      for (int j = 0; j < BTDy; j++)
        if (board[i][j] == color) {
          BTMove m;
          m.x = i;
          m.y = j;
          m.color = color;
          if (color == White) {
            if ((j - 1 >= 0) && (i + 1 < BTDx)) {
              m.x1 = i + 1;
              m.y1 = j - 1;
              if (board[m.x1][m.y1] == Empty)
                m.code = 0;
              else
                m.code = 6 * BTDx * BTDy;
              if (legalBTMove(m)) {
                moves[nb] = m;
                nb++;
              }
            }
            if ((j - 1 >= 0) && (i - 1 >= 0)) {
              m.x1 = i - 1;
              m.y1 = j - 1;
              if (board[m.x1][m.y1] == Empty)
                m.code = 0;
              else
                m.code = 6 * BTDx * BTDy;
              if (legalBTMove(m)) {
                moves[nb] = m;
                nb++;
              }
            }
            if ((j - 1 >= 0)) {
              m.x1 = i;
              m.y1 = j - 1;
              if (board[m.x1][m.y1] == Empty)
                m.code = 0;
              else
                m.code = 6 * BTDx * BTDy;
              if (legalBTMove(m)) {
                moves[nb] = m;
                nb++;
              }
            }
          } else {
            if ((j + 1 < BTDy) && (i + 1 < BTDx)) {
              m.x1 = i + 1;
              m.y1 = j + 1;
              if (board[m.x1][m.y1] == Empty)
                m.code = 0;
              else
                m.code = 6 * BTDx * BTDy;
              if (legalBTMove(m)) {
                moves[nb] = m;
                nb++;
              }
            }
            if ((j + 1 < BTDy) && (i - 1 >= 0)) {
              m.x1 = i - 1;
              m.y1 = j + 1;
              if (board[m.x1][m.y1] == Empty)
                m.code = 0;
              else
                m.code = 6 * BTDx * BTDy;
              if (legalBTMove(m)) {
                moves[nb] = m;
                nb++;
              }
            }
            if ((j + 1 < BTDy)) {
              m.x1 = i;
              m.y1 = j + 1;
              if (board[m.x1][m.y1] == Empty)
                m.code = 0;
              else
                m.code = 6 * BTDx * BTDy;
              if (legalBTMove(m)) {
                moves[nb] = m;
                nb++;
              }
            }
          }
        }
    for (int i = 0; i < nb; i++)
      orderBTMove[i] = order(moves[i]);
    for (int i = 0; i < nb; i++) {
      int imin = i;
      int o = orderBTMove[i];
      for (int j = i + 1; j < nb; j++) {
        int o1 = orderBTMove[j];
        if (o1 < o) {
          imin = j;
          o = o1;
        }
      }
      BTMove m = moves[i];
      moves[i] = moves[imin];
      moves[imin] = m;
      o = orderBTMove[i];
      orderBTMove[i] = orderBTMove[imin];
      orderBTMove[imin] = o;
    }
    /*
       for (int i = 0; i < nb; i++)
       if (order (moves [i]) == 1) {
       BTMove m = moves [0];
       moves [0] = moves [i];
       moves [i] = m;
       }
       for (int i = 0; i < nb; i++)
       if (order (moves [i]) == 0) {
       BTMove m = moves [0];
       moves [0] = moves [i];
       moves [i] = m;
       }
     */
    int codePrevious = 0;
    if (length > 0)
      // codePrevious = rollout [length].numberPrevious () + 12 * BTDx * BTDy;
      codePrevious = (rollout[length].x1 + BTDx * rollout[length].y1) * 2 * 2 *
                     (3 * BTDx * BTDy);
    for (int i = 0; i < nb; i++) {
      codePrevious = 0;
      if (color == White) {
        if (moves[i].y1 > 0) {
          if (moves[i].x1 == 0)
            codePrevious += 4;
          else
            codePrevious += board[moves[i].x1 - 1][moves[i].y1 - 1];
          codePrevious += 4 * board[moves[i].x1][moves[i].y1 - 1];
          if (moves[i].x1 == BTDx - 1)
            codePrevious += 16 * 4;
          else
            codePrevious += 16 * board[moves[i].x1 + 1][moves[i].y1 - 1];
        }
      } else {
        if (moves[i].y1 < BTDy - 1) {
          if (moves[i].x1 == 0)
            codePrevious += 4;
          else
            codePrevious += board[moves[i].x1 - 1][moves[i].y1 + 1];
          codePrevious += 4 * board[moves[i].x1][moves[i].y1 + 1];
          if (moves[i].x1 == BTDx - 1)
            codePrevious += 16 * 4;
          else
            codePrevious += 16 * board[moves[i].x1 + 1][moves[i].y1 + 1];
        }
      }
      moves[i].codePrevious = codePrevious * 12 * BTDx * BTDy;
    }
    return nb;
  }

  int playout(int joueur) {
    BTMove listeCoups[BTMaxLegalBTMoves];
    while (true) {
      if (terminal())
        return (score() > 0);
      int nb = legalBTMoves(joueur, listeCoups);
      int n = rand() % nb;
      play(listeCoups[n]);
      if (length >= BTMaxPlayoutLength - 20) {
        return 0;
      }
      joueur = opponent(joueur);
    }
  }

  float discountedPlayout(int joueur, int maxLength = BTMaxPlayoutLength - 20) {
    BTMove listeCoups[BTMaxLegalBTMoves];
    while (true) {
      if (terminal()) {
        if (score() > 0)
          return 1.0 / (length + 1);
        else
          return -1.0 / (length + 1);
      }
      int nb = legalBTMoves(joueur, listeCoups);
      int n = rand() % nb;
      play(listeCoups[n]);
      if (length >= maxLength) {
        return 0;
      }
      joueur = opponent(joueur);
    }
  }
};
