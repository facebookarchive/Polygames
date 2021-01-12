/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: Maria Elsa
// - Github: https://github.com/melsaa
// - Email:  m_elsa@ymail.com
// Facilitator: 邱顯棟 (Xiǎn-Dòng Qiū)
// - Github: https://github.com/YumJelly
// - Email:  yumjelly@gmail.com

#pragma once
#include <list>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

using namespace std;

const int SuraWhite = 0;
const int SuraBlack = 1;
const int SuraEmpty = 2;

const int SKDx = 6;
const int SKDy = 6;

const int SuraMaxPlayoutLength = 1000;

class SKHash {
 public:
  unsigned long long HashArray[2][SKDx][SKDy];
  unsigned long long HashTurn;

  bool InitHashCalled = false;

  void InitHash() {
    for (int player = 0; player < 2; player++)
      for (int i = 0; i < SKDx; i++)
        for (int j = 0; j < SKDy; j++) {
          HashArray[player][i][j] = 0;
          for (int k = 0; k < 36; k++)
            if ((rand() / (RAND_MAX + 1.0)) > 0.5)
              HashArray[player][i][j] |= (1ULL << k);
        }
    HashTurn = 0;
    for (int k = 0; k < 36; k++)
      if ((rand() / (RAND_MAX + 1.0)) > 0.5)
        HashTurn |= (1ULL << k);
  }
};

class SKPlayer {
 public:
  int player;

  bool operator==(SKPlayer p) {
    return (p.player == player);
  }
};

class SKMove {
 public:
  int x, y, x1, y1, color;
  bool operator==(const SKMove& m) {
    return (x == m.x && y == m.y && x1 == m.x1 && y1 == m.y1 &&
            color == m.color);
  }
  bool operator!=(const SKMove& m) {
    return !(x == m.x && y == m.y && x1 == m.x1 && y1 == m.y1 &&
             color == m.color);
  }
};

class SKBoard {
 public:
  int board[SKDx][SKDy];
  unsigned long long hash;
  SKMove rollout[SuraMaxPlayoutLength];
  int length, turn, nbPlay, repetition;
  bool isCapture, draw;
  vector<unsigned long long> history_move;
  SKHash Sura;

  void init() {
    for (int i = 0; i < SKDx; i++)
      for (int j = 0; j < SKDy; j++)
        board[i][j] = SuraEmpty;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < SKDx; j++)
        board[j][i] = SuraWhite;
    for (int i = SKDy - 2; i < SKDy; i++)
      for (int j = 0; j < SKDx; j++)
        board[j][i] = SuraBlack;
    hash = 0;
    length = 0;
    turn = SuraWhite;
    nbPlay = 0;
    repetition = 0;
    isCapture = false;
    draw = false;
    history_move.clear();
    history_move.push_back(hash);
    if (Sura.InitHashCalled == false) {
      Sura.InitHash();
      Sura.InitHashCalled = true;
    }
  }

  void print_board(FILE* fp) {
    fprintf(fp, "====================\n");
    for (int i = SKDy - 1; i >= 0; i--) {
      for (int j = 0; j < SKDx; j++)
        if (board[j][i] == SuraBlack)
          fprintf(fp, "x ");
        else if (board[j][i] == SuraWhite)
          fprintf(fp, "o ");
        else if (board[j][i] == SuraEmpty)
          fprintf(fp, "- ");
      fprintf(fp, "\n");
    }
    fprintf(fp, "====================\n");
  }

  bool won(int color) {
    if (color == SuraWhite) {
      for (int i = 0; i < SKDy; i++)
        for (int j = 0; j < SKDx; j++)
          if (board[j][i] == SuraBlack)
            return false;
      vector<SKMove> moves;
      int nb = legalMoves(SuraBlack, moves);
      if (nb == 0)
        return true;
    } else if (color == SuraBlack) {
      for (int i = 0; i < SKDy; i++)
        for (int j = 0; j < SKDx; j++)
          if (board[j][i] == SuraWhite)
            return false;
      vector<SKMove> moves;
      int nb = legalMoves(SuraWhite, moves);
      if (nb == 0)
        return true;
    }
    return false;
  }

  bool is_draw() {
    if (repetition == 3)
      draw = true;
    if (draw == true)
      return true;
    return false;
  }

  int opponent(int color) {
    if (color == SuraWhite)
      return SuraBlack;
    return SuraWhite;
  }

  bool legalMove(SKMove m, bool capture) {
    if (m.x1 < 0 || m.y1 < 0 || m.x1 >= SKDx || m.y1 >= SKDy)
      return false;
    if (board[m.x][m.y] != m.color)
      return false;
    if (board[m.x1][m.y1] == m.color)
      return false;
    if (capture == false && board[m.x1][m.y1] == opponent(m.color))
      return false;
    return true;
  }

  void play(SKMove m) {
    board[m.x][m.y] = SuraEmpty;
    hash ^= Sura.HashArray[m.color][m.x][m.y];
    if (board[m.x1][m.y1] != SuraEmpty) {
      hash ^= Sura.HashArray[board[m.x1][m.y1]][m.x1][m.y1];
      isCapture = true;
    }
    board[m.x1][m.y1] = m.color;
    hash ^= Sura.HashArray[m.color][m.x1][m.y1];
    hash ^= Sura.HashTurn;
    if (length < SuraMaxPlayoutLength) {
      rollout[length] = m;
      length++;
    }
    check_repetition();
    turn = opponent(turn);
    nbPlay++;
    if (nbPlay == 50 && isCapture == false) {
      draw = true;
    } else if (isCapture == true) {
      nbPlay = 0;
      isCapture = false;
    }
  }

  void check_repetition() {
    history_move.push_back(hash);
    int cur_index = history_move.size() - 1;
    if (cur_index >= 8 &&
        history_move[cur_index] == history_move[cur_index - 4] &&
        history_move[cur_index] == history_move[cur_index - 8]) {
      // std::cout << "Three fold Repetitions\n";
      repetition = 3;
    } else if (cur_index >= 4 &&
               history_move[cur_index] == history_move[cur_index - 4]) {
      // std::cout << "Two fold Repetitions\n";
      repetition = 2;
    }
  }

  int legalMoves(int color, vector<SKMove>& moves) {
    int nb = 0;
    bool capture = false;
    int dir[8][2] = {
        {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
    for (int i = 0; i < SKDx; i++)
      for (int j = 0; j < SKDy; j++)
        if (board[i][j] == color) {
          SKMove m;
          m.x = i;
          m.y = j;
          m.color = color;
          for (int k = 0; k < 8; k++) {
            int mx = dir[k][0];
            int my = dir[k][1];
            m.x1 = i + mx;
            m.y1 = j + my;
            if (legalMove(m, capture)) {
              moves.push_back(m);
              nb++;
            }
          }
        }
    return nb;
  }

  int legalCaptures(int nb, int color, vector<SKMove>& moves) {
    int nc = nb;
    bool capture = true;
    int dir[4][2] = {
        {0, 1}, {1, 0}, {0, -1}, {-1, 0}};  // up, right, down, left
    for (int i = 0; i < SKDx; i++)
      for (int j = 0; j < SKDy; j++)
        if (board[i][j] == color &&
            ((i >= 1 && i <= 4) || (j >= 1 && j <= 4))) {
          SKMove m;
          m.x = i;
          m.y = j;
          m.color = color;
          // std::cout << "origin = " << i << "," << j << std::endl;
          for (int k = 0; k < 4; k++) {
            int curK = k;
            int curX = i + dir[curK][0];
            int curY = j + dir[curK][1];
            bool loop = false;
            int step = 0;
            while (curX >= -1 && curY >= -1 && curX <= SKDx && curY <= SKDy) {
              step++;
              // std::cout << "dir = " << curK << " to " << curX << "," << curY
              // << std::endl;
              if (!(curX == i && curY == j) && curX >= 0 && curY >= 0 &&
                  curX < SKDx && curY < SKDy && board[curX][curY] == color) {
                // std::cout << "same color\n";
                break;
              } else if (curX >= 0 && curY >= 0 && curX < SKDx && curY < SKDy &&
                         board[curX][curY] == opponent(color)) {
                if (loop) {  // already go through a loop
                  m.x1 = curX;
                  m.y1 = curY;
                  if (legalMove(m, capture) &&
                      find(moves.begin(), moves.end(), m) == moves.end()) {
                    moves.push_back(m);
                    nc++;
                  }
                }  // else std::cout << "opponent but no loop\n";
                break;
              } else if (curX == i && curY == j && step >= 28)
                break;  // return to origin
              else if ((curX == 0 && curY == 0) ||
                       (curX == 0 && curY == SKDy - 1) ||
                       (curX == SKDx - 1 && curY == 0) ||
                       (curX == SKDx - 1 && curY == SKDy - 1))
                break;  // corner
              else {
                if (curX < 0 || curY < 0 || curX >= SKDx || curY >= SKDy) {
                  // std::cout << "go through a loop\n";
                  // std::cout << "prev " << curX << "," << curY << std::endl;
                  if (curX == 1 || curX == 2) {
                    curK = 1;  // right
                    if (curY > 0)
                      curY -= (curX + 1);
                    else
                      curY += (curX + 1);
                    curX = 0;
                    loop = true;
                  } else if (curX == 3 || curX == 4) {
                    curK = 3;  // left
                    if (curY > 0 && curX == 3)
                      curY -= 3;
                    else if (curY > 0 && curX == 4)
                      curY -= 2;
                    else if (curY < 0 && curX == 3)
                      curY += 3;
                    else if (curY < 0 && curX == 4)
                      curY += 2;
                    curX = 5;
                    loop = true;
                  } else if (curY == 1 || curY == 2) {
                    curK = 0;  // up
                    if (curX > 0)
                      curX -= (curY + 1);
                    else
                      curX += (curY + 1);
                    curY = 0;
                    loop = true;
                  } else if (curY == 3 || curY == 4) {
                    curK = 2;  // down
                    if (curX > 0 && curY == 3)
                      curX -= 3;
                    else if (curX > 0 && curY == 4)
                      curX -= 2;
                    else if (curX < 0 && curY == 3)
                      curX += 3;
                    else if (curX < 0 && curY == 4)
                      curX += 2;
                    curY = 5;
                    loop = true;
                  }
                  // std::cout << "cur = " << curX << "," << curY << " dir = "
                  // << curK << std::endl;
                } else {
                  // std::cout << "continue to next one under same direction\n";
                  curX += dir[curK][0];
                  curY += dir[curK][1];
                }
              }
            }
          }
        }
    return nc;
  }
};
