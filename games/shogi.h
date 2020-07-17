/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <assert.h>
#include <string>
#include <vector>

class Shogi {
 public:
  const static int Dx = 5;
  const static int Dy = 5;
  const static int White = 0;  // player0
  const static int Black = 1;  // player1
  const static int Empty = 2;

  const static int MaxPlayoutLength = 1000;

  enum class PieceType {
    None = 0,
    King,
    Gold,
    Silver,
    Bishop,
    Rook,
    Pawn,
    Gold2,
    Silver2,
    Bishop2,
    Rook2,
    Pawn2
  };

  class Position {
   public:
    int x, y;
    Position() {
      x = y = -1;
    }

    Position(int X, int Y) {
      x = X;
      y = Y;
    }

    bool on_board() const {
      assert((x == -1 && y == -1) ||
             (x >= -1 && y >= -1 && x <= Dx && y <= Dy));
      return (x >= 0 && y >= 0 && x < Dx && y < Dy);
    }

    Position operator+(const Position& p) {
      return Position(x + p.x, y + p.y);
    }

    bool operator==(const Position& p) const {
      return (x == p.x && y == p.y);
    }
    bool operator!=(const Position& p) const {
      return x != p.x || y != p.y;
    }
  };

  class Piece {
   public:
    PieceType type;
    int color;
    bool promoted;
    Position pos;

    Piece() {
      color = Empty;
      type = PieceType::None;
      promoted = false;
    }

    Piece(int c, PieceType t, bool p, Position P = Position(-1, -1)) {
      color = c;
      type = t;
      promoted = p;
      pos = P;
    }

    std::string print() const {
      std::string str;
      switch (type) {
      case PieceType::None:
        str += "  ";
        break;

      case PieceType::King:
        if (color == Black)
          str += "k";
        else
          str += "K";
        break;

      case PieceType::Gold:
      case PieceType::Gold2:
        if (color == Black)
          str += "g";
        else
          str += "G";
        break;

      case PieceType::Silver:
      case PieceType::Silver2:
        if (promoted)
          str += "+";
        if (color == Black)
          str += "s";
        else
          str += "S";
        break;

      case PieceType::Bishop:
      case PieceType::Bishop2:
        if (promoted)
          str += "+";
        if (color == Black)
          str += "b";
        else
          str += "B";
        break;

      case PieceType::Rook:
      case PieceType::Rook2:
        if (promoted)
          str += "+";
        if (color == Black)
          str += "r";
        else
          str += "R";
        break;

      case PieceType::Pawn:
      case PieceType::Pawn2:
        if (promoted)
          str += "+";
        if (color == Black)
          str += "p";
        else
          str += "P";
        break;

      default:
        break;
      }
      return str;
    }
  };

  class Move {
   public:
    Piece piece;
    Position next;
    bool promote;
  };

  Piece board[5][5];
  std::vector<std::vector<Piece>> chess;  // 0 = White, 1 = Black
  // Move rollout[1000];

  void king_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    short dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    short dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int i = 0; i < 8; ++i) {
      m.next = m.piece.pos + Position(dx[i], dy[i]);
      // 在棋盤上、是對方的棋或空
      if (m.next.on_board() && board[m.next.x][m.next.y].color != m.piece.color)
        moves.push_back(m);
    }
  }

  void gold_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    if (m.piece.color == White) {
      short dx[] = {1, 1, 0, -1, -1, 0};
      short dy[] = {0, 1, 1, 1, 0, -1};
      for (int i = 0; i < 6; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color)
          moves.push_back(m);
      }
    } else {
      short dx[] = {1, 0, -1, -1, 0, 1};
      short dy[] = {0, 1, 0, -1, -1, -1};
      for (int i = 0; i < 6; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color)
          moves.push_back(m);
      }
    }
  }

  void silver_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    if (m.piece.promoted) {
      gold_moves(p, moves);
      return;
    }
    if (m.piece.color == White) {
      short dx[] = {1, 0, -1, -1, 1};
      short dy[] = {1, 1, 1, -1, -1};
      for (int i = 0; i < 5; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          moves.push_back(m);
          if (m.next.y == 4) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        }
      }
    } else {
      short dx[] = {1, -1, -1, 0, 1};
      short dy[] = {1, 1, -1, -1, -1};
      for (int i = 0; i < 5; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          moves.push_back(m);
          if (m.next.y == 0) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        }
      }
    }
  }

  void bishop_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    short dx[] = {1, -1, -1, 1};
    short dy[] = {1, 1, -1, -1};
    for (int i = 0; i < 4; ++i) {
      m.next = m.piece.pos + Position(dx[i], dy[i]);
      while (m.next.on_board() &&
             board[m.next.x][m.next.y].color != m.piece.color) {
        moves.push_back(m);
        if (m.piece.color == White) {
          if (m.next.y == 4 && !m.piece.promoted) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        } else {  // Black
          if (m.next.y == 0 && !m.piece.promoted) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        }
        if (board[m.next.x][m.next.y].color != Empty)
          break;
        m.next = m.next + Position(dx[i], dy[i]);
      }
    }
    if (m.piece.promoted) {
      short dx[] = {1, 0, -1, 0};
      short dy[] = {0, 1, 0, -1};
      for (int i = 0; i < 4; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color)
          moves.push_back(m);
      }
    }
  }

  void rook_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    short dx[] = {1, 0, -1, 0};
    short dy[] = {0, 1, 0, -1};
    for (int i = 0; i < 4; ++i) {
      m.next = m.piece.pos + Position(dx[i], dy[i]);
      while (m.next.on_board() &&
             board[m.next.x][m.next.y].color != m.piece.color) {
        moves.push_back(m);
        if (m.piece.color == White) {
          if (m.next.y == 4 && !m.piece.promoted) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        } else {  // Black
          if (m.next.y == 0 && !m.piece.promoted) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        }
        if (board[m.next.x][m.next.y].color != Empty)
          break;
        m.next = m.next + Position(dx[i], dy[i]);
      }
    }
    if (m.piece.promoted) {
      short dx[] = {1, -1, -1, 1};
      short dy[] = {1, 1, -1, -1};
      for (int i = 0; i < 4; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color)
          moves.push_back(m);
      }
    }
  }

  void pawn_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    if (m.piece.promoted) {
      gold_moves(p, moves);
      return;
    }
    if (m.piece.color == White) {
      m.next = m.piece.pos + Position(0, 1);
      if (m.next.on_board() &&
          board[m.next.x][m.next.y].color != m.piece.color) {
        if (m.next.y != 4)
          moves.push_back(m);
        else {
          m.promote = true;
          moves.push_back(m);
          m.promote = false;
        }
      }
    } else {
      m.next = m.piece.pos + Position(0, -1);
      if (m.next.on_board() &&
          board[m.next.x][m.next.y].color != m.piece.color) {
        if (m.next.y != 0)
          moves.push_back(m);
        else {
          m.promote = true;
          moves.push_back(m);
          m.promote = false;
        }
      }
    }
  }

  void legal_king_moves(Piece p, std::vector<Move>& moves) {
    // fprintf(stderr, "legal king moves (%c, %d)\n", p.pos.x+'A', p.pos.y);
    Move m;
    m.piece = p;
    m.promote = false;
    short dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    short dy[] = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int i = 0; i < 8; ++i) {
      m.next = m.piece.pos + Position(dx[i], dy[i]);
      if (m.next.on_board() &&
          board[m.next.x][m.next.y].color != m.piece.color) {
        Piece pp;
        if (board[m.next.x][m.next.y].color != Empty) {
          pp = board[m.next.x][m.next.y];
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.pos == pp.pos) {
              i.pos = Position(-1, -1);
              break;
            }
          }
        }
        board[m.piece.pos.x][m.piece.pos.y] = Piece();
        board[m.next.x][m.next.y] = m.piece;

        if (!check(m.next, opponent(m.piece.color))) {
          moves.push_back(m);
        }

        board[m.next.x][m.next.y] = pp;
        board[m.piece.pos.x][m.piece.pos.y] = m.piece;
        for (auto& i : chess[opponent(m.piece.color)]) {
          if (i.type == pp.type && !i.pos.on_board()) {
            i.pos = pp.pos;
            break;
          }
        }
      }
    }

    // fprintf(stderr, "end legal king moves\n");
  }

  void legal_gold_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }
    if (m.piece.color == White) {
      short dx[] = {1, 1, 0, -1, -1, 0};
      short dy[] = {0, 1, 1, 1, 0, -1};
      for (int i = 0; i < 6; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        // fprintf(stderr, "g next: %c %d\n", m.next.x+'A', m.next.y);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          Piece pp;
          if (board[m.next.x][m.next.y].color != Empty) {
            pp = board[m.next.x][m.next.y];
            for (auto& i : chess[opponent(m.piece.color)]) {
              if (i.pos == pp.pos) {
                i.pos = Position(-1, -1);
                break;
              }
            }
          }
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          board[m.next.x][m.next.y] = m.piece;

          // std::string str;
          // str += "   A| B| C| D| E\n";
          // for(int i=Dy-1; i>=0; --i) {
          //     str += std::to_string(i) + ' ';
          //     for(int j=0; j<Dx; ++j) {
          //         if(j > 0) str += '|';
          //         str += board[j][i].print();
          //     }
          //     str += '\n';
          // }
          // std::cerr << str;

          if (!check(king.pos, opponent(king.color)))
            moves.push_back(m);

          board[m.next.x][m.next.y] = pp;
          board[m.piece.pos.x][m.piece.pos.y] = m.piece;
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.type == pp.type && !i.pos.on_board()) {
              i.pos = pp.pos;
              break;
            }
          }
        }
      }
    } else {
      short dx[] = {1, 0, -1, -1, 0, 1};
      short dy[] = {0, 1, 0, -1, -1, -1};
      for (int i = 0; i < 6; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        // fprintf(stderr, "g next: %c %d\n", m.next.x+'A', m.next.y);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          Piece pp;
          if (board[m.next.x][m.next.y].color != Empty) {
            pp = board[m.next.x][m.next.y];
            for (auto& i : chess[opponent(m.piece.color)]) {
              if (i.pos == pp.pos) {
                i.pos = Position(-1, -1);
                break;
              }
            }
          }
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          board[m.next.x][m.next.y] = m.piece;

          if (!check(king.pos, opponent(king.color)))
            moves.push_back(m);

          board[m.next.x][m.next.y] = pp;
          board[m.piece.pos.x][m.piece.pos.y] = m.piece;
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.type == pp.type && !i.pos.on_board()) {
              i.pos = pp.pos;
              break;
            }
          }
        }
      }
    }
  }

  void legal_silver_moves(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    if (m.piece.promoted) {
      legal_gold_moves(p, moves);
      return;
    }
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }
    if (m.piece.color == White) {
      short dx[] = {1, 0, -1, -1, 1};
      short dy[] = {1, 1, 1, -1, -1};
      for (int i = 0; i < 5; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          Piece pp;
          if (board[m.next.x][m.next.y].color != Empty) {
            pp = board[m.next.x][m.next.y];
            for (auto& i : chess[opponent(m.piece.color)]) {
              if (i.pos == pp.pos) {
                i.pos = Position(-1, -1);
                break;
              }
            }
          }
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          board[m.next.x][m.next.y] = m.piece;

          if (!check(king.pos, opponent(king.color))) {
            moves.push_back(m);
            if ((m.next.y == 4 || m.piece.pos.y == 4) && !m.piece.promoted) {
              m.promote = true;
              moves.push_back(m);
              m.promote = false;
            }
          }

          board[m.next.x][m.next.y] = pp;
          board[m.piece.pos.x][m.piece.pos.y] = m.piece;
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.type == pp.type && !i.pos.on_board()) {
              i.pos = pp.pos;
              break;
            }
          }
        }
      }
    } else {
      short dx[] = {1, -1, -1, 0, 1};
      short dy[] = {1, 1, -1, -1, -1};
      for (int i = 0; i < 5; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          Piece pp;
          if (board[m.next.x][m.next.y].color != Empty) {
            pp = board[m.next.x][m.next.y];
            for (auto& i : chess[opponent(m.piece.color)]) {
              if (i.pos == pp.pos) {
                i.pos = Position(-1, -1);
                break;
              }
            }
          }
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          board[m.next.x][m.next.y] = m.piece;

          if (!check(king.pos, opponent(king.color))) {
            moves.push_back(m);
            if ((m.next.y == 0 || m.piece.pos.y == 0) && !m.piece.promoted) {
              m.promote = true;
              moves.push_back(m);
              m.promote = false;
            }
          }

          board[m.next.x][m.next.y] = pp;
          board[m.piece.pos.x][m.piece.pos.y] = m.piece;
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.type == pp.type && !i.pos.on_board()) {
              i.pos = pp.pos;
              break;
            }
          }
        }
      }
    }
  }

  void legal_bishop_moves(Piece p, std::vector<Move>& moves) {
    // fprintf(stderr, "innnnnnnnnnnnnnBishop\n");
    // for(int i=0; i<2; ++i) {
    //     for(auto j : chess[i]) {
    //         fprintf(stderr, "(%c,%d) ", j.pos.x+'A', j.pos.y);
    //     }
    //     std::cerr << std::endl;
    // }
    Move m;
    m.piece = p;
    m.promote = false;
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }
    short dx[] = {1, -1, -1, 1};
    short dy[] = {1, 1, -1, -1};
    for (int i = 0; i < 4; ++i) {
      m.next = m.piece.pos + Position(dx[i], dy[i]);
      while (m.next.on_board() &&
             board[m.next.x][m.next.y].color != m.piece.color) {
        Piece pp;
        if (board[m.next.x][m.next.y].color != Empty) {
          pp = board[m.next.x][m.next.y];
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.pos == pp.pos) {
              i.pos = Position(-1, -1);
              break;
            }
          }
        }
        board[m.piece.pos.x][m.piece.pos.y] = Piece();
        board[m.next.x][m.next.y] = m.piece;

        if (!check(king.pos, opponent(king.color))) {
          moves.push_back(m);
          if (m.piece.color == White) {
            if ((m.next.y == 4 || m.piece.pos.y == 4) && !m.piece.promoted) {
              m.promote = true;
              moves.push_back(m);
              m.promote = false;
            }
          } else {  // Black
            if ((m.next.y == 0 || m.piece.pos.y == 0) && !m.piece.promoted) {
              m.promote = true;
              moves.push_back(m);
              m.promote = false;
            }
          }
        }

        board[m.next.x][m.next.y] = pp;
        board[m.piece.pos.x][m.piece.pos.y] = m.piece;
        for (auto& i : chess[opponent(m.piece.color)]) {
          if (i.type == pp.type && !i.pos.on_board()) {
            i.pos = pp.pos;
            break;
          }
        }

        if (board[m.next.x][m.next.y].color != Empty)
          break;
        m.next = m.next + Position(dx[i], dy[i]);
      }
    }
    if (m.piece.promoted) {
      short dx[] = {1, 0, -1, 0};
      short dy[] = {0, 1, 0, -1};
      for (int i = 0; i < 4; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          Piece pp;
          if (board[m.next.x][m.next.y].color != Empty) {
            pp = board[m.next.x][m.next.y];
            for (auto& i : chess[opponent(m.piece.color)]) {
              if (i.pos == pp.pos) {
                i.pos = Position(-1, -1);
                break;
              }
            }
          }
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          board[m.next.x][m.next.y] = m.piece;

          if (!check(king.pos, opponent(king.color))) {
            moves.push_back(m);
          }

          board[m.next.x][m.next.y] = pp;
          board[m.piece.pos.x][m.piece.pos.y] = m.piece;
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.type == pp.type && !i.pos.on_board()) {
              i.pos = pp.pos;
              break;
            }
          }
        }
      }
    }
    // fprintf(stderr, "innnnnnnnnnnnnnBiiiiiii\n");
    // for(int i=0; i<2; ++i) {
    //     for(auto j : chess[i]) {
    //         fprintf(stderr, "(%c,%d) ", j.pos.x+'A', j.pos.y);
    //     }
    //     std::cerr << std::endl;
    // }
  }

  void legal_rook_moves(Piece p, std::vector<Move>& moves) {
    // fprintf(stderr, "innnnnnnnnnnnnnRook\n");
    // for(int i=0; i<2; ++i) {
    //     for(auto j : chess[i]) {
    //         fprintf(stderr, "(%c,%d) ", j.pos.x+'A', j.pos.y);
    //     }
    //     std::cerr << std::endl;
    // }
    Move m;
    m.piece = p;
    m.promote = false;
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }
    short dx[] = {1, 0, -1, 0};
    short dy[] = {0, 1, 0, -1};
    for (int i = 0; i < 4; ++i) {
      m.next = m.piece.pos + Position(dx[i], dy[i]);
      while (m.next.on_board() &&
             board[m.next.x][m.next.y].color != m.piece.color) {
        Piece pp;
        if (board[m.next.x][m.next.y].color != Empty) {
          pp = board[m.next.x][m.next.y];
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.pos == pp.pos) {
              i.pos = Position(-1, -1);
              break;
            }
          }
        }
        board[m.piece.pos.x][m.piece.pos.y] = Piece();
        board[m.next.x][m.next.y] = m.piece;

        if (!check(king.pos, opponent(king.color))) {
          moves.push_back(m);
          if (m.piece.color == White) {
            if ((m.next.y == 4 || m.piece.pos.y == 4) && !m.piece.promoted) {
              m.promote = true;
              moves.push_back(m);
              m.promote = false;
            }
          } else {  // Black
            if ((m.next.y == 0 || m.piece.pos.y == 0) && !m.piece.promoted) {
              m.promote = true;
              moves.push_back(m);
              m.promote = false;
            }
          }
        }

        board[m.next.x][m.next.y] = pp;
        board[m.piece.pos.x][m.piece.pos.y] = m.piece;
        for (auto& i : chess[opponent(m.piece.color)]) {
          if (i.type == pp.type && !i.pos.on_board()) {
            i.pos = pp.pos;
            break;
          }
        }

        if (board[m.next.x][m.next.y].color != Empty)
          break;
        m.next = m.next + Position(dx[i], dy[i]);
      }
    }
    if (m.piece.promoted) {
      short dx[] = {1, -1, -1, 1};
      short dy[] = {1, 1, -1, -1};
      for (int i = 0; i < 4; ++i) {
        m.next = m.piece.pos + Position(dx[i], dy[i]);
        if (m.next.on_board() &&
            board[m.next.x][m.next.y].color != m.piece.color) {
          Piece pp;
          if (board[m.next.x][m.next.y].color != Empty) {
            pp = board[m.next.x][m.next.y];
            for (auto& i : chess[opponent(m.piece.color)]) {
              if (i.pos == pp.pos) {
                i.pos = Position(-1, -1);
                break;
              }
            }
          }
          board[m.piece.pos.x][m.piece.pos.y] = Piece();
          board[m.next.x][m.next.y] = m.piece;

          if (!check(king.pos, opponent(king.color))) {
            moves.push_back(m);
          }

          board[m.next.x][m.next.y] = pp;
          board[m.piece.pos.x][m.piece.pos.y] = m.piece;
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.type == pp.type && !i.pos.on_board()) {
              i.pos = pp.pos;
              break;
            }
          }
        }
      }
    }
    // fprintf(stderr, "innnnnnnnnnnnnnRook\n");
    // for(int i=0; i<2; ++i) {
    //     for(auto j : chess[i]) {
    //         fprintf(stderr, "(%c,%d) ", j.pos.x+'A', j.pos.y);
    //     }
    //     std::cerr << std::endl;
    // }
  }

  void legal_pawn_moves(Piece p, std::vector<Move>& moves) {

    Move m;
    m.piece = p;
    m.promote = false;
    if (m.piece.promoted) {
      legal_gold_moves(p, moves);
      return;
    }
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }
    if (m.piece.color == White) {
      m.next = m.piece.pos + Position(0, 1);
      if (m.next.on_board() &&
          board[m.next.x][m.next.y].color != m.piece.color) {
        Piece pp;
        if (board[m.next.x][m.next.y].color != Empty) {
          pp = board[m.next.x][m.next.y];
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.pos == pp.pos) {
              i.pos = Position(-1, -1);
              break;
            }
          }
        }
        board[m.piece.pos.x][m.piece.pos.y] = Piece();
        board[m.next.x][m.next.y] = m.piece;

        if (!check(king.pos, opponent(king.color))) {
          if (m.next.y != 4)
            moves.push_back(m);
          else if (!m.piece.promoted) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        }

        board[m.next.x][m.next.y] = pp;
        board[m.piece.pos.x][m.piece.pos.y] = m.piece;
        for (auto& i : chess[opponent(m.piece.color)]) {
          if (i.type == pp.type && !i.pos.on_board()) {
            i.pos = pp.pos;
            break;
          }
        }
      }
    } else {
      m.next = m.piece.pos + Position(0, -1);
      // fprintf(stderr, "pawn: %d %d\n", m.next.x, m.next.y);
      if (m.next.on_board() &&
          board[m.next.x][m.next.y].color != m.piece.color) {
        Piece pp;
        if (board[m.next.x][m.next.y].color != Empty) {
          pp = board[m.next.x][m.next.y];
          for (auto& i : chess[opponent(m.piece.color)]) {
            if (i.pos == pp.pos) {
              i.pos = Position(-1, -1);
              break;
            }
          }
        }
        board[m.piece.pos.x][m.piece.pos.y] = Piece();
        board[m.next.x][m.next.y] = m.piece;

        if (!check(king.pos, opponent(king.color))) {
          if (m.next.y != 0)
            moves.push_back(m);
          else if (!m.piece.promoted) {
            m.promote = true;
            moves.push_back(m);
            m.promote = false;
          }
        }

        board[m.next.x][m.next.y] = pp;
        board[m.piece.pos.x][m.piece.pos.y] = m.piece;
        for (auto& i : chess[opponent(m.piece.color)]) {
          if (i.type == pp.type && !i.pos.on_board()) {
            i.pos = pp.pos;
            break;
          }
        }
      }
    }
  }

  void legal_pawn_drop(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }

    // find another pawn
    int cannotdrop = Dx;
    for (auto c : chess[p.color]) {
      if ((c.type == PieceType::Pawn || c.type == PieceType::Pawn2) &&
          c.pos.x != -1)
        cannotdrop = c.pos.x;
    }

    if (p.color == White) {
      for (int i = 0; i < Dx; ++i) {
        if (i == cannotdrop)
          continue;
        for (int j = 0; j < Dy - 1; ++j) {
          if (board[i][j].color == Empty) {
            if (board[i][j + 1].type == PieceType::King &&
                board[i][j + 1].color != p.color) {
              board[i][j] = p;
              for (auto& v : chess[m.piece.color]) {
                if (v.type == p.type && !v.pos.on_board()) {
                  v.pos = Position(i, j);
                  break;
                }
              }

              bool cm = checkmate(board[i][j + 1].color);
              board[i][j] = Piece();

              for (auto& v : chess[m.piece.color]) {
                if (v.pos == Position(i, j)) {
                  v.pos = Position(-1, -1);
                  break;
                }
              }
              if (cm) {
                continue;
              }
            }
            m.next = Position(i, j);

            Piece pp;
            board[m.next.x][m.next.y] = m.piece;

            if (!check(king.pos, opponent(king.color))) {
              moves.push_back(m);
            }

            board[m.next.x][m.next.y] = pp;
          }
        }
      }
    } else {
      for (int i = 0; i < Dx; ++i) {
        if (i == cannotdrop)
          continue;
        for (int j = 1; j < Dy; ++j) {
          if (board[i][j].color == Empty) {
            if (board[i][j - 1].type == PieceType::King &&
                board[i][j - 1].color != p.color) {
              board[i][j] = p;
              for (auto& v : chess[m.piece.color]) {
                if (v.type == p.type && !v.pos.on_board()) {
                  v.pos = Position(i, j);
                  break;
                }
              }

              bool cm = checkmate(board[i][j - 1].color);
              board[i][j] = Piece();

              for (auto& v : chess[m.piece.color]) {
                if (v.pos == Position(i, j)) {
                  v.pos = Position(-1, -1);
                  break;
                }
              }
              if (cm) {
                continue;
              }
            }
            m.next = Position(i, j);
            Piece pp;
            board[m.next.x][m.next.y] = m.piece;

            if (!check(king.pos, opponent(king.color))) {
              moves.push_back(m);
            }

            board[m.next.x][m.next.y] = pp;
          }
        }
      }
    }
  }

  void legal_drop(Piece p, std::vector<Move>& moves) {
    Move m;
    m.piece = p;
    m.promote = false;
    Piece king;
    for (auto i : chess[p.color]) {
      if (i.type == PieceType::King) {
        king = i;
        break;
      }
    }
    for (int i = 0; i < Dx; ++i) {
      for (int j = 0; j < Dy; ++j) {
        if (board[i][j].color == Empty) {
          m.next = Position(i, j);
          Piece pp;
          board[m.next.x][m.next.y] = m.piece;

          if (!check(king.pos, opponent(king.color)))
            moves.push_back(m);

          board[m.next.x][m.next.y] = pp;
        }
      }
    }
  }

  int opponent(int player) const {
    if (player == White)
      return Black;
    return White;
  }

  // pos: king position
  // opponent can eat king
  bool check(Position pos, int op) {
    for (auto i : chess[op]) {
      std::vector<Move> moves;
      if (i.pos.on_board()) {
        switch (i.type) {
        case PieceType::King:
          king_moves(i, moves);
          break;

        case PieceType::Gold:
        case PieceType::Gold2:
          gold_moves(i, moves);
          break;

        case PieceType::Silver:
        case PieceType::Silver2:
          silver_moves(i, moves);
          break;

        case PieceType::Bishop:
        case PieceType::Bishop2:
          bishop_moves(i, moves);
          break;

        case PieceType::Rook:
        case PieceType::Rook2:
          rook_moves(i, moves);
          break;

        case PieceType::Pawn:
        case PieceType::Pawn2:
          pawn_moves(i, moves);
          break;

        default:
          break;
        }
      }
      for (auto m : moves) {
        if (m.next == pos) {
          return true;
        }
      }
    }
    return false;
  }

  bool checkmate(int color) {
    std::vector<Move> moves;

    for (auto i : chess[color]) {
      legalMoves(i, moves);
    }
    return moves.empty();
  }

  void legalMoves(Piece p, std::vector<Move>& moves) {
    if (p.pos.on_board()) {
      switch (p.type) {
      case PieceType::King:
        legal_king_moves(p, moves);
        break;

      case PieceType::Gold:
      case PieceType::Gold2:
        legal_gold_moves(p, moves);
        break;

      case PieceType::Silver:
      case PieceType::Silver2:
        legal_silver_moves(p, moves);
        break;

      case PieceType::Bishop:
      case PieceType::Bishop2:
        legal_bishop_moves(p, moves);
        break;

      case PieceType::Rook:
      case PieceType::Rook2:
        legal_rook_moves(p, moves);
        break;

      case PieceType::Pawn:
      case PieceType::Pawn2:
        legal_pawn_moves(p, moves);
        break;

      default:
        break;
      }
    } else {
      switch (p.type) {
      case PieceType::King:
        fprintf(stderr, "Error: King drop\n");
        break;

      case PieceType::Gold:
      case PieceType::Gold2:
      case PieceType::Silver:
      case PieceType::Silver2:
      case PieceType::Bishop:
      case PieceType::Bishop2:
      case PieceType::Rook:
      case PieceType::Rook2:
        legal_drop(p, moves);
        break;

      case PieceType::Pawn:
      case PieceType::Pawn2:
        legal_pawn_drop(p, moves);
        break;

      default:
        break;
      }
    }
  }

  int type_to_z(Piece p) {
    if (!p.promoted)
      return (int)p.type - 1;
    switch (p.type) {
    case PieceType::Silver:
      return 11;
    case PieceType::Bishop:
      return 12;
    case PieceType::Rook:
      return 13;
    case PieceType::Pawn:
      return 14;
    case PieceType::Silver2:
      return 15;
    case PieceType::Bishop2:
      return 16;
    case PieceType::Rook2:
      return 17;
    case PieceType::Pawn2:
      return 18;

    default:
      fprintf(
          stderr, "%s type to z error %d\n", p.print().c_str(), (int)p.type);
      return -1;
    }
  }

  bool z_promoted(int z) const {
    return z >= 11;
  }

  PieceType z_to_type(int z) const {
    switch (z) {
    case 0:
      return PieceType::King;
    case 1:
      return PieceType::Gold;
    case 2:
      return PieceType::Silver;
    case 3:
      return PieceType::Bishop;
    case 4:
      return PieceType::Rook;
    case 5:
      return PieceType::Pawn;
    case 6:
      return PieceType::Gold2;
    case 7:
      return PieceType::Silver2;
    case 8:
      return PieceType::Bishop2;
    case 9:
      return PieceType::Rook2;
    case 10:
      return PieceType::Pawn2;
    case 11:
      return PieceType::Silver;
    case 12:
      return PieceType::Bishop;
    case 13:
      return PieceType::Rook;
    case 14:
      return PieceType::Pawn;
    case 15:
      return PieceType::Silver2;
    case 16:
      return PieceType::Bishop2;
    case 17:
      return PieceType::Rook2;
    case 18:
      return PieceType::Pawn2;

    default:
      fprintf(stderr, "z to type error %d\n", z);
      return PieceType::None;
    }
  }

  PieceType new_type(PieceType p) const {
    PieceType t = p;
    switch (p) {
    case PieceType::Gold:
      t = PieceType::Gold2;
      break;
    case PieceType::Gold2:
      t = PieceType::Gold;
      break;
    case PieceType::Silver:
      t = PieceType::Silver2;
      break;
    case PieceType::Silver2:
      t = PieceType::Silver;
      break;
    case PieceType::Bishop:
      t = PieceType::Bishop2;
      break;
    case PieceType::Bishop2:
      t = PieceType::Bishop;
      break;
    case PieceType::Rook:
      t = PieceType::Rook2;
      break;
    case PieceType::Rook2:
      t = PieceType::Rook;
      break;
    case PieceType::Pawn:
      t = PieceType::Pawn2;
      break;
    case PieceType::Pawn2:
      t = PieceType::Pawn;
      break;
    default:
      break;
    }
    return t;
  }
};
