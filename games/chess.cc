/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "chess.h"

namespace chess {

struct ZobrishHash {
  std::array<std::array<uint64_t, 13 * 2>, 12 * 12> hash;
  ZobrishHash() {
    std::mt19937_64 rng(std::random_device{}() + 42);
    rng.discard(1024);
    for (auto& v : hash) {
      for (auto& v2 : v) {
        v2 = rng();
      }
    }
  }
};

static inline ZobrishHash zhash;

void ChessBoard::init() {
  board.fill(0);
  for (int i = 0; i != boardDim; ++i) {
    board[i] = OOB;
    board[i + boardDim] = OOB;
    board[boardDim * i] = OOB;
    board[boardDim * i + 1] = OOB;
    board[boardDim * (boardDim - 1) + i] = OOB;
    board[boardDim * (boardDim - 2) + i] = OOB;
    board[boardDim * i + boardDim - 1] = OOB;
    board[boardDim * i + boardDim - 2] = OOB;
  }

  board[boardDim * 2 + 2] = WHITE | ROOK;
  board[boardDim * 2 + 3] = WHITE | KNIGHT;
  board[boardDim * 2 + 4] = WHITE | BISHOP;
  board[boardDim * 2 + 5] = WHITE | QUEEN;
  board[boardDim * 2 + 6] = WHITE | KING;
  board[boardDim * 2 + 7] = WHITE | BISHOP;
  board[boardDim * 2 + 8] = WHITE | KNIGHT;
  board[boardDim * 2 + 9] = WHITE | ROOK;
  board[boardDim * (boardDim - 3) + 2] = BLACK | ROOK;
  board[boardDim * (boardDim - 3) + 3] = BLACK | KNIGHT;
  board[boardDim * (boardDim - 3) + 4] = BLACK | BISHOP;
  board[boardDim * (boardDim - 3) + 5] = BLACK | QUEEN;
  board[boardDim * (boardDim - 3) + 6] = BLACK | KING;
  board[boardDim * (boardDim - 3) + 7] = BLACK | BISHOP;
  board[boardDim * (boardDim - 3) + 8] = BLACK | KNIGHT;
  board[boardDim * (boardDim - 3) + 9] = BLACK | ROOK;
  for (int i = 0; i != 8; ++i) {
    board[boardDim * 3 + 2 + i] = WHITE | PAWN;
    board[boardDim * (boardDim - 4) + 2 + i] = BLACK | PAWN;
  }

  moveflags = castleleft | castleright;
  moveflags |= (castleleft | castleright) << 1;
  turn = 0;
  moves.clear();
  done = false;
  winner = -1;
  fiftyMoveCounter = 100;
}

void ChessBoard::findMoves() {

  int color = turn;
  const int ahead = color == 0 ? boardDim : -boardDim;

  char colorbit = 1 << (5 + color);
  char opponentcolorbit = 1 << (5 + (color ^ 1));
  char occupied = colorbit | OOB;

  moves.clear();

  size_t end = boardDim * (boardDim - 2) - 2;
  size_t king = 0;
  for (size_t i = 2 + boardDim * 2; i != end; ++i) {
    if (board[i] == (colorbit | KING)) {
      king = i;
    }
  }
  size_t kingx = king % boardDim;
  size_t kingy = king / boardDim;

  auto checkcheck = [&]() {
    uint8_t pawn = opponentcolorbit | PAWN;
    if ((board[king + ahead + 1] & pawn) == pawn)
      return true;
    if ((board[king + ahead - 1] & pawn) == pawn)
      return true;
    uint8_t rook = opponentcolorbit | ROOK;
    for (size_t xx = kingx + 1, ii = king + 1; xx != boardDim - 2; ++xx, ++ii) {
      if ((board[ii] & rook) == rook)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    for (size_t xx = kingx - 1, ii = king - 1; xx != 1; --xx, --ii) {
      if ((board[ii] & rook) == rook)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    for (size_t xx = kingy - 1, ii = king - boardDim; xx != 1;
         --xx, ii -= boardDim) {
      if ((board[ii] & rook) == rook)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    for (size_t xx = kingy + 1, ii = king + boardDim; xx != boardDim - 2;
         ++xx, ii += boardDim) {
      if ((board[ii] & rook) == rook)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    uint8_t bishop = opponentcolorbit | BISHOP;
    for (size_t xx = std::min(kingx, kingy) - 1, ii = king - 1 - boardDim;
         xx != 1; --xx, ii += -1 - boardDim) {
      if ((board[ii] & bishop) == bishop)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    for (size_t xx = std::min(kingx, boardDim - 1 - kingy) - 1,
                ii = king - 1 + boardDim;
         xx != 1; --xx, ii += -1 + boardDim) {
      if ((board[ii] & bishop) == bishop)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    for (size_t xx = std::min(boardDim - 1 - kingx, boardDim - 1 - kingy) - 1,
                ii = king + 1 + boardDim;
         xx != 1; --xx, ii += 1 + boardDim) {
      if ((board[ii] & bishop) == bishop)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    for (size_t xx = std::min(boardDim - 1 - kingx, kingy) - 1,
                ii = king + 1 - boardDim;
         xx != 1; --xx, ii += 1 - boardDim) {
      if ((board[ii] & bishop) == bishop)
        return true;
      else if (board[ii] != EMPTY)
        break;
    }
    uint8_t knight = opponentcolorbit | KNIGHT;
    if (board[king + ahead + ahead - 1] == knight)
      return true;
    if (board[king + ahead + ahead + 1] == knight)
      return true;
    if (board[king - ahead - ahead - 1] == knight)
      return true;
    if (board[king - ahead - ahead + 1] == knight)
      return true;
    if (board[king - 1 - 1 - ahead] == knight)
      return true;
    if (board[king - 1 - 1 + ahead] == knight)
      return true;
    if (board[king + 1 + 1 - ahead] == knight)
      return true;
    if (board[king + 1 + 1 + ahead] == knight)
      return true;
    return false;
  };

  for (size_t i = 2 + boardDim * 2; i != end; ++i) {
    if ((board[i] & colorbit) == 0)
      continue;
    int piece = board[i] & 0xf;

    unsigned char* relative = &board[i];

    auto check = [&](size_t to) {
      auto dst = board[to];
      auto src = board[i];
      board[to] = src;
      board[i] = EMPTY;
      bool r;
      if (piece == KING) {
        uint8_t knight = opponentcolorbit | KNIGHT;
        uint8_t pawn = opponentcolorbit | PAWN;
        if ((board[to + ahead] & 0xf) == KING)
          r = true;
        else if ((board[to + ahead + 1] & pawn) == pawn)
          r = true;
        else if ((board[to + ahead - 1] & pawn) == pawn)
          r = true;
        else if ((board[to + 1] & 0xf) == KING)
          r = true;
        else if ((board[to - 1] & 0xf) == KING)
          r = true;
        else if ((board[to - ahead] & 0xf) == KING)
          r = true;
        else if ((board[to - ahead + 1] & 0xf) == KING)
          r = true;
        else if ((board[to - ahead - 1] & 0xf) == KING)
          r = true;
        else if (board[to + ahead + ahead - 1] == knight)
          r = true;
        else if (board[to + ahead + ahead + 1] == knight)
          r = true;
        else if (board[to - ahead - ahead - 1] == knight)
          r = true;
        else if (board[to - ahead - ahead + 1] == knight)
          r = true;
        else if (board[to - 1 - 1 - ahead] == knight)
          r = true;
        else if (board[to - 1 - 1 + ahead] == knight)
          r = true;
        else if (board[to + 1 + 1 - ahead] == knight)
          r = true;
        else if (board[to + 1 + 1 + ahead] == knight)
          r = true;
        else {
          king = to;
          kingx = to % boardDim;
          kingy = to / boardDim;
          r = checkcheck();
          king = i;
          kingx = i % boardDim;
          kingy = i / boardDim;
        }
      } else {
        r = checkcheck();
      }
      board[i] = src;
      board[to] = dst;
      return r;
    };

    auto addMove = [&](size_t to) {
      if (!check(to)) {
        if (piece == PAWN && relative[ahead + ahead] == OOB) {
          moves.push_back(i | (to << 17));
          moves.push_back(i | (to << 17) | 0x8000);
          moves.push_back(i | (to << 17) | 0x10000);
          moves.push_back(i | (to << 17) | 0x18000);
        } else {
          moves.push_back(i | (to << 17));
        }
      }
    };

    auto test = [&](size_t to) {
      board.at(i + to);
      if ((relative[to] & occupied) == 0) {
        addMove(i + to);
        return true;
      } else {
        return false;
      }
    };

    auto tryMove = [&](size_t to) {
      board.at(to);
      int v = board[to];
      if (v != EMPTY) {
        if (v & occupied)
          return false;
        addMove(to);
        return false;
      }
      addMove(to);
      return true;
    };

    switch (piece) {
    case PAWN:
      if (relative[ahead] == EMPTY) {
        addMove(i + ahead);
        if (color == 0 ? i - boardDim * 3 < boardDim
                       : i - boardDim * (boardDim - 4) < boardDim) {
          if (relative[ahead + ahead] == EMPTY && !check(i + ahead + ahead)) {
            moves.push_back(i | 0x8000 | ((i + ahead + ahead) << 17));
          }
        }
      }
      if ((relative[ahead + 1] & opponentcolorbit) == opponentcolorbit) {
        addMove(i + ahead + 1);
      }
      if ((relative[ahead - 1] & opponentcolorbit) == opponentcolorbit) {
        addMove(i + ahead - 1);
      }
      if (moveflags & 0x8000) {
        size_t x = moveflags & 0x7fff;
        if (i + 1 == x) {
          int tmp = board[i + 1];
          board[i + 1] = EMPTY;
          if (!check(i + ahead + 1)) {
            moves.push_back(i | 0x10000 | ((i + ahead + 1) << 17));
          }
          board[i + 1] = tmp;
        } else if (i - 1 == x) {
          int tmp = board[i - 1];
          board[i - 1] = EMPTY;
          if (!check(i + ahead - 1)) {
            moves.push_back(i | 0x10000 | ((i + ahead - 1) << 17));
          }
          board[i - 1] = tmp;
        }
      }
      break;
    case KNIGHT:
      test(ahead + ahead + 1);
      test(ahead + ahead - 1);
      test(-ahead - ahead - 1);
      test(-ahead - ahead + 1);
      test(-1 - 1 - ahead);
      test(-1 - 1 + ahead);
      test(+1 + 1 - ahead);
      test(+1 + 1 + ahead);
      break;
    case KING:
      test(ahead);
      test(ahead + 1);
      test(ahead - 1);
      test(1);
      test(-1);
      test(-ahead);
      test(-ahead + 1);
      test(-ahead - 1);
      if ((moveflags & (castleleft << turn)) && !checkcheck()) {
        size_t x = i % boardDim;
        for (size_t xx = x - 1, ii = i - 1;; --xx, --ii) {
          if (xx == 2) {
            if (!check(i - 1) && !check(i - 2)) {
              moves.push_back(i | 0x8000 | ((i - 1 - 1) << 17));
            }
            break;
          }
          if (board[ii] != EMPTY)
            break;
        }
      }
      if ((moveflags & (castleright << turn)) && !checkcheck()) {
        size_t x = i % boardDim;
        for (size_t xx = x + 1, ii = i + 1;; ++xx, ++ii) {
          if (xx == boardDim - 3) {
            if (!check(i + 1) && !check(i + 2)) {
              moves.push_back(i | 0x8000 | ((i + 1 + 1) << 17));
            }
            break;
          }
          if (board[ii] != EMPTY)
            break;
        }
      }
      break;
    default:
      size_t x = i % boardDim;
      size_t y = i / boardDim;
      switch (piece) {
      case QUEEN:
      case ROOK:
        for (size_t xx = x + 1, ii = i + 1; xx != boardDim - 2 && tryMove(ii);
             ++xx, ++ii)
          ;
        for (size_t xx = x - 1, ii = i - 1; xx != 1 && tryMove(ii); --xx, --ii)
          ;
        for (size_t xx = y - 1, ii = i - boardDim; xx != 1 && tryMove(ii);
             --xx, ii -= boardDim)
          ;
        for (size_t xx = y + 1, ii = i + boardDim;
             xx != boardDim - 2 && tryMove(ii); ++xx, ii += boardDim)
          ;
        if (piece != QUEEN)
          break;
        [[fallthrough]];
      case BISHOP:
        for (size_t xx = std::min(x, y) - 1, ii = i - 1 - boardDim;
             xx != 1 && tryMove(ii); --xx, ii += -1 - boardDim)
          ;
        for (size_t xx = std::min(x, boardDim - 1 - y) - 1,
                    ii = i - 1 + boardDim;
             xx != 1 && tryMove(ii); --xx, ii += -1 + boardDim)
          ;
        for (size_t xx = std::min(boardDim - 1 - x, boardDim - 1 - y) - 1,
                    ii = i + 1 + boardDim;
             xx != 1 && tryMove(ii); --xx, ii += 1 + boardDim)
          ;
        for (size_t xx = std::min(boardDim - 1 - x, y) - 1,
                    ii = i + 1 - boardDim;
             xx != 1 && tryMove(ii); --xx, ii += 1 - boardDim)
          ;
        break;
      }
    }
  }

  if (moves.empty()) {
    done = true;
    if (checkcheck()) {
      winner = turn ^ 1;
    }
  } else if (fiftyMoveCounter <= 0) {
    done = true;
    winner = -1;
  }
}

void ChessBoard::move(uint_fast32_t move) {
  size_t to = move >> 17;
  size_t from = move & 0x7fff;

  int v = board[from];
  int piece = v & 0xf;

  --fiftyMoveCounter;

  switch (piece) {
  case KING:
    moveflags &= ~((castleleft | castleright) << turn);
    break;
  case ROOK: {
    size_t x = from % boardDim;
    size_t y = from / boardDim;
    if (x == 2 || x == boardDim - 3) {
      if (y == (turn == 0 ? 2 : boardDim - 3)) {
        if (x == 2)
          moveflags &= ~(castleleft << turn);
        else
          moveflags &= ~(castleright << turn);
      }
    }
    break;
  }
  case PAWN:
    fiftyMoveCounter = 100;
    for (auto& v : repetitions) {
      v.clear();
    }
    if (board[to + (turn == 0 ? boardDim : -boardDim)] == OOB) {
      v &= ~PAWN;
      switch ((move >> 15) & 3) {
      case 0:
        v |= QUEEN;
        break;
      case 1:
        v |= ROOK;
        break;
      case 2:
        v |= BISHOP;
        break;
      case 3:
        v |= KNIGHT;
        break;
      }
      move = 0;
    }
    break;
  }

  moveflags &= ~0xffff;
  if ((move & 0x8000) != 0) {
    if (piece == PAWN) {
      moveflags |= to | 0x8000;
    } else if (piece == KING) {
      size_t y = from / boardDim;
      if (to < from) {
        std::swap(board[y * boardDim + 2], board[to + 1]);
      } else {
        std::swap(board[y * boardDim + boardDim - 3], board[to - 1]);
      }
    }
  } else {
    if ((move & 0x10000) != 0) {
      int dx = to % boardDim - from % boardDim;
      board.at(from + dx) = EMPTY;

      fiftyMoveCounter = 100;
      for (auto& v : repetitions) {
        v.clear();
      }
    }
  }

  if (board[to] != EMPTY) {
    fiftyMoveCounter = 100;
    for (auto& v : repetitions) {
      v.clear();
    }
  }
  if ((board[to] & 0xf) == ROOK) {
    size_t x = to % boardDim;
    size_t y = to / boardDim;
    if (x == 2 || x == boardDim - 3) {
      if (y == (turn == 1 ? 2 : boardDim - 3)) {
        if (x == 2)
          moveflags &= ~(castleleft << (turn ^ 1));
        else
          moveflags &= ~(castleright << (turn ^ 1));
      }
    }
  }

  hash ^= zhash.hash.at(from).at(13 * turn + piece);
  hash ^= zhash.hash.at(to).at(13 * turn + piece);

  board[from] = EMPTY;
  board[to] = v;

  turn ^= 1;

  uint64_t fullhash = hash ^ moveflags ^ zhash.hash.at(5).at(turn);
  size_t index = fullhash % repetitions.size();
  bool found = false;
  for (auto& v : repetitions[index]) {
    if (v.first == fullhash) {
      ++v.second;
      if (v.second >= 3) {
        done = true;
        winner = -1;
        found = true;
      }
      break;
    }
  }
  if (!found) {
    repetitions[index].emplace_back(fullhash, 1);
  }
}

std::string ChessBoard::moveString(uint_fast32_t move) const {
  size_t from = move & 0x7fff;
  size_t fx = from % boardDim - 2;
  size_t fy = from / boardDim - 2;
  size_t to = move >> 17;
  size_t tx = to % boardDim - 2;
  size_t ty = to / boardDim - 2;
  int piece = board[from];
  bool disx = false;
  bool disy = false;
  for (auto& v : moves) {
    size_t vfrom = v & 0x7fff;
    size_t vto = v >> 17;
    if (vfrom != from && board[vfrom] == piece && vto == to) {
      size_t vx = vfrom % boardDim - 2;
      if (vx == fx) {
        disy = true;
      } else {
        disx = true;
      }
    }
  }

  std::string str;
  switch (piece & 0xf) {
  case KNIGHT:
    str += 'N';
    break;
  case BISHOP:
    str += 'B';
    break;
  case ROOK:
    str += 'R';
    break;
  case QUEEN:
    str += 'Q';
    break;
  case KING:
    str += 'K';
    break;
  }
  bool promotion = false;
  if ((piece & 0xf) == PAWN) {
    if (board[to + (turn == 0 ? boardDim : -boardDim)] == OOB) {
      promotion = true;
    }
  } else if ((piece & 0xf) == KING) {
    if ((move & 0x8000) != 0) {
      if (to < from) {
        return "O-O-O";
      } else {
        return "O-O";
      }
    }
  }
  bool capture = board[to] != EMPTY;
  if (!promotion && (move & 0x10000) != 0) {
    capture = true;
  }
  if (capture && str.empty()) {
    disx = true;
  }
  if (disx) {
    str += char('a' + fx);
  }
  if (disy) {
    str += char('1' + fy);
  }
  if (capture != EMPTY) {
    str += 'x';
  }
  str += char('a' + tx);
  str += char('1' + ty);
  if (promotion) {
    switch ((move >> 15) & 3) {
    case 0:
      str += "=Q";
      break;
    case 1:
      str += "=R";
      break;
    case 2:
      str += "=B";
      break;
    case 3:
      str += "=N";
      break;
    }
  }
  return str;
}

}  // namespace chess
