/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Author: 葉士誠 (SHI-CHENG YE)
// affiliation: National Dong Hwa University(NDHU)
// email: 410521206@gms.ndhu.edu.tw / 0930164@gmail.com

#include "chinesecheckers.h"

namespace ChineseCheckers {

string Board::sprintBoard(string_view prefix,
                          const set<tuple<int, int>>& markedPos) const {
  prefix.data();                    // ignore compile warning
  assert(markedPos.size() == 0UL);  // ignore compile warning
  return showBoard(false);
}

string Board::getPosStr(int p) const {
  assert(isPosInBoard(p));
  int tp = p % 10 + 1;
  string curPOS = to_string(tp);
  if (p / 10 > 5) {
    tp = (p - 60) % 61 + 1;
    curPOS = to_string(tp);
    return tp + 1 > 10 ? "G" + curPOS : "G0" + curPOS;
  } else {
    switch (p / 10) {
    case 0:
      return tp + 1 > 10 ? "A" + curPOS : "A0" + curPOS;
    case 1:
      return tp + 1 > 10 ? "B" + curPOS : "B0" + curPOS;
    case 2:
      return tp + 1 > 10 ? "C" + curPOS : "C0" + curPOS;
    case 3:
      return tp + 1 > 10 ? "D" + curPOS : "D0" + curPOS;
    case 4:
      return tp + 1 > 10 ? "E" + curPOS : "E0" + curPOS;
    case 5:
      return tp + 1 > 10 ? "F" + curPOS : "F0" + curPOS;
    default:
      tp = (p - 60) % 61 + 1;
      curPOS = to_string(tp);
      return tp + 1 > 10 ? "G" + curPOS : "G0" + curPOS;
    }
  }
}

string Board::showBoard(bool prePlay) const {
  ostringstream ostr;
  auto f = [&](bool prePlay, int p) {
    string_view sv;
    ostringstream ostr;
    if (!prePlay && p != pass) {  // show cur board
      sv = getChessSymbol(getChess(p));
    } else {  // show legal move board
      if (find(legal.begin(), legal.end(), p) != legal.end()) {
        if (p == pass)
          sv = "You can input \"p\" or \"pass\"to pass\n";
        else
          sv = getPosStr(p);
      } else
        sv = getChessSymbol(Chesses::empty);
    }
    ostr << sv;
    return ostr.str();
  };

  ostr << "                          A\n";
  ostr << "                         " << f(prePlay, P_A1).c_str() << endl;
  ostr << "                         / \\\n";
  ostr << "                       " << f(prePlay, P_A2).c_str() << "-"
       << f(prePlay, P_A3).c_str() << endl;
  ostr << "                       / \\ / \\\n";
  ostr << "                     " << f(prePlay, P_A4).c_str() << "-"
       << f(prePlay, P_A5).c_str() << "-" << f(prePlay, P_A6).c_str() << endl;
  ostr << "                     / \\ / \\ / \\\n";
  ostr << "                   " << f(prePlay, P_A7).c_str() << "-"
       << f(prePlay, P_A8).c_str() << "-" << f(prePlay, P_A9).c_str() << "-"
       << f(prePlay, P_A10).c_str() << endl;
  ostr << "F                  / \\ / \\ / \\ / \\                  B\n";
  ostr << " " << f(prePlay, P_F1).c_str() << "-" << f(prePlay, P_F3).c_str()
       << "-" << f(prePlay, P_F6).c_str() << "-" << f(prePlay, P_F10).c_str()
       << "-" << f(prePlay, P_G1).c_str() << "-" << f(prePlay, P_G2).c_str()
       << "-" << f(prePlay, P_G3).c_str() << "-" << f(prePlay, P_G4).c_str()
       << "-" << f(prePlay, P_G5).c_str() << "-" << f(prePlay, P_B7).c_str()
       << "-" << f(prePlay, P_B4).c_str() << "-" << f(prePlay, P_B2).c_str()
       << "-" << f(prePlay, P_B1).c_str() << endl;
  ostr << "   \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ /\n";
  ostr << "   " << f(prePlay, P_F2).c_str() << "-" << f(prePlay, P_F5).c_str()
       << "-" << f(prePlay, P_F9).c_str() << "-" << f(prePlay, P_G6).c_str()
       << "-" << f(prePlay, P_G7).c_str() << "-" << f(prePlay, P_G8).c_str()
       << "-" << f(prePlay, P_G9).c_str() << "-" << f(prePlay, P_G10).c_str()
       << "-" << f(prePlay, P_G11).c_str() << "-" << f(prePlay, P_B8).c_str()
       << "-" << f(prePlay, P_B5).c_str() << "-" << f(prePlay, P_B3).c_str()
       << endl;
  ostr << "     \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ /\n";
  ostr << "     " << f(prePlay, P_F4).c_str() << "-" << f(prePlay, P_F8).c_str()
       << "-" << f(prePlay, P_G12).c_str() << "-" << f(prePlay, P_G13).c_str()
       << "-" << f(prePlay, P_G14).c_str() << "-" << f(prePlay, P_G15).c_str()
       << "-" << f(prePlay, P_G16).c_str() << "-" << f(prePlay, P_G17).c_str()
       << "-" << f(prePlay, P_G18).c_str() << "-" << f(prePlay, P_B9).c_str()
       << "-" << f(prePlay, P_B6).c_str() << endl;
  ostr << "       \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ /\n";
  ostr << "       " << f(prePlay, P_F7).c_str() << "-"
       << f(prePlay, P_G19).c_str() << "-" << f(prePlay, P_G20).c_str() << "-"
       << f(prePlay, P_G21).c_str() << "-" << f(prePlay, P_G22).c_str() << "-"
       << f(prePlay, P_G23).c_str() << "-" << f(prePlay, P_G24).c_str() << "-"
       << f(prePlay, P_G25).c_str() << "-" << f(prePlay, P_G26).c_str() << "-"
       << f(prePlay, P_B10).c_str() << endl;
  ostr << "         \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ /\n";
  ostr << "         " << f(prePlay, P_G27).c_str() << "-"
       << f(prePlay, P_G28).c_str() << "-" << f(prePlay, P_G29).c_str() << "-"
       << f(prePlay, P_G30).c_str() << "-" << f(prePlay, P_G31).c_str() << "-"
       << f(prePlay, P_G32).c_str() << "-" << f(prePlay, P_G33).c_str() << "-"
       << f(prePlay, P_G34).c_str() << "-" << f(prePlay, P_G35).c_str() << endl;
  ostr << "         / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\\n";
  ostr << "       " << f(prePlay, P_E10).c_str() << "-"
       << f(prePlay, P_G36).c_str() << "-" << f(prePlay, P_G37).c_str() << "-"
       << f(prePlay, P_G38).c_str() << "-" << f(prePlay, P_G39).c_str() << "-"
       << f(prePlay, P_G40).c_str() << "-" << f(prePlay, P_G41).c_str() << "-"
       << f(prePlay, P_G42).c_str() << "-" << f(prePlay, P_G43).c_str() << "-"
       << f(prePlay, P_C7).c_str() << endl;
  ostr << "       / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\\n";
  ostr << "     " << f(prePlay, P_E6).c_str() << "-" << f(prePlay, P_E9).c_str()
       << "-" << f(prePlay, P_G44).c_str() << "-" << f(prePlay, P_G45).c_str()
       << "-" << f(prePlay, P_G46).c_str() << "-" << f(prePlay, P_G47).c_str()
       << "-" << f(prePlay, P_G48).c_str() << "-" << f(prePlay, P_G49).c_str()
       << "-" << f(prePlay, P_G50).c_str() << "-" << f(prePlay, P_C8).c_str()
       << "-" << f(prePlay, P_C4).c_str() << endl;
  ostr << "     / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\\n";
  ostr << "   " << f(prePlay, P_E3).c_str() << "-" << f(prePlay, P_E5).c_str()
       << "-" << f(prePlay, P_E8).c_str() << "-" << f(prePlay, P_G51).c_str()
       << "-" << f(prePlay, P_G52).c_str() << "-" << f(prePlay, P_G53).c_str()
       << "-" << f(prePlay, P_G54).c_str() << "-" << f(prePlay, P_G55).c_str()
       << "-" << f(prePlay, P_G56).c_str() << "-" << f(prePlay, P_C9).c_str()
       << "-" << f(prePlay, P_C5).c_str() << "-" << f(prePlay, P_C2).c_str()
       << endl;
  ostr << "   / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\ / \\\n";
  ostr << " " << f(prePlay, P_E1).c_str() << "-" << f(prePlay, P_E2).c_str()
       << "-" << f(prePlay, P_E4).c_str() << "-" << f(prePlay, P_E7).c_str()
       << "-" << f(prePlay, P_G57).c_str() << "-" << f(prePlay, P_G58).c_str()
       << "-" << f(prePlay, P_G59).c_str() << "-" << f(prePlay, P_G60).c_str()
       << "-" << f(prePlay, P_G61).c_str() << "-" << f(prePlay, P_C10).c_str()
       << "-" << f(prePlay, P_C6).c_str() << "-" << f(prePlay, P_C3).c_str()
       << "-" << f(prePlay, P_C1).c_str() << endl;
  ostr << "E                  \\ / \\ / \\ / \\ /                  C\n";
  ostr << "                   " << f(prePlay, P_D10).c_str() << "-"
       << f(prePlay, P_D9).c_str() << "-" << f(prePlay, P_D8).c_str() << "-"
       << f(prePlay, P_D7).c_str() << endl;
  ostr << "                     \\ / \\ / \\ /\n";
  ostr << "                     " << f(prePlay, P_D6).c_str() << "-"
       << f(prePlay, P_D5).c_str() << "-" << f(prePlay, P_D4).c_str() << endl;
  ostr << "                       \\ / \\ /\n";
  ostr << "                       " << f(prePlay, P_D3).c_str() << "-"
       << f(prePlay, P_D2).c_str() << endl;
  ostr << "                         \\ /\n";
  ostr << "                         " << f(prePlay, P_D1).c_str() << endl;
  ostr << "                          D\n";
  ostr << f(prePlay, pass);
  return ostr.str();
}

void Game::initialize() {
  board.initialize();
  setInitialChesses();
  hands = 0;
  prePos = -1;
  preChess = -1;
  continue_jump = false;
  winner = -1;
  legalMoves.clear();
  way.clear();
}

void Game::setInitialChesses() {
  Chess chess;
  for (int i = 0; i < boardWH * boardWH; i++) {
    if (i >= P_A1 && i <= P_A10)
      chess = Chesses::White;
    else if (i >= P_D1 && i <= P_D10)
      chess = Chesses::Black;
    else
      chess = Chesses::empty;
    board.setChess(i, chess);
  }
}

void Game::play(Move& m) {
  if (hands >= maxHands)
    fprintf(stderr, "hand %d out of range 0~%d", hands, maxHands - 1);
  assert(m.chess == board.getChess(m.x));

  board.setChess(m.x, Chesses::empty);
  board.setChess(m.tx, m.chess);
  way.push_back(m);
}

bool Game::Won() {
  bool win = true;
  for (int i = P_D1; i <= P_D10; i++)
    if (board.getChess(i) != Chesses::White) {
      win = false;
      break;
    }
  if (win) {
    winner = Players::player0;
    return true;
  }

  win = true;
  for (int i = P_A1; i <= P_A10; i++)
    if (board.getChess(i) != Chesses::Black) {
      win = false;
      break;
    }

  if (win) {
    winner = Players::player1;
    return true;
  }

  return false;
}

void Game::findLegalMoves(Player player) {
  legalMoves.clear();
  Chess chess = playerToChess(player);
  for (int i = 0; i < boardWH * boardWH; i++)
    if (board.getChess(i) == chess) {
      LegalMove(i, chess, true);
      LegalMove(i, chess, false);
    }
}

void Game::LegalMove(int cur, Chess chess, bool isJump) {
  Move m = Move{};
  m.x = cur;
  m.chess = chess;
  m.method = isJump ? 1 : 0;
  // byJump is decide to whether change method, cur is transform from (x,y):
  //   2D -> 1D
  int next, byJump = isJump ? 6 : 0;
  for (int md = M_LU + byJump; md <= M_RD + byJump; md++) {
    next = canGo(cur, md, chess);
    if (next >= 0 && next <= 120) {
      m.tx = next;
      legalMoves.push_back(m);
    }
  }
}

int Game::canGo(int cur, int md, Chess chess) {
  // six direction, two method, refer to define in ChineseCheckers_graph.h
  // md%6 is take pure direction
  int dirct = md % 6, next;
  bool isJump = md / 6 == 1;
  if (cur >= 0 && cur <= 120 &&
      board.getChess(cur) == chess) {  // check choose right chess
    next = isJump ? JumpOBS(cur, dirct) : move_table[cur][dirct];

    if (next >= 0 && next <= 120 && board.getChess(next) == Chesses::empty)
      return next;
    return -1;
  }
  return -1;
}

int Game::JumpOBS(int cur, int dirct) {
  for (int step = 1; step <= 6;
       step++) {  // up to six steps, or it'll jump outside
    cur = move_table[cur][dirct];
    if (cur >= 0 && cur <= 120) {
      if (board.getChess(cur) != Chesses::empty) {  // find one obstacle
        for (int j = 0; j < step; j++) {
          cur = move_table[cur][dirct];
          if (cur >= 0 && cur <= 120) {
            if (board.getChess(cur) != Chesses::empty) {
              return -1;  // find >2 obstacles -> can't jump
            }
          } else {
            return -1;  // out of chessboard
          }
        }
        return cur;
      }
    } else {
      return -1;  // out of chessboard
    }
  }
  return -1;  // must out of range
}

template <typename R> void Game::setupBoard(const R& re) {
  Board::setup({"Empty", "White", "Black"}, {"   ", " ● ", " ○ "}, re);
}

constexpr Player Game::chessToPlayer(Chess chess) {
  if (chess == Chesses::White)
    return Players::player0;
  else if (chess == Chesses::Black)
    return Players::player1;
  assert(chess == Chesses::White || chess == Chesses::Black);
  return -1;
}

constexpr Chess Game::playerToChess(Player player) {
  if (player == Players::player0)
    return Chesses::White;
  else if (player == Players::player1)
    return Chesses::Black;
  assert(player == Players::player0 || player == Players::player1);
  return 0xFF;
}

State::State(int seed)
    : core::State(seed)
    , Game() {
  // fprintf(stderr,"State(int seed)\n");
  call_once(setupCalled, [&] { setupBoard(_rng); });
}

void State::Initialize() {
  // fprintf(stderr,"Initialize()\n");
  _moves.clear();

  _featSize[0] = featuresSizeX;
  _featSize[1] = featuresSizeY;
  _featSize[2] = featuresSizeZ;

  _actionSize[0] = 3;  // move + jump + pass
  _actionSize[1] = boardWH * boardWH;
  _actionSize[2] = boardWH * boardWH;

  _status = GameStatus::player0Turn;
  _features.resize(featuresSize);

  initialize();
  findLegalMoves(Players::player0);
  findActions();
  findFeatures();
  _hash = board.getHash();
}

unique_ptr<core::State> State::clone_() const {
  // fprintf(stderr,"clone_()\n");
  return make_unique<State>(*this);
}

void State::ApplyAction(const _Action& action) {
  // fprintf(stderr,"ApplyAction()\n");
  if (!terminated()) {
    Move m{};
    Player Player = -1;
    // player0 plays white (●), player1 plays black (○)
    if (_status == GameStatus::player0Turn) {
      m.chess = Chesses::White;
      Player = Players::player0;
    } else if (_status == GameStatus::player1Turn) {
      m.chess = Chesses::Black;
      Player = Players::player1;
    }
    m.x = action.GetY();
    m.tx = action.GetZ();
    m.method = action.GetX();

    // check pass hand
    if (action.GetX() == 2) {
      if (canChange(Player)) {
        findActions();
        findFeatures();
      }
    } else {
      play(m);

      legalMoves.clear();

      if (m.method == 1) {  // jump
        LegalMove(m.tx, m.chess, m.method);
      }

      if (legalMoves.empty()) {  // move or no more jump
        if (canChange(Player)) {
          continue_jump = false;
          findActions();
          findFeatures();
        }
      } else {
        legalMoves.push_back(Pass);
        findActions();
        continue_jump = true;
      }
    }
  }
  _hash = board.getHash();
}

Player State::changeTurn(Player player) {
  Player nextPlayer = -1;
  if (player == Players::player1) {
    nextPlayer = Players::player0;
    _status = GameStatus::player0Turn;
  } else if (player == Players::player0) {
    nextPlayer = Players::player1;
    _status = GameStatus::player1Turn;
  }

  hands += 1;
  board.turnHash();
  way.clear();
  return nextPlayer;
}

bool State::canChange(Player player) {
  // fprintf(stderr,"canGoNext()\n");
  Player nextPlayer = changeTurn(player);
  if (Won()) {
    if (winner == Players::player1)
      _status = GameStatus::player1Win;
    else if (winner == Players::player0)
      _status = GameStatus::player0Win;
    else
      assert(winner == Players::player1 || winner == Players::player0);
    return false;
  }
  if (hands >= maxHands) {
    _status = GameStatus::tie;
    return false;
  }

  findLegalMoves(nextPlayer);
  assert(!legalMoves.empty());
  return true;
}

void State::findActions() {
  // fprintf(stderr,"findActions()\n");
  clearActions();
  for (auto& m : legalMoves) {
    if (m.x != pass && m.tx != pass) {
      addAction(m.method, m.x, m.tx);
    } else {
      addAction(2, 0, 0);
    }
  }
}

void State::findFeatures() {
  // fprintf(stderr,"findFeatures()\n");
  std::fill(_features.begin(), _features.end(), 0.0);
  auto* f = _features.data();
  for (size_t c = 0; c < chesses; c++) {
    Chess chess = static_cast<Chess>(c + 1);
    for (int i = 0; i < boardWH * boardWH; i++) {
      if (board.getChess(i) == chess)
        *f = 1.0;
      f++;
    }
  }
  fillFullFeatures();
}

void State::DoGoodAction() {
  // fprintf(stderr,"DoGoodAction()\n");
  // DoRandomAction();
  assert(!_legalActions.empty());
  std::uniform_int_distribution<size_t> distr(0, _legalActions.size() - 1);
  size_t i = distr(_rng);
  _Action a = _legalActions[i];
  int cur = a.GetY(), next = a.GetZ();
  Chess chess = board.getChess(cur);
  while (prePos == next && preChess == chess) {
    std::uniform_int_distribution<size_t> distr(0, _legalActions.size() - 1);
    // size_t i = distr(_rng);
    //_Action a = _NewlegalActions[i];
  }
  prePos = cur;
  preChess = chess;
  ApplyAction(a);
}

void State::printCurrentBoard() const {
  // fprintf(stderr,"printCurrentBoard()\n");
  cout << board.sprint("  ");
}

string State::stateDescription() const {
  // fprintf(stderr,"stateDescription()\n");
  return board.sprint("  ");
}

string State::actionDescription(const ::_Action& action) const {
  ostringstream ostr;
  if (continue_jump)
    ostr << "<continue jumps>\t";

  int cur = way.front().x, goal = action.GetZ(), method = action.GetX();
  if (method == 2) {
    ostr << " <-Pass-> ";
    goal = way[way.size() - 2].tx;
  } else if (method == 0)
    ostr << "move from ";
  else
    ostr << "jump from ";
  ostr << board.getPosStr(cur) << " to " << board.getPosStr(goal);
  return ostr.str();
}

string State::actionsDescription() const {
  // fprintf(stderr,"actionsDescription()\n");
  ostringstream ostr;
  auto board = this->board;
  board.legal.clear();

  for (auto& m : legalMoves) {
    board.legal.push_back(m.x);
  }
  ostr << "Chesses which can move:\n" << board.showBoard(true);
  return ostr.str();
}

int State::parseAction(const string& str) const {
  auto parse = [&](const string& s, int& x) {
    if (s == "pass" || s == "p") {
      x = pass;
      return true;
    }
    int pos = -1, len = 0;
    for (size_t i = 0; i < s.size(); i++) {
      if (!isalnum(s[i]) && pos >= 0)
        break;
      if (isalnum(s[i])) {
        if (pos < 0)
          pos = (int)i;
        len += 1;
      }
    }
    if (len == 0 || pos < 0)
      return false;
    string ts = s.substr(pos, len);
    if (ts.size() < 2)
      return false;
    else if (!isalpha(ts[0]))
      return false;
    string num = ts.substr(1, ts.size() - 1);
    int n = 0;
    for (char c : num) {
      if (!isdigit(c))
        return false;
      n = c - '0' + n * 10;
    }
    char c = toupper(s[0]);
    if (c >= 'A' && c <= 'F') {
      if (n <= 0 || n >= 11)
        return false;
      n--;
    }
    if (c == 'G') {
      if (n <= 0 || n >= 62)
        return false;
      n--;
    }
    switch (c) {
    case 'A':
      n += 0;
      break;
    case 'B':
      n += 10;
      break;
    case 'C':
      n += 20;
      break;
    case 'D':
      n += 30;
      break;
    case 'E':
      n += 40;
      break;
    case 'F':
      n += 50;
      break;
    case 'G':
      n += 60;
      break;
    default:
      return false;
    }
    x = n;
    return true;
  };
  int cur = -1, goal = -1;
  if (!parse(str, cur))
    return -1;

  auto board = this->board;
  board.legal.clear();
  bool found = false;
  for (auto& m : legalMoves) {
    if (m.x == cur) {
      board.legal.push_back(m.tx);
      found = true;
    }
  }
  if (!found) {
    cout << "Error! No any legal move." << endl;
    return -1;
  }

  if (cur != pass) {
    cout << board.showBoard(true);
    cout << "Where you wanna go:" << endl << "> ";
    string input;
    cin >> input;
    if (!parse(input, goal))
      return -1;
  }

  auto& legalActions = GetLegalActions();
  for (size_t i = 0; i < legalActions.size(); i++) {
    if (cur == pass) {
      if (legalActions[i].GetX() == 2) {
        return (int)i;
      }
    } else {
      if (legalActions[i].GetY() == cur && legalActions[i].GetZ() == goal)
        return (int)i;
    }
  }
  return -1;
}

int State::humanInputAction(function<optional<int>(string)> specialAction) {
  cout << "hands: " << hands << endl;
  cout << "Current Board:" << endl << stateDescription() << endl;
  cout << actionsDescription() << endl;
  cout << "Please choose chess and move to where: ";
  cout << "(Use \"a01\", \"B5\",\"C10\", etc. to correspond to the correct "
          "format in "
          "chess board)"
       << endl;

  string input;
  int index = -1;
  size_t max = GetLegalActions().size();
  while (index < 0 || index >= (int)max) {
    cout << "Chess you choose is:" << endl << "> ";
    cin >> input;
    index = parseAction(input);

    if (index == -1) {
      if (auto r = specialAction(input); r) {
        return *r;
      }
    }

    if (index < 0 || index >= (int)max) {
      cout << "This is invalid Input! Please try again." << endl;
      cout << "Current Board:" << endl << stateDescription() << endl;
      cout << actionsDescription() << endl;
    }
  }
  return index;
}

}  // namespace ChineseCheckers
