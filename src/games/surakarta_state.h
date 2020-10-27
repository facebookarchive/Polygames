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

#include "../core/state.h"

typedef unsigned short Coord;

#include "time.h"
#include <iostream>
#include <random>
#include <string>
#include <vector>

const int StateForSurakartaX = 2;
const int StateForSurakartaY = 6;
const int StateForSurakartaZ = 6;

#include "surakarta.h"

class StateForSurakarta : public core::State, public SKBoard {
 public:
  StateForSurakarta(int seed)
      : State(seed) {
  }

  virtual void Initialize() override {
    _moves.clear();

    // the features are just one number between 0 and 1 (the distance,
    // normalized).
    _featSize[0] = StateForSurakartaX;
    _featSize[1] = StateForSurakartaY;
    _featSize[2] = StateForSurakartaZ;

    // size of the output of the neural network; this should cover the positions
    // of actions (above).
    _actionSize[0] = 36;
    _actionSize[1] = 6;
    _actionSize[2] = 6;

    _hash = 0;
    _status = GameStatus::player0Turn;

    _features.resize(StateForSurakartaX * StateForSurakartaY *
                     StateForSurakartaZ);

    init();
    findFeatures();
    findActions(SuraWhite);
    fillFullFeatures();
  }

  virtual std::string stateDescription() const override {
    std::string str;
    str += "  A|B|C|D|E|F\n";
    for (int i = SKDy - 1; i >= 0; i--) {
      str += to_string(i + 1) + ' ';
      for (int j = 0; j < SKDx; j++) {
        if (j > 0)
          str += '|';
        if (board[j][i] == SuraEmpty)
          str += ' ';
        else if (board[j][i] == SuraBlack)
          str += 'x';
        else
          str += 'o';
      }
      str += '\n';
    }

    return str;
  }

  virtual std::string actionsDescription() const override {
    std::stringstream ss;
    char x, y, x1, y1;
    for (int i = 0; i < (int)_legalActions.size(); i++) {
      const _Action& action = _legalActions[i];
      x = static_cast<char>(action.GetY() + 'A');
      y = static_cast<char>(action.GetZ() + '1');
      int curY = action.GetX() / SKDx;
      y1 = static_cast<char>(curY + '1');
      x1 = static_cast<char>(action.GetX() - curY * SKDx + 'A');
      ss << "Action " << i << ": " << x << y << "-" << x1 << y1 << std::endl;
    }
    ss << "\nInput format : <Alphabet><Digit>-<Alphabet><Digit> e.g. A1-A2\n";
    return ss.str();
  }

  virtual std::string actionDescription(const _Action& action) const {
    std::stringstream ss;
    char x, y, x1, y1;
    x = static_cast<char>(action.GetY() + 'A');
    y = static_cast<char>(action.GetZ() + '1');
    int curY = action.GetX() / SKDx;
    y1 = static_cast<char>(curY + '1');
    x1 = static_cast<char>(action.GetX() - curY * SKDx + 'A');
    ss << x << y << "-" << x1 << y1;

    return ss.str();
  }

  virtual int parseAction(const std::string& str) const override {
    int x = -1, y = -1, x1 = -1, y1 = -1;
    if (!isalpha(str[0]) || !isalpha(str[3]))
      return -1;
    if (!isdigit(str[1]) || !isdigit(str[4]))
      return -1;

    x = static_cast<int>(toupper(str[0]) - 'A');
    y = static_cast<int>(str[1] - '1');
    x1 = static_cast<int>(toupper(str[3]) - 'A');
    y1 = static_cast<int>(str[4] - '1');

    if (x < 0 || y < 0 || x1 < 0 || y1 < 0 || x >= SKDx || y >= SKDy ||
        x1 >= SKDx || y1 >= SKDy)
      return -1;

    for (int i = 0; i < (int)_legalActions.size(); i++) {
      if (_legalActions[i].GetX() == (y1 * SKDy + x1) &&
          _legalActions[i].GetY() == x && _legalActions[i].GetZ() == y)
        return i;
    }
    return -1;
  }

  virtual std::unique_ptr<core::State> clone_() const override {
    return std::make_unique<StateForSurakarta>(*this);
  }

  void findActions(int color) {
    vector<SKMove> moves;
    int nb = legalMoves(color, moves);
    int nc = legalCaptures(nb, color, moves);

    clearActions();
    for (int i = 0; i < nc; i++) {
      int x = moves[i].x;
      int y = moves[i].y;
      int final_pos = moves[i].y1 * SKDy + moves[i].x1;
      addAction(final_pos, x, y);
    }
  }

  void findFeatures() {
    if ((_status == GameStatus::player0Win) ||
        (_status == GameStatus::player1Win) || (_status == GameStatus::tie))
      return;
    if (_status == GameStatus::player0Turn) {  // SuraWhite
      for (int i = 0; i < 72; i++)
        _features[i] = 0;
      for (int i = 0; i < 36; i++)
        if (board[i % 6][i / 6] == SuraBlack)
          _features[i] = 1;
      for (int i = 0; i < 36; i++)
        if (board[i % 6][i / 6] == SuraWhite)
          _features[36 + i] = 1;
    } else if (_status == GameStatus::player1Turn) {  // SuraBlack
      assert(_status == GameStatus::player1Turn);
      for (int i = 0; i < 72; i++)
        _features[i] = 0;
      for (int i = 0; i < 36; i++)
        if (board[i % 6][5 - (i / 6)] == SuraWhite)
          _features[i] = 1;
      for (int i = 0; i < 36; i++)
        if (board[i % 6][5 - (i / 6)] == SuraBlack)
          _features[36 + i] = 1;
    }
  }
  // The action just decreases the distance and swaps the turn to play.
  virtual void ApplyAction(const _Action& action) override {
    SKMove m;

    if (_status == GameStatus::player0Turn) {  // SuraWhite
      m.color = SuraWhite;
      m.x = action.GetY();
      m.y = action.GetZ();

      m.y1 = action.GetX() / SKDx;
      m.x1 = action.GetX() - m.y1 * SKDx;

      play(m);
      findActions(SuraBlack);
      if (won(SuraWhite) || _legalActions.size() == 0) {
        _legalActions.clear();
        _status = GameStatus::player0Win;
      } else if (is_draw()) {
        _status = GameStatus::tie;
        _legalActions.clear();
      } else
        _status = GameStatus::player1Turn;
    } else {
      // SuraBlack
      m.color = SuraBlack;
      m.x = action.GetY();
      m.y = action.GetZ();

      m.y1 = action.GetX() / SKDx;
      m.x1 = action.GetX() - m.y1 * SKDx;

      play(m);
      findActions(SuraWhite);
      if (won(SuraBlack) || _legalActions.size() == 0) {
        _legalActions.clear();
        _status = GameStatus::player1Win;
      } else if (is_draw()) {
        _legalActions.clear();
        _status = GameStatus::tie;
      } else
        _status = GameStatus::player0Turn;
    }
    findFeatures();
    _hash = hash;
    fillFullFeatures();
  }

  // For this trivial example we just compare to random play
  virtual void DoGoodAction() override {
    DoRandomAction();
  }
};
