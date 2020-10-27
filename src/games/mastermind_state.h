/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

#include "../core/state.h"
// #include <boost/stacktrace.hpp>

/*****************************
  Mastermind with black pegs only.
  Each time steps decides one color.
  HORIZON is the number of rows that player 1 can fill for trying to find the
  target.
  ARITY is the number of colors.
  SIZE is the number of slots per row (and, equivalently, the number of colors
  to
  guess).
  The number of time steps is HORIZON * SIZE because SIZE is the number of
  colors
  to decide
  at each row.
                        */
namespace Mastermind {

template <int SIZE, int HORIZON, int ARITY> class State : public core::State {
 private:
  // just for enabling verbosity
  bool mmverbose;

  // representation
  int _board[HORIZON][SIZE];
  int _results[HORIZON];
  int _currentAction[SIZE];
  int real[SIZE];
  int _timeStep;

  // helper functions
  int mmhamming(int real[], int action[]);
  void rejection(int real[], int board[][SIZE], int results[], int time);
  bool unicity(int real[], int board[][SIZE], int results[], int time);

  // representation of the state
  virtual std::string stateDescription() const override {
    std::string res;
    res += "time";
    res += std::to_string(_timeStep);
    res += "  corresponding to line ";
    res += std::to_string(_timeStep / SIZE);
    res += " and slot ";
    res += std::to_string(_timeStep % SIZE);
    res += "\n";
    for (int i = 0; i < 1 + (_timeStep / SIZE); i++) {
      for (int j = 0; j < SIZE; j++) {
        if (i * SIZE + j < _timeStep) {
          res += " ";
          res += std::to_string(_board[i][j]);
        }
      }
      if (i * SIZE + SIZE - 1 < _timeStep) {
        res += " ==> score ";
        res += std::to_string(_results[i]);
        res += "     win:";
        res += std::to_string(_status == GameStatus::player0Win);
        res += "(";
        res += std::to_string(GetLegalActions().size());
        res += ")";
        res += "\n";
      }
    }
    return res;
  }

 public:
  virtual bool isOnePlayerGame() const override {
    return true;
  }
  State(int seed);
  void findActions();
  void Initialize() override;
  void ApplyAction(const _Action& action) override;
  void DoGoodAction() override;
  std::unique_ptr<core::State> clone_() const override;
};

// Hamming distance.
template <int SIZE, int HORIZON, int ARITY>
int State<SIZE, HORIZON, ARITY>::mmhamming(int real[], int action[]) {
  int result = SIZE;
  for (int i = 0; i < SIZE; i++) {
    if (real[i] != action[i]) {
      result--;
    }
  }
  return result;
}

// Rejection sampling -- we need better than that TODO(oteytaud).
template <int SIZE, int HORIZON, int ARITY>
void State<SIZE, HORIZON, ARITY>::rejection(int real[],
                                            int board[][SIZE],
                                            int results[],
                                            int time) {
  assert(time < HORIZON);
  std::uniform_int_distribution<int> distribution(0, ARITY - 1);
  if (mmverbose) {
    std::cerr << "rejection" << std::endl;
  }
  auto dice = std::bind(distribution, _rng);
  bool found = false;
  while (!found) {
    if (mmverbose) {
      std::cerr << " let us try..." << std::endl;
    }
    for (int i = 0; i < SIZE; i++) {
      real[i] = dice();
    }
    found = true;
    for (int j = 0; j < time; j++) {
      int localdistance = mmhamming(real, board[j]);
      if (localdistance != results[j]) {
        if (mmverbose) {
          std::cerr << "fail at time " << j << std::endl;
        }
        found = false;
        break;
      }
    }
    // if found, then it's ok we can proceed
  }
  if (mmverbose) {
    for (int i = 0; i < SIZE; i++)
      std::cout << real[i];
    std::cout << "\n";
  }
}  // end of rejection

// Checking if the solution is unique
template <int SIZE, int HORIZON, int ARITY>
bool State<SIZE, HORIZON, ARITY>::unicity(int real[],
                                          int board[][SIZE],
                                          int results[],
                                          int time) {
  int num_found = 0;
  assert(time < HORIZON);
  std::uniform_int_distribution<int> distribution(0, ARITY - 1);
  if (mmverbose) {
    std::cout << "unicity == == == == == == == == == == == = " << std::endl;
  }
  bool found = false;
  int index = 0;
  int maxIndex = 1;
  for (int i = 0; i < SIZE; i++)
    maxIndex *= ARITY;
  while (index < maxIndex) {
    if (mmverbose) {
      std::cout << " let us try ... ";
    }
    int tempoIndex = index;
    for (int i = 0; i < SIZE; i++) {
      real[i] = tempoIndex % ARITY;
      if (mmverbose) {
        std::cout << real[i];
      }
      tempoIndex /= ARITY;
    }
    found = true;
    if (mmverbose) {
      std::cout << std::endl;
    }
    for (int j = 0; j < time; j++) {
      int localdistance = mmhamming(real, board[j]);
      if (mmverbose) {
        std::cout << "distance " << localdistance << "/" << results[j]
                  << std::endl;
      }
      if (localdistance != results[j]) {
        if (mmverbose) {
          std::cout << "fail at time " << j << std::endl;
        }
        found = false;
      }
    }
    if (found) {
      num_found++;
      if (mmverbose) {
        std::cout << "success";
      }
      if (num_found > 1) {
        if (mmverbose) {
          std::cout << "several sols " << std::endl;
        }
        return false;
      }
    }
    index++;
  }
  if (num_found <= 0) {
    std::cout << "State with no solution : " << stateDescription() << std::endl;
  }
  assert(num_found == 1);
  return true;
}
// end of unicity
}  // namespace Mastermind

///////////////////////////////////////////////////////////////////////////////
// Mastermind::State
///////////////////////////////////////////////////////////////////////////////

template <int SIZE, int HORIZON, int ARITY>
Mastermind::State<SIZE, HORIZON, ARITY>::State(int seed)
    : core::State(seed) {
  mmverbose = false;
  if (mmverbose) {
    std::cerr << " cretion" << std::endl;
  }
  long s = std::chrono::system_clock::now().time_since_epoch().count();
  _rng.seed(s);
  Initialize();
  if (mmverbose) {
    std::cerr << " creation done" << std::endl;
  }
}

template <int SIZE, int HORIZON, int ARITY>
void Mastermind::State<SIZE, HORIZON, ARITY>::findActions() {
  clearActions();
  int time = _timeStep / SIZE;
  int slot = _timeStep % SIZE;
  if (_status == GameStatus::player0Turn) {
    assert(time < HORIZON);
    for (int i = 0; i < ARITY; i++) {
      addAction(i, time, slot);
    }
    if (mmverbose) {
      std::cerr << " fa done" << std::endl;
    }
  }
}

template <int SIZE, int HORIZON, int ARITY>
void Mastermind::State<SIZE, HORIZON, ARITY>::Initialize() {
  if (mmverbose) {
    std::cerr << " initialize" << std::endl;
  }
  _timeStep = 0;
  memset(_board, 0, sizeof(_board));
  memset(_currentAction, 0, sizeof(_currentAction));
  for (int i = 0; i < HORIZON; i++)
    _results[i] = -1;
  _hash = 0;
  _status = GameStatus::player0Turn;

  // features
  _featSize = {ARITY + 1, HORIZON, SIZE};
  _features.resize(_featSize[0] * _featSize[1] * _featSize[2]);
  for (int i = 0; i < (int)_features.size(); i++)
    _features[i] = 0.;
  _actionSize = {(ARITY + 1), (HORIZON), (SIZE)};
  if (mmverbose) {
    std::cerr << " init --> findactions " << std::endl;
  }
  findActions();
  if (mmverbose) {
    std::cerr << " init --> fff " << std::endl;
  }
  fillFullFeatures();
  _stochastic = false;
  if (mmverbose) {
    std::cerr << " init ok " << std::endl;
  }
}

template <int SIZE, int HORIZON, int ARITY>
void Mastermind::State<SIZE, HORIZON, ARITY>::ApplyAction(
    const _Action& action) {
  assert(_legalActions.size() > 0);
  assert(forcedDice <
         0);  // mastermind does not have a human mode for the moment.
  if (mmverbose) {
    std::cout << "before:\n" << stateDescription();
  }
  if (_legalActions.size() == 0) {
    // std::cout << boost::stacktrace::stacktrace();
    std::cout << "no legal action " << stateDescription() << std::endl;
    std::cout << "but playing " << actionDescription(action) << std::endl;
    std::cout << "wonstatus=" << (_status == GameStatus::player0Win)
              << std::endl;
    assert(_legalActions.size() > 0);
  }
  assert(_status == GameStatus::player0Turn);
  int time = _timeStep / SIZE;
  int slot = _timeStep % SIZE;

  if (mmverbose) {
    std::cerr << " timestep = " << _timeStep << std::endl;
    std::cerr << " slot=" << slot << "/" << SIZE << std::endl;
    std::cerr << " time=" << time << "/" << HORIZON << std::endl;
  }
  assert(time == action.GetY());
  assert(slot == action.GetZ());
  _currentAction[slot] = action.GetX();
  assert(time < HORIZON);
  assert(slot < SIZE);
  _board[time][slot] = action.GetX();
  assert(action.GetX() * HORIZON * SIZE + time * SIZE + slot <
         (int)_features.size());
  _features[action.GetX() * HORIZON * SIZE + time * SIZE + slot] = 1;
  if (slot == SIZE - 1) {
    rejection(real, _board, _results, time);
    if (mmverbose) {
      std::cerr << " now computing distance" << std::endl;
    }
    unsigned int distance = mmhamming(real, _currentAction);
    if (mmverbose) {
      std::cerr << "ARITY=" << ARITY << std::endl
                << "HORIZON=" << HORIZON << std::endl
                << "SIZE=" << SIZE << std::endl
                << "     AHS=" << ARITY * HORIZON * SIZE << std::endl
                << "time=" << time << std::endl
                << "     timeSIZE=" << time * SIZE << std::endl
                << "AHS+timeSIZE+SIZE="
                << ARITY * HORIZON * SIZE + time * SIZE + SIZE << std::endl
                << "versus " << _features.size() << std::endl;
    }
    {
      _results[time] = distance;
      assert(ARITY * HORIZON * SIZE + time * SIZE + SIZE - 1 <
             (int)_features.size());
      for (int i = 0; i < SIZE; i++) {
        _features[ARITY * HORIZON * SIZE + time * SIZE + i] =
            float(distance) / float(ARITY);
      }
    }
    _hash = distance;
    if (distance == SIZE) {
      if (mmverbose) {
        std::cout << " won by found at time " << time << std::endl;
      }
      _status = GameStatus::player0Win;
    } else if ((time < HORIZON - 1) &&
               (unicity(real, _board, _results, time))) {
      if (mmverbose) {
        std::cout << " won by unicity of solution and time=" << time << "<"
                  << HORIZON - 1 << std::endl;
        std::cerr << " won!" << std::endl;
      }
      _status = GameStatus::player0Win;
    } else if (time == HORIZON - 1) {
      if (mmverbose) {
        std::cerr << " lost!" << std::endl;
      }
      _status = GameStatus::player1Win;
    }
  }

  // TODO(oteytaud): cartesian product of actions would be better!
  _timeStep++;
  // if slot is SIZE-2 then the next step corresponds
  // to the last slot and therefore the next state is stochastic.
  if (slot == SIZE - 2) {
    if (mmverbose) {
      std::cerr << " go on" << std::endl;
    }
    _stochastic = true;
  } else {
    _stochastic = false;
  }

  // TODO(oteytaud): cartesian product of actions would be better!
  if (mmverbose) {
    std::cerr << " findactions " << std::endl;
  }
  findActions();
  if (mmverbose) {
    std::cerr << " fff " << std::endl;
  }
  fillFullFeatures();
  if (mmverbose) {
    std::cerr << "AA done" << std::endl;
  }
}

template <int SIZE, int HORIZON, int ARITY>
void Mastermind::State<SIZE, HORIZON, ARITY>::DoGoodAction() {
  if (mmverbose) {
    std::cerr << " do random action" << std::endl;
  }
  DoRandomAction();
}

template <int SIZE, int HORIZON, int ARITY>
std::unique_ptr<core::State> Mastermind::State<SIZE, HORIZON, ARITY>::clone_()
    const {
  return std::make_unique<Mastermind::State<SIZE, HORIZON, ARITY>>(*this);
}
