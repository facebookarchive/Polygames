/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "mcts/state.h"
#include "mcts/types.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>

/*****
 Action and State are abstract classes.
 Derived classes correspond to various problems.

 A difference with the AZ setting  is that several actions can correspond to the
same logit
 from the neural net. This is useful for complex action spaces in which the list
of possible
 actions is tricky: the MCTS then takes care of differentiating the possible
actions.

 In most of our games, we are still in bijection, but in the case of draughts
this makes a difference.

******/

// TODO[doc]: ideally Action should just be private class known
// only to State. the Actor and State represent actions using
// the mcts::Action (i.e. int64_t)
class _Action {
 public:
  _Action() {
    _loc.resize(3);
  }
  virtual ~_Action() {
  }
  // Get the location of the move in the neural network output.
  // Several moves might fall in the same location, no pb.

  virtual int GetX() const {
    return _loc[0];
  }

  virtual int GetY() const {
    return _loc[1];
  }

  virtual int GetZ() const {
    return _loc[2];
  }

  uint64_t GetHash() const {
    return _hash;
  }

  void SetIndex(int i) {
    _i = i;
  }

  int GetIndex() const {
    return _i;
  }

 protected:
  uint64_t _hash;

  // Warning! Two actions might have the same position _loc.  position
  // of the action in {0,...,GetXActionSize()-1} *
  // {0,...,GetYActionSize()-1} * {0,...,GetZActionSize()-1}
  std::vector<int> _loc;

  // index of the action in the list of legal actions in the
  // corresponding state.
  // _i makes sense since an action is never applied to two distinct
  // states. We could have a pointer to the state this action is
  // associated to.
  mcts::Action _i;
};

std::ostream& operator<<(std::ostream& os, const _Action& action);

enum class GameStatus {
  player0Turn = 0,
  player1Turn,
  tie,
  player0Win,
  player1Win
};

class State : public mcts::State {
 public:
  void setSeed(int seed) {
    _rng.seed(seed);
  }
  State(int seed) {
    _rng.seed(seed);
    _stochasticReset = false;
    _hash = 0;
    _featSize.resize(3);
    _actionSize.resize(3);
    /*  _outFeatures = false;   // true if we want a feature for the frontier.
      _turnFeatures = false;  // true if we want a feature for the frontier.
      _history = 0;           // > 0 if we want to automatically add an history.
      _geometricFeatures = false;  // true if we want geometric features.
      _randomFeatures = 0;         // > 0 if we want random features.
      _oneFeature = false;         // true if we want a "plain 1" features.
      */
    _stochastic = false;
    forcedDice = -1;
  }

  virtual ~State() {
  }

  // -----overriding mcts::State's virtual functions-----

  virtual std::unique_ptr<mcts::State> clone() const final {
    auto s = clone_();
    auto state = dynamic_cast<State*>(s.get());
    state->_rng.seed(_rng());
    return s;
  }

  virtual int getCurrentPlayer() const final {
    if ((_status == GameStatus::player0Turn) ||
        (_status == GameStatus::player0Win)) {
      return 0;
    } else if ((_status == GameStatus::player1Turn) ||
               (_status == GameStatus::player1Win)) {
      return 1;
    } else {
      // assert(false);    do not assert this! there might be ties :-)
      return 0;  // the current player does not matter if we have ties.
    }
  }

  std::string history() {
    std::string str;
    auto sc = clone();
    auto* s = (State*)&*sc;
    auto moves = _moves;
    auto rngs = _moveRngs;
    s->reset();
    for (size_t i = 0; i != moves.size(); ++i) {
      // if (!str.empty()) str += " ";
      str += s->actionDescription(*s->GetLegalActions().at(moves.at(i)));
      std::tie(s->_rng, s->forcedDice) = rngs.at(i);
      s->forward(moves.at(i));
    }
    return str;
  }

  virtual int getCurrentPlayerColor() const {
    return getCurrentPlayer();
  }

  virtual int getStepIdx() const final {
    return _moves.size();
  }

  virtual float getReward(int player) const {
    assert(player == 0 || player == 1);
    if (_status == GameStatus::player0Win) {
      return player == 0 ? 1.0 : -1.0;
    } else if (_status == GameStatus::player1Win) {
      return player == 1 ? 1.0 : -1.0;
    } else {
      return 0.0;
    }
  }

  virtual bool terminated() const final {
    return (_status == GameStatus::tie || _status == GameStatus::player0Win ||
            _status == GameStatus::player1Win);
  };

  virtual float getRandomRolloutReward(int player) const final {
    const int numSimulation = 10;
    float sumReward = 0.0;
    for (int i = 0; i < numSimulation; ++i) {
      auto clonedState = clone();
      auto s = dynamic_cast<State*>(clonedState.get());
      while (!s->terminated()) {
        // TODO: random or good?
        s->DoRandomAction();
      }
      sumReward += s->getReward(player);
    }
    return sumReward / numSimulation;
  }

  virtual bool forward(const mcts::Action& action) final {
    // std::cerr << "forward" << std::endl;
    // {
    //   auto aa = *GetLegalActions()[action];
    //   std::cout << "forward: " << aa.GetX()
    //             << ", " << aa.GetY()
    //             << ", " << aa.GetZ() << std::endl;
    // }
    assert(action != mcts::InvalidAction);
    _moves.push_back(action);
    _moveRngs.emplace_back(_rng, forcedDice);
    ApplyAction(*GetLegalActions().at(action));

    // printCurrentBoard();
    // std::cout << "=========" << std::endl;

    return true;  // FIXME forward always return true ?
  }

  virtual bool operator==(const mcts::State& s) const final {
    const auto state = dynamic_cast<const State*>(&s);
    if (state == nullptr) {
      return false;
    }
    return _hash == state->_hash;
  }

  // -----interface for games to implement-----

  virtual void Initialize() = 0;

  virtual std::unique_ptr<mcts::State> clone_() const = 0;

  virtual void ApplyAction(const _Action& action) = 0;

  virtual void DoGoodAction() = 0;

  virtual void printCurrentBoard() const {
    std::cout << stateDescription() << std::endl;
  }

  virtual void errPrintCurrentBoard() const {
    std::cerr << stateDescription() << std::endl;
  }

  virtual const std::vector<std::shared_ptr<_Action>>& GetLegalActions() const {
    return _legalActions;
  }

  virtual std::string stateDescription() const {
    std::string str;
    auto& feats = GetFeatures();
    auto& sizes = GetFeatureSize();
    if (sizes[0] == 2) {
      bool allOnesOrZero = true;
      for (auto& v : feats) {
        if (v != 0 && v != 1) {
          allOnesOrZero = false;
          break;
        }
      }
      if (allOnesOrZero) {
        size_t index = 0;
        size_t offset = sizes[1] * sizes[2];
        for (int64_t y = 0; y != sizes[1]; ++y) {
          for (int64_t z = 0; z != sizes[2]; ++z) {
            if (z) {
              str += '|';
            }
            char c = ' ';
            if (feats[index] && feats[offset + index]) {
              c = '!';
            } else if (feats[index]) {
              c = 'x';
            } else if (feats[offset + index]) {
              c = 'o';
            }
            str += c;
            ++index;
          }
          str += '\n';
        }
        return str;
      }
    }
    size_t index = 0;
    for (int64_t x = 0; x != sizes[0]; ++x) {
      for (int64_t y = 0; y != sizes[1]; ++y) {
        for (int64_t z = 0; z != sizes[2]; ++z) {
          if (z) {
            str += ' ';
          }
          str += feats[index] == int(feats[index])
                     ? std::to_string(int(feats[index]))
                     : std::to_string(feats[index]);
          ++index;
        }
        str += '\n';
      }
      if (x != sizes[0] - 1) {
        str += '\n';
        str += '\n';
      }
    }
    return str;
  }

  virtual std::string actionDescription(const _Action& action) const {
    std::stringstream ss;
    ss << action;
    return ss.str();
  }

  virtual std::string actionsDescription() {
    std::stringstream ss;
    for (int i = 0; i < (int)_legalActions.size(); i++) {
      _Action& action = *(_legalActions[i]);
      ss << "Action " << i << ": " << actionDescription(action) << std::endl;
    }
    return ss.str();
  }

  virtual int parseAction(const std::string& str) {
    try {
      return std::stoul(str, nullptr, 10);
    } catch (...) {
      return -1;
    }
  }

  int TPInputAction(
      std::function<std::optional<int>(std::string)> specialAction =
          [](std::string) { return std::nullopt; }) {
    /*std::cout << "Current board:" << std::endl
              << stateDescription() << std::endl;
    std::cout << "Legal Actions:" << std::endl
              << actionsDescription() << std::endl;*/
    // Second, receive human feedback.
    std::string line1;
    std::string line2;
    std::string line3;
    auto& legalActions = GetLegalActions();
    int index = -1;
    int index1 = -1;
    int index2 = -1;
    int index3 = -1;
    std::cout << "# Last action" << std::endl;
    std::cerr << stateDescription() << std::endl;
    printLastActionXYZ();
    while (index < 0 || index >= (int)legalActions.size()) {
      std::cout << "#Input action as x y z: ";
      std::cin >> line1;
      std::cin >> line2;
      std::cin >> line3;
      index1 = parseAction(line1);
      index2 = parseAction(line2);
      index3 = parseAction(line3);
      for (size_t i = 0; i < legalActions.size(); i++) {
        if (((*GetLegalActions().at(i)).GetX() == index1) &&
            ((*GetLegalActions().at(i)).GetY() == index2) &&
            ((*GetLegalActions().at(i)).GetZ() == index3)) {
          index = i;
          break;
        }
        if (i == legalActions.size()) {
          std::cout << "# bad answer!" << std::endl;
        }
      }
      if (index == -1) {
        if (auto r = specialAction(line1); r) {
          return *r;
        }
      }
    }
    return index;
  }

  virtual int humanInputAction(
      std::function<std::optional<int>(std::string)> specialAction =
          [](std::string) { return std::nullopt; }) {
    std::cout << "Current board:" << std::endl
              << stateDescription() << std::endl;
    std::cout << "Legal Actions:" << std::endl
              << actionsDescription() << std::endl;
    // Second, receive human feedback.
    std::string line;
    auto& legalActions = GetLegalActions();
    int index = -1;
    while (index < 0 || index >= (int)legalActions.size()) {
      std::cout << "Input action: ";
      std::cin >> line;
      index = parseAction(line);
      if (index == -1) {
        if (auto r = specialAction(line); r) {
          return *r;
        }
      }
    }
    return index;
  }

  virtual void undoLastMove() {
    if (_moves.empty()) {
      return;
    }
    auto moves = _moves;
    auto rngs = _moveRngs;
    reset();
    for (size_t i = 0; i != moves.size() - 1; ++i) {
      std::tie(_rng, forcedDice) = rngs.at(i);
      forward(moves.at(i));
    }
  }


  virtual void setStateFromStr(const std::string& /*str*/) {
  }

  void printLastAction() {
    if (_moves.empty()) {
      std::cout << "no moves" << std::endl;
      return;
    }
    auto moves = _moves;
    auto rngs = _moveRngs;
    reset();
    for (size_t i = 0; i != moves.size(); ++i) {
      std::tie(_rng, forcedDice) = rngs.at(i);
      if (i == moves.size() - 1) {
        std::cout << actionDescription(*GetLegalActions().at(moves.at(i)))
                  << std::endl;
      }
      forward(moves.at(i));
    }
  }

  void printLastActionXYZ() {
    if (_moves.empty()) {
      std::cout << 0 << std::endl;
      std::cout << 0 << std::endl;
      std::cout << 0 << std::endl;
      return;
    }
    auto moves = _moves;
    auto rngs = _moveRngs;
    reset();
    for (size_t i = 0; i != moves.size(); ++i) {
      std::tie(_rng, forcedDice) = rngs.at(i);
      if (i == moves.size() - 1) {
        std::cout << (*GetLegalActions().at(moves.at(i))).GetX() << std::endl;
        std::cout << (*GetLegalActions().at(moves.at(i))).GetY() << std::endl;
        std::cout << (*GetLegalActions().at(moves.at(i))).GetZ() << std::endl;
      }
      forward(moves.at(i));
    }
  }

  virtual void undoLastMoveForPlayer(int player) {
    auto moves = _moves;
    auto rngs = _moveRngs;
    reset();
    size_t resetToIndex = moves.size();
    // Find the last move that was ours
    for (size_t i = 0; i != moves.size(); ++i) {
      std::tie(_rng, forcedDice) = rngs.at(i);
      auto prevPlayer = getCurrentPlayer();
      forward(moves[i]);
      if (prevPlayer == player) {
        resetToIndex = i;
      }
    }
    // Reset to it
    reset();
    for (size_t i = 0; i != resetToIndex; ++i) {
      std::tie(_rng, forcedDice) = rngs.at(i);
      forward(moves.at(i));
    }
    if (getCurrentPlayer() != player) {
      throw std::runtime_error("Undo error: expected player " +
                               std::to_string(player) + ", got " +
                               std::to_string(getCurrentPlayer()));
    }
  }

  // -----other non-virtual functions-----

  // only for stochastic games
  std::string performActionDescription(const _Action& action) {
    _status = (_status == GameStatus::player0Turn) ? GameStatus::player1Turn
                                                   : GameStatus::player0Turn;
    std::string ss = actionDescription(action);
    _status = (_status == GameStatus::player0Turn) ? GameStatus::player1Turn
                                                   : GameStatus::player0Turn;
    return ss;
  }

  void fillFullFeatures();

  void DoRandomAction() {
    assert(!_legalActions.empty());
    std::uniform_int_distribution<size_t> distr(0, _legalActions.size() - 1);
    size_t i = distr(_rng);
    _Action a = *(_legalActions[i].get());
    ApplyAction(a);
  }

  void doIndexedAction(int j) {
    int i = j % _legalActions.size();
    _Action a = *(_legalActions[i].get());
    ApplyAction(a);
  }

  bool checkMove(const mcts::Action& c) const {
    return c < (int)_legalActions.size();
  }

  uint64_t getHash() const final {
    return _hash;
  }

  const std::vector<float>& GetRawFeatures() const {
    return _features;
  }
  const std::vector<int64_t>& GetRawFeatureSize() const {
    return _featSize;
  }

  // Returns GetXSize x GetYSize x GetZSize float input for the NN.
  const std::vector<float>& GetFeatures() const {
    return _fullFeatures.empty() ? _features : _fullFeatures;
  }
  const std::vector<int64_t>& GetFeatureSize() const {
    return _outFeatSize.empty() ? _featSize : _outFeatSize;
  }

  int GetFeatureLength() const {
    auto featureSize = GetFeatureSize();
    return featureSize[0] * featureSize[1] * featureSize[2];
  }

  const std::vector<int64_t>& GetActionSize() const {
    return _actionSize;
  }

  void reset() {
    _moves.clear();
    _moveRngs.clear();
    _previousFeatures.clear();
    _previousFeaturesOffset = 0;
    _turnFeaturesOffset = 0;
    _outFeatSize.clear();
    _fullFeatures.clear();
    Initialize();
  }

  static void setFeatures(bool outFeatures,
                          bool turnFeatures,
                          bool geometricFeatures,
                          int history,
                          int randomFeatures,
                          bool oneFeature) {
    _outFeatures = outFeatures;
    _turnFeatures = turnFeatures;
    _geometricFeatures = geometricFeatures;
    _history = history;
    _randomFeatures = randomFeatures;
    _oneFeature = oneFeature;
  }

 protected:
  mutable std::minstd_rand _rng;

  GameStatus _status;
  uint64_t _hash;

  std::vector<float> _features;  // neural network input
  std::vector<std::shared_ptr<_Action>> _legalActions;
  std::vector<int64_t> _featSize;    // size of the neural network input
  std::vector<int64_t> _actionSize;  // size of the neural network output

  std::vector<mcts::Action> _moves;
  std::vector<std::pair<std::minstd_rand, int>> _moveRngs;

  // Data specifying the way we generate generic features.
  static bool _outFeatures;        // do we add a feature for borders
  static bool _turnFeatures;       // do we add a feature for turn
  static bool _geometricFeatures;  // do we add geometric features
  static int
      _history;  // do we add a feature for history and how long (0 = none)
  static int _randomFeatures;  // how many random features (could be 0)
  static bool _oneFeature;     // do we want a plane of 1s
  // if one of the values above is true or > 0 then we should call
  // fillFullFeatures at the end of the constructor of the derived class.

  // Below the std::vector involved in the generic added features.
  // size of the neural network input if using _outFeature or _history > 0:
  std::vector<int64_t> _outFeatSize;
  std::vector<float> _fullFeatures;      // neural network input, completed
  std::vector<float> _previousFeatures;  // history of features
  size_t _previousFeaturesOffset = 0;
  size_t _turnFeaturesOffset = 0;
};
