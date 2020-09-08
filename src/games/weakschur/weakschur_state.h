/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../../core/state.h"
#include "WeakSchur.hpp"
// #include <boost/stacktrace.hpp> // TODO #ifdef
#include <sstream>

namespace weakschur {

  template <int NBSUBSETS, int MAXNUMBER>
  class Action : public ::_Action {
    public:
      Action(int subset, int number, int indexInActions);
      std::string to_string() const;
  };

  template <int NBSUBSETS, int MAXNUMBER>
  class State : public core::State {
    private:
      WeakSchur _weakschur;

    public:
      State(int seed);
      bool isOnePlayerGame() const override;
      void Initialize() override;
      void ApplyAction(const _Action& action) override;
      void DoGoodAction() override;
      float getReward(int player) const override final {
	//if (player != 0)
	//	std::cout << boost::stacktrace::stacktrace();
        if (_weakschur.getScore() == MAXNUMBER) {
          std::cout << "Found Good Schur:" << _weakschur << std::endl;
          std::cerr << "Found Good Schur:" << _weakschur << std::endl;
          abort();
        }
        float value = float(_weakschur.getScore()) / float(MAXNUMBER);
        return player == 0 ? value : -value;
      };
      std::unique_ptr<core::State> clone_() const override;

    private:
      std::string stateDescription() const override;
      void findActions();
  };

}  // namespace weakschur

///////////////////////////////////////////////////////////////////////////////
// weakschur::Action
///////////////////////////////////////////////////////////////////////////////

template <int NBSUBSETS, int MAXNUMBER>
weakschur::Action<NBSUBSETS, MAXNUMBER>::Action(int subset, int number, int indexInActions) {
  _loc[0] = 0;
  _loc[1] = subset;
  _loc[2] = number;
  _hash = uint32_t(subset * (MAXNUMBER) + number);
  _i = indexInActions;
}

template <int NBSUBSETS, int MAXNUMBER>
std::string weakschur::Action<NBSUBSETS, MAXNUMBER>::to_string() const {
  return "(subset=" + std::to_string(_loc[1]) 
       + ", number=" + std::to_string(_loc[2]) + ")";
}

///////////////////////////////////////////////////////////////////////////////
// weakschur::State
///////////////////////////////////////////////////////////////////////////////

template <int NBSUBSETS, int MAXNUMBER>
weakschur::State<NBSUBSETS, MAXNUMBER>::State(int seed) :
  core::State(seed),
  _weakschur(NBSUBSETS, MAXNUMBER)
{
  Initialize();
}

template <int NBSUBSETS, int MAXNUMBER>
bool weakschur::State<NBSUBSETS, MAXNUMBER>::isOnePlayerGame() const {
  return true;
}

template <int NBSUBSETS, int MAXNUMBER>
void weakschur::State<NBSUBSETS, MAXNUMBER>::DoGoodAction() {
  DoRandomAction();
}

template <int NBSUBSETS, int MAXNUMBER>
std::unique_ptr<core::State> weakschur::State<NBSUBSETS, MAXNUMBER>::clone_() const {
  return std::make_unique<weakschur::State<NBSUBSETS, MAXNUMBER>>(*this);
}

template <int NBSUBSETS, int MAXNUMBER>
std::string weakschur::State<NBSUBSETS, MAXNUMBER>::stateDescription() const {
  std::ostringstream oss;
  oss << _weakschur;
  return oss.str();
}

template <int NBSUBSETS, int MAXNUMBER>
void weakschur::State<NBSUBSETS, MAXNUMBER>::Initialize() {

  // _weakschur
  _weakschur.reset();

  // state
  _hash = 0;
  _status = GameStatus::player0Turn;

  // features
  // TODO channels
  _featSize = {9, NBSUBSETS, MAXNUMBER};
  _features = std::vector<float>(_featSize[0] * _featSize[1] * _featSize[2], 0.f);
  _features[0] = 1.f;  // _weakschur always does the first action {1, 1}

  const int channelSize = NBSUBSETS*MAXNUMBER;

  // 1 features: i / first(t)
  // 2 features: longest seq(t) / maxnumber
  // 3 features: #longest

  // 4 features: #possible for i / nbsubsets
  for (int i=0; i<channelSize; i++) 
      _features[channelSize*4+i] = 1.f;
  // 5 features: #possible for t / maxnumber
  for (int i=0; i<channelSize; i++) 
      _features[channelSize*5+i] = (MAXNUMBER-1) / float(MAXNUMBER);

  // 6 features: board (t, i-1)
  // 7 features: board (t, i-2)
  // 8 features: board (t, i-3)
  fillFullFeatures();


  // actions
  _actionSize = {1, NBSUBSETS+1, MAXNUMBER+1};
  findActions();
}

template <int NBSUBSETS, int MAXNUMBER>
void weakschur::State<NBSUBSETS, MAXNUMBER>::ApplyAction(const _Action& action) {

  const int channelSize = NBSUBSETS*MAXNUMBER;

  auto feature = [channelSize,this] (int c, int subset, int number) -> float&{
      return this->_features[channelSize*c + (subset-1)*MAXNUMBER + number - 1];
  };

  // update weakschur
  assert(not _weakschur.isTerminated());
  int subset = action.GetY();
  int number = action.GetZ();
  _weakschur.applyAction({subset, number});
  
  // update status
  if (_weakschur.isTerminated()) {
    // std::cout << "WS:" << _weakschur.getScore() << "," << MAXNUMBER << std::endl;
    _status = _weakschur.getScore() == MAXNUMBER ? GameStatus::player0Win
                                                 : GameStatus::player1Win;
  }

  // update state
  int k = (subset-1)*MAXNUMBER + (number-1);
  assert(_features[k] == 0.f);
  _features[k] = 1.f;

  // 1 features: i / first(t)
  int firstT = MAXNUMBER+1;
  for (int n=1; n<=MAXNUMBER; n++) {
      if (_weakschur._subsetOfNumber.get(n) == subset) {
          firstT = n;
          break;
      }
  } 
  for (int n=1; n<=MAXNUMBER; n++) {
      feature(1, subset, n) = n / float(firstT);
  }

  // 2 features: longest seq(t) / maxnumber
  // 3 features: #longest
  auto longestAndNb = _weakschur.getLongestSeq(subset);
  for (int n=1; n<=MAXNUMBER; n++) {
      feature(2, subset, n) = longestAndNb.first / float(MAXNUMBER);
      feature(3, subset, n) = longestAndNb.second / float(MAXNUMBER);
  }

  // 4 features: #possible for i / nbsubsets
  for (int n=1; n<=MAXNUMBER; n++)
	if (_weakschur._subsetOfNumber.get(n) != 0)
		feature(4, subset, n) = _weakschur.getLegalSubsets(n).size() / float(NBSUBSETS);
  	else
		feature(4, subset, n) = 0.f;


  // 5 features: #possible for t / maxnumber
  for (int n=1; n<=MAXNUMBER; n++)
      for (int s=1; s<=NBSUBSETS; s++)
          feature(5, s, n) = _weakschur._nbFreeNumbersOfSubset.get(s) / float(MAXNUMBER);

  // 6 features: board (t, i-1)
  for (int n=2; n<=MAXNUMBER; n++)
      for (int s=1; s<=NBSUBSETS; s++)
          feature(6, s, n) = _weakschur._subsetOfNumber.get(n-1) == s ? 1.f : 0.f;
 
  // 7 features: board (t, i-2)
  for (int n=3; n<=MAXNUMBER; n++)
      for (int s=1; s<=NBSUBSETS; s++)
          feature(7, s, n) = _weakschur._subsetOfNumber.get(n-2) == s ? 1.f : 0.f;

  // 8 features: board (t, i-3)
  for (int n=4; n<=MAXNUMBER; n++)
      for (int s=1; s<=NBSUBSETS; s++)
          feature(8, s, n) = _weakschur._subsetOfNumber.get(n-3) == s ? 1.f : 0.f;

  fillFullFeatures();

  // update actions
  findActions();

}

template <int NBSUBSETS, int MAXNUMBER>
void weakschur::State<NBSUBSETS, MAXNUMBER>::findActions() {
  // TODO multiple "most contrained" numbers ?

  _legalActions.clear();
  if (not _weakschur.isTerminated()) {

    // get possible numbers
    int number1 = _weakschur.getFirstLegalNumber();
    int number2 = _weakschur.getMostConstrainedNumber();

    // get subsets and update actions
    if (number1 == number2) {
      std::vector<int> subsets1 = _weakschur.getLegalSubsets(number1);
      _legalActions.reserve(subsets1.size());
      int index = 0;
      for (unsigned k=0; k<subsets1.size(); ++k, ++index) 
        _legalActions.push_back(std::make_shared<weakschur::Action<NBSUBSETS, MAXNUMBER>>(subsets1[k], number1, index));
    }
    else {
      std::vector<int> subsets1 = _weakschur.getLegalSubsets(number1);
      std::vector<int> subsets2 = _weakschur.getLegalSubsets(number2);
      _legalActions.reserve(subsets1.size() + subsets2.size());
      int index = 0;
      for (unsigned k=0; k<subsets1.size(); ++k, ++index) 
        _legalActions.push_back(std::make_shared<weakschur::Action<NBSUBSETS, MAXNUMBER>>(subsets1[k], number1, index));
      for (unsigned k=0; k<subsets2.size(); ++k, ++index) 
        _legalActions.push_back(std::make_shared<weakschur::Action<NBSUBSETS, MAXNUMBER>>(subsets2[k], number2, index));
    }

  }
}

