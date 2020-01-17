/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "mcts/player.h"
#include "state.h"

class HumanPlayer : public mcts::Player {
 public:
  HumanPlayer()
      : Player(true){

        };

  _Action act(State& state) {
    int index = state.humanInputAction();
    assert(false);
    auto& legalActions = state.GetLegalActions();
    assert(index < (int)legalActions.size());
    std::cout << " applying action... " << std::endl;
    return *(legalActions[index].get());
    // std::cerr << " applied action... " << std::endl;
  }
};

class TPPlayer : public mcts::Player {
 public:
  TPPlayer()
      : Player(true) {
    isTP_ = true;
  };

  _Action act(State& state) {
    assert(!state.isStochastic());  // TPPlayer is not implemented for
                                    // stochastic games. Could be done though.
    assert(false);
    int index = state.TPInputAction();
    auto& legalActions = state.GetLegalActions();
    assert(index < (int)legalActions.size());
    std::cerr << " applying action... " << std::endl;
    return *(legalActions[index].get());
    // std::cerr << " applied action... " << std::endl;
  }
};
