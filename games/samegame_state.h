/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "samegame.h"
#include "../core/state.h"

//#include <algorithm>
//#include <chrono>

namespace Samegame {

 class Action : public ::_Action {
  public:
   Action(int i, int j, int indexInActions, int nj);
 };

 class State : public ::State {
  private:
   Board _board;

  public:
   State(int seed);
   State(int seed, int history, bool turnFeatures);
   void findFeatures();
   void findActions();
   void Initialize() override;
   void ApplyAction(const _Action& action) override;
   void DoGoodAction() override;
   std::unique_ptr<mcts::State> clone_() const override;
   // std::string stateDescription() const override;
   // std::string actionDescription(const _Action & action) const override;
   // std::string actionsDescription() override;
   // int parseAction(const std::string& str) override;
   // virtual int getCurrentPlayerColor() const override;
 };

}  // namespace Samegame

