/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "game.h"
#include "state.h"
#include <iostream>

float goodEval(State& s) {
  float numWins = 0;
  int gameCount = 0;
  while (gameCount < 100) {
    s.reset();
    while (!s.terminated()) {
      s.DoGoodAction();
    }
    numWins += 0.5 * (1 + s.getReward(0));
    ++gameCount;
  }
  float winRate = numWins / float(gameCount);
  std::cout << "good win rate = " << winRate << std::endl;
  if ((winRate <= 0.01) || (winRate >= 0.99)) {
    throw std::runtime_error(
        "this game has a random win rate beyond acceptable.");
  }
  return true;
}

float randEval(State& s) {
  float numWins = 0;
  int gameCount = 0;
  while (gameCount < 100) {
    s.reset();
    while (!s.terminated()) {
      s.DoRandomAction();
    }
    numWins += 0.5 * (1 + s.getReward(0));
    ++gameCount;
  }
  float winRate = numWins / float(gameCount);
  std::cout << "win rate = " << winRate << std::endl;
  if ((winRate <= 0.01) || (winRate >= 0.99)) {
    throw std::runtime_error(
        "this game has a random win rate beyond acceptable.");
  }
  return true;
}

int doSimpleTest(State& s) {
  // goodEval(s);
  // Test that everything is fine.
  // win_frequency = 0 or 1 in purely random play is weird.
  randEval(s);

  // Now testing if the game looks stochastic.
  bool isStochastic = false;
  // We will check this for various lengths of simulations, i is the length.
  // if isStochastic switches to true (i.e. a non-determinism is already
  // detected),
  // then we stop the loop.
  bool theoreticallyStochastic = s.isStochastic();
  for (int umax = 8; ((umax < 70) && (!isStochastic)); umax += 1) {
    s.Initialize();
    s.setSeed(5678);
    if (s.isStochastic()) {
      theoreticallyStochastic = true;
    }
    for (int u = 0; u < umax; u++) {
      if (!s.terminated())
        s.doIndexedAction(int(umax * 7.123 + u * 1.35));
      if (s.isStochastic()) {
        theoreticallyStochastic = true;
      }
      // s.stateDescription();
      // std::cout << "old:" << u << ":" << s.GetFeatures() << std::endl;
    }
    // std::cerr << s.stateDescription() << std::endl;
    auto oldFeatures = s.GetFeatures();
    // we play another game of length u.
    s.Initialize();
    s.setSeed(1234);
    for (int u = 0; u < umax; u++) {
      if (!s.terminated())
        s.doIndexedAction(int(umax * 7.123 + u * 1.35));
      // std::cout << "=====" << std::endl << s.stateDescription() << std::endl;
      // std::cout << "new:" << u << ":" << s.GetFeatures() << std::endl;
    }
    // std::cerr << s.stateDescription() << std::endl;
    if ((int)s.GetFeatures().size() != s.GetFeatureLength()) {
      throw std::runtime_error("wrong feature length");
    }
    for (int j = 0; ((!isStochastic) && (j < s.GetFeatureLength())); j++) {
      if (s.GetFeatures()[j] != oldFeatures[j]) {
        std::cout << "#horizon" << umax << "+feature" << j << "/"
                  << s.GetFeatureLength() << "--" << s.GetFeatures()[j]
                  << " vs " << oldFeatures[j] << std::endl;
        isStochastic = true;
      }
    }
    // if (isStochastic && (!theoreticallyStochastic)) {
    //   std::cout << "original:" << oldFeatures << std::endl;
    //   std::cout << "current: " << s.GetFeatures() << std::endl;
    // }
  }
  if (isStochastic != theoreticallyStochastic) {
    std::cout << s.stateDescription() << std::endl;
    std::cout << " Theoretically: " << theoreticallyStochastic << std::endl;
    std::cout << " Practically: " << isStochastic << std::endl;
    throw std::runtime_error("stochasticity violated");
  }
  // s.Initialize();
  return s.GetFeatureSize()[0];
}

void doTest(State& s) {
  doSimpleTest(s);
  std::cout << "testing: fillFullFeatures at the end of ApplyAction and of "
               "Initialize."
            << std::endl;
  s.setFeatures(false, false, false, 0, 3, false);
  doSimpleTest(s);
}

// TODO: there should be better way (using gtest?) than writing a main
// this is just for demo purpose
// After compilation, run test_state from build folder to run the test
int main() {
  int seed = 999;

  {
    std::cout << "testing: tristannogo" << std::endl;
    auto state = StateForTristannogo(seed);
    doTest(state);
    std::cout << "test pass: tristannogo" << std::endl;
  }

  {
    std::cout << "testing: BlockGo" << std::endl;
    auto state = StateForBlockGo(seed);
    doTest(state);
    std::cout << "test pass: BlockGo" << std::endl;
  }
  {
#ifdef NO_JAVA
    std::cout << "skipping: Ludii Tic-Tac-Toe" << std::endl;
#else
    std::cout << "testing: Ludii Tic-Tac-Toe" << std::endl;
    Ludii::JNIUtils::InitJVM("");  // Use default /ludii/Ludii.jar path
    JNIEnv* jni_env = Ludii::JNIUtils::GetEnv();

    if (jni_env) {
      Ludii::LudiiGameWrapper game_wrapper("Tic-Tac-Toe.lud");
      auto state = std::make_unique<Ludii::LudiiStateWrapper>(
          seed, std::move(game_wrapper));
      doTest(*state);
      Ludii::JNIUtils::CloseJVM();
      std::cout << "test pass: Ludii Tic-Tac-Toe" << std::endl;
    } else {
      std::cout << "skipping: Ludii Tic-Tac-Toe" << std::endl;
    }
#endif
  }

  {
    std::cout << "testing: connect four" << std::endl;
    auto state = StateForConnectFour(seed);
    doTest(state);
    std::cout << "test pass: connect four" << std::endl;
  }

  {
    std::cout << "testing: breakthrough" << std::endl;
    auto state = StateForBreakthrough(seed);
    doTest(state);
    std::cout << "test pass: breakthrough" << std::endl;
  }

  {
    std::cout << "testing: Connect6" << std::endl;
    auto state = Connect6::StateForConnect6(seed);
    doTest(state);
    std::cout << "test pass: Connect6" << std::endl;
  }

  {
    std::cout << "testing: Tic-tac-toe" << std::endl;
    auto state = MNKGame::State<3, 3, 3>(seed);
    doTest(state);
    std::cout << "test pass: Tic-tac-toe" << std::endl;
  }

  {
    std::cout << "testing: Free-style gomoku" << std::endl;
    auto state = MNKGame::State<15, 15, 5>(seed);
    doTest(state);
    std::cout << "test pass: Free-style gomoku" << std::endl;
  }

  {
    std::cout << "testing: Othello" << std::endl;
    auto state8 = Othello::State<8>(seed);
    doTest(state8);
    std::cout << "test pass: 8×8 Othello" << std::endl;
    auto state10 = Othello::State<10>(seed);
    doTest(state10);
    std::cout << "test pass: 10×10 Othello" << std::endl;
    auto state16 = Othello::State<16>(seed);
    doTest(state16);
    std::cout << "test pass: 16×16 Othello" << std::endl;
  }

  {
    std::cout << "testing: Game of the Amazons" << std::endl;
    auto state = Amazons::State(seed);
    doTest(state);
    std::cout << "test pass: Game of the Amazons" << std::endl;
  }

  {
    std::cout << "testing: Chinese Checkers" << std::endl;
    auto state = ChineseCheckers::State(seed);
    doTest(state);
    std::cout << "test pass: Chinese Checkers" << std::endl;
  }

  {
    std::cout << "testing: Gomoku swap2" << std::endl;
    auto state = GomokuSwap2::State(seed);
    doTest(state);
    std::cout << "test pass: Gomoku swap2" << std::endl;
  }

  {
    std::cout << "testing: hex5pie" << std::endl;
    auto state = Hex::State<5, true>(seed);
    doTest(state);
    std::cout << "test pass: hex5pie" << std::endl;
  }

  {
    std::cout << "testing: hex11pie" << std::endl;
    auto state = Hex::State<11, true>(seed);
    doTest(state);
    std::cout << "test pass: hex11pie" << std::endl;
  }

  {
    std::cout << "testing: hex13pie" << std::endl;
    auto state = Hex::State<13, true>(seed);
    doTest(state);
    std::cout << "test pass: hex13pie" << std::endl;
  }

  {
    std::cout << "testing: hex19pie" << std::endl;
    auto state = Hex::State<19, true>(seed);
    doTest(state);
    std::cout << "test pass: hex19pie" << std::endl;
  }

  {
    std::cout << "testing: hex5" << std::endl;
    auto state = Hex::State<5, false>(seed);
    doTest(state);
    std::cout << "test pass: hex5" << std::endl;
  }

  {
    std::cout << "testing: hex11" << std::endl;
    auto state = Hex::State<11, false>(seed);
    doTest(state);
    std::cout << "test pass: hex11" << std::endl;
  }

  {
    std::cout << "testing: hex13" << std::endl;
    auto state = Hex::State<13, false>(seed);
    doTest(state);
    std::cout << "test pass: hex13" << std::endl;
  }

  {
    std::cout << "testing: hex19" << std::endl;
    auto state = Hex::State<19, false>(seed);
    doTest(state);
    std::cout << "test pass: hex19" << std::endl;
  }

  {
    std::cout << "testing: havannah5pieExt" << std::endl;
    auto state = Havannah::State<5, true, true>(seed);
    doTest(state);
    std::cout << "test pass: havannah5pieExt" << std::endl;
  }

  {
    std::cout << "testing: havannah8pieExt" << std::endl;
    auto state = Havannah::State<8, true, true>(seed);
    doTest(state);
    std::cout << "test pass: havannah8pieExt" << std::endl;
  }

  {
    std::cout << "testing: havannah5pie" << std::endl;
    auto state = Havannah::State<5, true, false>(seed);
    doTest(state);
    std::cout << "test pass: havannah5pie" << std::endl;
  }

  {
    std::cout << "testing: havannah8pie" << std::endl;
    auto state = Havannah::State<8, true, false>(seed);
    doTest(state);
    std::cout << "test pass: havannah8pie" << std::endl;
  }

  {
    std::cout << "testing: havannah5" << std::endl;
    auto state = Havannah::State<5, false, false>(seed);
    doTest(state);
    std::cout << "test pass: havannah5" << std::endl;
  }

  {
    std::cout << "testing: havannah8" << std::endl;
    auto state = Havannah::State<8, false, false>(seed);
    doTest(state);
    std::cout << "test pass: havannah8" << std::endl;
  }

  {
    std::cout << "testing: Outer Open Gomoku" << std::endl;
    auto state = StateForOOGomoku(seed);
    doTest(state);
    std::cout << "test pass: Outer Open Gomoku" << std::endl;
  }

  {
    std::cout << "testing: Mastermind" << std::endl;
    auto state = Mastermind::State<10, 7, 2>(seed);
    doTest(state);
    std::cout << "test pass: Mastermind" << std::endl;
  }
  {
    std::cout << "testing: Minesweeper beginner" << std::endl;
    auto state = Minesweeper::State<8, 8, 10>(seed);
    doTest(state);
    std::cout << "test pass: Minesweeper beginner" << std::endl;
  }

  /* win rates for intermediate and expert are too low
     when taking random actions
  {
    std::cout << "testing: Minesweeper intermediate" << std::endl;
    auto state = Minesweeper::State<15, 13, 40>(seed);
    doTest(state);
    std::cout << "test pass: Minesweeper intermediate" << std::endl;
  }

  {
    std::cout << "testing: Minesweeper expert" << std::endl;
    auto state = Minesweeper::State<30, 16, 99>(seed);
    doTest(state);
    std::cout << "test pass: Minesweeper expert" << std::endl;
  }
  */

  {
    std::cout << "testing: Outer Open Gomoku" << std::endl;
    auto state = StateForOOGomoku(seed);
    doTest(state);
    std::cout << "test pass: Outer Open Gomoku" << std::endl;
  }

  {
    std::cout << "testing: Surakarta" << std::endl;
    auto state = StateForSurakarta(seed);
    doTest(state);
    std::cout << "test pass: Surakarta" << std::endl;
  }

  {
    std::cout << "testing: Einstein" << std::endl;
    auto state = StateForEinstein(seed);
    doTest(state);
    std::cout << "test pass: Einstein" << std::endl;
  }

  {
    std::cout << "testing: Minishogi" << std::endl;
    auto state = StateForMinishogi(seed);
    doTest(state);
    std::cout << "test pass: Minishogi" << std::endl;
  }

  {
    std::cout << "testing: Diceshogi" << std::endl;
    auto state = StateForDiceshogi(seed);
    doTest(state);
    std::cout << "test pass: Diceshogi" << std::endl;
  }

  {
    std::cout << "testing: YINSH" << std::endl;
    auto state = StateForYinsh(seed);
    doTest(state);
    std::cout << "test pass: YINSH" << std::endl;
  }

  {
    std::cout << "testing: Kyotoshogi" << std::endl;
    auto state = StateForKyotoshogi(seed);
    doTest(state);
    std::cout << "test pass: Kyotoshogi" << std::endl;
  }
}
