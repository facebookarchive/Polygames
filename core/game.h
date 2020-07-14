/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "mcts/mcts.h"
#include "tube/src_cpp/data_channel.h"
#include "tube/src_cpp/dispatcher.h"
#include "tube/src_cpp/env_thread.h"

#ifndef NO_JAVA
#include "../games/ludii/jni_utils.h"
#include "../games/ludii/ludii_state_wrapper.h"
#endif

#include "../games/amazons.h"
#include "../games/breakthrough_state.h"
#include "../games/chess.h"
#include "../games/chinesecheckers.h"
#include "../games/connect6_state.h"
#include "../games/connectfour.h"
#include "../games/diceshogi.h"
#include "../games/einstein.h"
#include "../games/havannah_state.h"
#include "../games/hex_state.h"
#include "../games/kyotoshogi_state.h"
#include "../games/mastermind_state.h"
#include "../games/minesweeper_state.h"
#include "../games/minishogi.h"
#include "../games/mnkgame.h"
// #include "../games/nogo_zestate.h"
#include "../games/block_go.h"
#include "../games/gomoku_swap2.h"
#include "../games/othello.h"
#include "../games/othello_opt.h"
#include "../games/outeropengomoku_new.h"
#include "../games/surakarta_state.h"
#include "../games/tristannogo_state.h"
#include "../games/weakschur/weakschur_state.h"
#include "../games/yinsh.h"
#include "human_player.h"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

//#define DEBUG_GAME

// Class for 2player fully observable game.
class Game : public tube::EnvThread {
 public:
  Game(std::string gameName,
       int numEpisode,
       int seed,
       bool evalMode,
       bool outFeatures,
       bool turnFeatures,
       bool geometricFeatures,
       int history,
       int randomFeatures,
       bool oneFeature,
       int perThreadBatchSize)
      : numEpisode(numEpisode)
      , evalMode(evalMode)
      , perThreadBatchSize(perThreadBatchSize)
      , result_(2, 0) {
    State::setFeatures(outFeatures, turnFeatures, geometricFeatures, history,
                       randomFeatures, oneFeature);
    gameName_ = gameName;
    if (isGameNameMatched({"Connect6"})) {
      state_ = std::make_unique<Connect6::StateForConnect6>(seed);
    } else if (isGameNameMatched({"Connect4"})) {
      state_ = std::make_unique<StateForConnectFour>(seed);
      /*
combinations with numcolors 2,6,8 and slots 4,5,6,10 are interesting. For time
horizon it is more difficult to guess. I’d go with 0.5*slots, 1.0 slots, 1.5
slots, 2 slots and then check numbers (just send us results in csv if easy to
share and we can suggest additional settings). If those experiments are too
long, you should start with colors=6, slots=4 and horizon \in {3,4,5}. With
horizon=5 you should get a winning proba of 1. This could be a reasonable check.
And we can check (by evaluating the decision tree - maybe we can find a student
to look into this) if the strategy is identical to knuth’s.
      */
      // Mastermind_<size>_<horizon>_<arity>
    } else if (isGameNameMatched({"Mastermind_4_5_6"})) {
      // should be winning proba 1
      state_ = std::make_unique<Mastermind::State<4, 5, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_6_6"})) {
      // should be winning proba 1
      state_ = std::make_unique<Mastermind::State<4, 6, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_7_6"})) {
      // should be winning proba 1
      state_ = std::make_unique<Mastermind::State<4, 7, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_3_6"})) {
      state_ = std::make_unique<Mastermind::State<4, 4, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_4_6"})) {
      state_ = std::make_unique<Mastermind::State<4, 3, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_5_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 5, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_6_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 6, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_7_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 7, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_8_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 8, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_9_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 9, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_10_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 10, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_15_2"})) {
      state_ = std::make_unique<Mastermind::State<10, 15, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind"})) {
      state_ = std::make_unique<Mastermind::State<3, 2, 2>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_4_4_4"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<4, 4, 4>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_3_1_1"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<3, 1, 1>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_5_2_3"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<5, 2, 3>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_5_5_10"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<5, 5, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_10_1_5"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<10, 1, 5>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_7_3_10"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<7, 3, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_5_5_15"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<5, 5, 15>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_8_8_10"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<8, 8, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_9_9_10"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<9, 9, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_16_16_40"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<16, 16, 40>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_30_16_99"})) {  // width, height, mines
      state_ = std::make_unique<Minesweeper::State<30, 16, 99>>(seed);
    } else if (isGameNameMatched({"TicTacToe", "NoughtsAndCrosses", "XsAndOs",
                                  "MNKGame_3_3_3"})) {
      state_ = std::make_unique<MNKGame::State<3, 3, 3>>(seed);
    } else if (isGameNameMatched(
                   {"FreeStyleGomoku", "GomokuFreeStyle", "MNKGame_15_15_5"})) {
      state_ = std::make_unique<MNKGame::State<15, 15, 5>>(seed);
    } else if (isGameNameMatched(
                   {"Othello4", "Reversi4", "Othello04", "Reversi04"})) {
      state_ = std::make_unique<Othello::State<6>>(seed);
    } else if (isGameNameMatched(
                   {"Othello6", "Reversi6", "Othello06", "Reversi06"})) {
      state_ = std::make_unique<Othello::State<6>>(seed);
    } else if (isGameNameMatched({"Othello8", "Reversi8", "Othello08",
                                  "Reversi08", "Othello", "Reversi"})) {
      state_ = std::make_unique<Othello::State<8>>(seed);
    } else if (isGameNameMatched({"Othello10", "Reversi10"})) {
      state_ = std::make_unique<Othello::State<10>>(seed);
    } else if (isGameNameMatched({"Othello12", "Reversi12"})) {
      state_ = std::make_unique<Othello::State<12>>(seed);
    } else if (isGameNameMatched({"Othello14", "Reversi14"})) {
      state_ = std::make_unique<Othello::State<14>>(seed);
    } else if (isGameNameMatched({"Othello16", "Reversi16"})) {
      state_ = std::make_unique<Othello::State<16>>(seed);
    } else if (isGameNameMatched({"OthelloOpt8", "OthelloOpt", "ReversiOpt8",
                                  "ReversiOpt"})) {
      state_ = std::make_unique<Othello2::State<8>>(seed);
    } else if (isGameNameMatched({"OthelloOpt10", "ReversiOpt10"})) {
      state_ = std::make_unique<Othello2::State<10>>(seed);
    } else if (isGameNameMatched({"OthelloOpt16", "ReversiOpt16"})) {
      state_ = std::make_unique<Othello2::State<16>>(seed);
    } else if (isGameNameMatched({"GameOfTheAmazons", "Amazons"})) {
      state_ = std::make_unique<Amazons::State>(seed);
    } else if (isGameNameMatched({"ChineseCheckers"})) {
      state_ = std::make_unique<ChineseCheckers::State>(seed);
    } else if (isGameNameMatched({"Hex5pie"})) {
      state_ = std::make_unique<Hex::State<5, true>>(seed);
    } else if (isGameNameMatched({"Hex11pie"})) {
      state_ = std::make_unique<Hex::State<11, true>>(seed);
    } else if (isGameNameMatched({"Hex13pie"})) {
      state_ = std::make_unique<Hex::State<13, true>>(seed);
    } else if (isGameNameMatched({"Hex19pie"})) {
      state_ = std::make_unique<Hex::State<19, true>>(seed);
    } else if (isGameNameMatched({"Hex5"})) {
      state_ = std::make_unique<Hex::State<5, false>>(seed);
    } else if (isGameNameMatched({"Hex11"})) {
      state_ = std::make_unique<Hex::State<11, false>>(seed);
    } else if (isGameNameMatched({"Hex13"})) {
      state_ = std::make_unique<Hex::State<13, false>>(seed);
    } else if (isGameNameMatched({"Hex19"})) {
      state_ = std::make_unique<Hex::State<19, false>>(seed);
    } else if (isGameNameMatched(
                   {"Havannah5pieExt"})) {  // ext = borders, corners
      state_ = std::make_unique<Havannah::State<5, true, true>>(seed);
    } else if (isGameNameMatched({"Havannah10pieExt"})) {
      state_ = std::make_unique<Havannah::State<10, true, true>>(seed);
    } else if (isGameNameMatched({"Havannah8pieExt"})) {
      state_ = std::make_unique<Havannah::State<8, true, true>>(seed);
    } else if (isGameNameMatched({"Havannah5pie"})) {
      state_ = std::make_unique<Havannah::State<5, true, false>>(seed);
    } else if (isGameNameMatched({"Havannah8pie"})) {
      state_ = std::make_unique<Havannah::State<8, true, false>>(seed);
    } else if (isGameNameMatched({"Havannah10pie"})) {
      state_ = std::make_unique<Havannah::State<10, true, false>>(seed);
    } else if (isGameNameMatched({"Havannah5"})) {
      state_ = std::make_unique<Havannah::State<5, false, false>>(seed);
    } else if (isGameNameMatched({"Havannah8"})) {
      state_ = std::make_unique<Havannah::State<8, false, false>>(seed);
    } else if (isGameNameMatched({"Havannah10"})) {
      state_ = std::make_unique<Havannah::State<10, false, false>>(seed);
    } else if (isGameNameMatched({"Breakthrough"})) {
      state_ = std::make_unique<StateForBreakthrough<false>>(seed);
    } else if (isGameNameMatched({"BreakthroughV2"})) {
      state_ = std::make_unique<StateForBreakthrough<true>>(seed);
    } else if (gameName.rfind("Ludii", 0) == 0) {
#ifdef NO_JAVA
      throw std::runtime_error(
          "Java/JNI support has not been built in, but is required for Ludii");
#else
      std::string ludii_name = gameName.substr(5);
      Ludii::JNIUtils::InitJVM("");  // Use default /ludii/Ludii.jar path
      JNIEnv* jni_env = Ludii::JNIUtils::GetEnv();
      Ludii::LudiiGameWrapper game_wrapper(jni_env, ludii_name);
      state_ = std::make_unique<Ludii::LudiiStateWrapper>(
          seed, jni_env, std::move(game_wrapper));
#endif
    } else if (isGameNameMatched({"Tristannogo"})) {
      state_ = std::make_unique<StateForTristannogo>(seed);
    } else if (isGameNameMatched({"OuterOpenGomoku", "OOGomoku"})) {
      state_ = std::make_unique<StateForOOGomoku>(seed);
    } else if (isGameNameMatched({"Minishogi"})) {
      state_ = std::make_unique<StateForMinishogi>(seed);
    } else if (isGameNameMatched({"Surakarta"})) {
      state_ = std::make_unique<StateForSurakarta>(seed);
    } else if (isGameNameMatched({"DiceShogi"})) {
      state_ = std::make_unique<StateForDiceshogi>(seed);
    } else if (isGameNameMatched({"BlockGo"})) {
      state_ = std::make_unique<StateForBlockGo>(seed);
    } else if (isGameNameMatched({"YINSH"})) {
      state_ = std::make_unique<StateForYinsh>(seed);
    } else if (isGameNameMatched({"GomokuSwap2", "Swap2Gomoku", "Gomoku"})) {
      state_ = std::make_unique<GomokuSwap2::State>(seed);
    } else if (isGameNameMatched({"KyotoShogi"})) {
      state_ = std::make_unique<StateForKyotoshogi>(seed);
    } else if (isGameNameMatched({"Einstein"})) {
      state_ = std::make_unique<StateForEinstein>(seed);
    } else if (isGameNameMatched({"WeakSchur_3_20"})) {  // subsets, maxNumber
      state_ = std::make_unique<weakschur::State<3, 20>>(seed);

    } else if (isGameNameMatched({"WeakSchur_4_66"})) {  // subsets, maxNumber
      state_ = std::make_unique<weakschur::State<4, 66>>(seed);
      // } else if (isGameNameMatched(gameName, {"Nogo"})) {
      //   state_ = std::make_unique<StateForNogo>();
    } else if (isGameNameMatched(
                   {"WeakSchur_5_197",
                    "WalkerSchur"})) {  // subsets, maxNumber  // is Walker
                                        // right ?   (1952! he said 197...)
      state_ = std::make_unique<weakschur::State<5, 197>>(seed);
    } else if (isGameNameMatched({"WeakSchur_3_70", "ImpossibleSchur"})) {
      state_ = std::make_unique<weakschur::State<3, 70>>(seed);
    } else if (isGameNameMatched(
                   {"WeakSchur_6_583",
                    "FabienSchur"})) {  // subsets, maxNumber  // beating F.
                                        // Teytaud et al
      state_ = std::make_unique<weakschur::State<6, 583>>(seed);
    } else if (isGameNameMatched({"WeakSchur_7_1737",
                                  "Arpad7Schur"})) {  // beating A. Rimmel et al
      state_ = std::make_unique<weakschur::State<7, 1737>>(seed);
    } else if (isGameNameMatched({"WeakSchur_8_5197",
                                  "Arpad8Schur"})) {  // beating A. Rimmel et al
      state_ = std::make_unique<weakschur::State<8, 5197>>(seed);
    } else if (isGameNameMatched({"WeakSchur_9_15315",
                                  "Arpad9Schur"})) {  // beating A. Rimmel et al
      state_ = std::make_unique<weakschur::State<9, 15315>>(seed);
    } else if (isGameNameMatched({"Chess"})) {
      state_ = std::make_unique<chess::State>(seed);
    } else {
      throw std::runtime_error("Unknown game name '" + gameName + "'");
    }
    // this is now useless thanks to the use of static variables above, but in
    // case we want to play:
    // state_->setFeatures(outFeatures, turnFeatures, geometricFeatures,
    // history, randomFeatures, oneFeature);

    state_->Initialize();
  }

  virtual bool isOnePlayerGame() const {
    return state_->isOnePlayerGame();
  }

  void setFeatures(bool outFeatures,
                   bool turnFeatures,
                   bool geometricFeatures,
                   int history,
                   int randomFeatures,
                   bool oneFeature) {
    state_->setFeatures(outFeatures, turnFeatures, geometricFeatures, history,
                        randomFeatures, oneFeature);
  }

  void addHumanPlayer(std::shared_ptr<HumanPlayer> player) {
    players_.push_back(std::move(player));
  }

  void addTPPlayer(std::shared_ptr<TPPlayer> player) {
    players_.push_back(std::move(player));
  }

  void addEvalPlayer(std::shared_ptr<mcts::MctsPlayer> player) {
    assert(evalMode);
    players_.push_back(std::move(player));
  }

  void addPlayer(std::shared_ptr<mcts::MctsPlayer> player,
                 std::shared_ptr<tube::DataChannel> dc) {
    assert(dc != nullptr && !evalMode);

    players_.push_back(std::move(player));

    auto feat = tube::EpisodicTrajectory(
        "s", state_->GetFeatureSize(), torch::kFloat32);
    auto pi = tube::EpisodicTrajectory(
        "pi", state_->GetActionSize(), torch::kFloat32);
    auto piMask = tube::EpisodicTrajectory(
        "pi_mask", state_->GetActionSize(), torch::kFloat32);
    auto v = tube::EpisodicTrajectory("v", {1}, torch::kFloat32);

    tube::Dispatcher dispatcher(std::move(dc));
    dispatcher.addDataBlocks(
        {feat.buffer, pi.buffer, piMask.buffer, v.buffer}, {});

    feature_.push_back(feat);
    pi_.push_back(pi);
    piMask_.push_back(piMask);
    v_.push_back(v);
    dispatchers_.push_back(dispatcher);
  }

  const std::vector<int64_t>& getRawFeatSize() {
    return state_->GetRawFeatureSize();
  }

  const std::vector<int64_t>& getFeatSize() {
    return state_->GetFeatureSize();
  }

  const std::vector<int64_t>& getActionSize() {
    return state_->GetActionSize();
  }

  virtual void mainLoop() override;

  std::vector<float> getResult() {
    return result_;
  }

  virtual void terminate() override {
#ifdef DEBUG_GAME
    std::cout << "game " << this << ", setting terminating flag" << std::endl;
#endif
    EnvThread::terminate();
// std::unique_lock<std::mutex> lk(terminateMutex_);
#ifdef DEBUG_GAME
    std::cout << "game " << this << ", terminating dispatchers" << std::endl;
#endif
    for (auto& v : dispatchers_) {
      v.terminate();
    }
#ifdef DEBUG_GAME
    std::cout << "game " << this << ", terminating players" << std::endl;
#endif
    for (auto& v : players_) {
      v->terminate();
    }
  }

  virtual EnvThread::Stats get_stats() override;

  const int numEpisode;
  // const int threadIdx;
  const bool evalMode;
  const int perThreadBatchSize;
  // const bool humanModeFirst;
  // const bool humanModeSecond;
  // const bool strong_baseline;        // = true;
  // const bool use_also_mcts_in_eval;  // = true;

 private:
  bool isGameNameMatched(const std::vector<std::string>&& allowedNames) {
    auto strToLower = [&](const std::string& str) {
      std::string s = std::string(str);
      std::transform(s.begin(), s.end(), s.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      return s;
    };

    std::string nameLower = strToLower(gameName_);
    for (auto& allowedName : allowedNames) {
      if (nameLower == strToLower(allowedName))
        return true;
    }
    return false;
  }

  void reset() {
    state_->reset();
    for (auto& player : players_) {
      player->newEpisode();
    }
  }

  std::optional<int> parseSpecialAction(const std::string& str);

  void step();

  void setReward(const State& state, int resigned = -1);

  void sendTrajectory();

  bool prepareForSend(int playerId);

  std::unique_ptr<State> state_;
  std::vector<std::shared_ptr<mcts::Player>> players_;

  std::vector<tube::EpisodicTrajectory> feature_;
  std::vector<tube::EpisodicTrajectory> pi_;
  std::vector<tube::EpisodicTrajectory> piMask_;
  std::vector<tube::EpisodicTrajectory> v_;

  std::vector<tube::Dispatcher> dispatchers_;

  std::vector<float> result_;

  std::mutex mutexStats_;
  EnvThread::Stats stats_;

  std::string lastAction_;
  bool hasPrintedHumanHelp_ = false;
  bool isInSingleMoveMode_ = false;
  float lastMctsValue_ = 0.0f;

  std::string gameName_;
  std::minstd_rand rng_{std::random_device{}()};
  bool canResign_ = false;
  std::vector<int> resignCounter_;
  int resigned_ = -1;
};
