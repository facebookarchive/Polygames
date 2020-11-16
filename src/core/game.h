/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "forward_player.h"
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

namespace core {

// Class for 2player fully observable game.
class Game : public tube::EnvThread {
  friend struct BatchExecutor;

 public:
  Game(std::string gameName,
       std::vector<std::string> gameOptions,
       int numEpisode,
       int seed,
       bool evalMode,
       bool outFeatures,
       bool turnFeaturesSingleChannel,
       bool turnFeaturesMultiChannel,
       bool geometricFeatures,
       int history,
       int randomFeatures,
       bool oneFeature,
       int perThreadBatchSize,
       int maxRewinds,
       bool predictEndState,
       int predictNStates)
      : numEpisode(numEpisode)
      , evalMode(evalMode)
      , perThreadBatchSize(perThreadBatchSize)
      , maxRewinds(maxRewinds)
      , predictEndState(predictEndState)
      , predictNStates(predictNStates)
      , result_(2, 0) {
    gameName_ = gameName;
    if (isGameNameMatched({"Connect6"})) {
      state_ = newState<Connect6::StateForConnect6<1>>(seed);
    } else if (isGameNameMatched({"Connect6v2"})) {
      state_ = newState<Connect6::StateForConnect6<2>>(seed);
    } else if (isGameNameMatched({"Connect4"})) {
      state_ = newState<StateForConnectFour>(seed);
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
      state_ = newState<Mastermind::State<4, 5, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_6_6"})) {
      // should be winning proba 1
      state_ = newState<Mastermind::State<4, 6, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_7_6"})) {
      // should be winning proba 1
      state_ = newState<Mastermind::State<4, 7, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_3_6"})) {
      state_ = newState<Mastermind::State<4, 4, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_4_4_6"})) {
      state_ = newState<Mastermind::State<4, 3, 6>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_5_2"})) {
      state_ = newState<Mastermind::State<10, 5, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_6_2"})) {
      state_ = newState<Mastermind::State<10, 6, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_7_2"})) {
      state_ = newState<Mastermind::State<10, 7, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_8_2"})) {
      state_ = newState<Mastermind::State<10, 8, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_9_2"})) {
      state_ = newState<Mastermind::State<10, 9, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_10_2"})) {
      state_ = newState<Mastermind::State<10, 10, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind_10_15_2"})) {
      state_ = newState<Mastermind::State<10, 15, 2>>(seed);
    } else if (isGameNameMatched({"Mastermind"})) {
      state_ = newState<Mastermind::State<3, 2, 2>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_4_4_4"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<4, 4, 4>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_3_1_1"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<3, 1, 1>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_5_2_3"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<5, 2, 3>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_5_5_10"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<5, 5, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_10_1_5"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<10, 1, 5>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_7_3_10"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<7, 3, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_5_5_15"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<5, 5, 15>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_8_8_10"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<8, 8, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_9_9_10"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<9, 9, 10>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_16_16_40"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<16, 16, 40>>(seed);
    } else if (isGameNameMatched(
                   {"Minesweeper_30_16_99"})) {  // width, height, mines
      state_ = newState<Minesweeper::State<30, 16, 99>>(seed);
    } else if (isGameNameMatched({"TicTacToe", "NoughtsAndCrosses", "XsAndOs",
                                  "MNKGame_3_3_3"})) {
      state_ = newState<MNKGame::State<3, 3, 3>>(seed);
    } else if (isGameNameMatched(
                   {"FreeStyleGomoku", "GomokuFreeStyle", "MNKGame_15_15_5"})) {
      state_ = newState<MNKGame::State<15, 15, 5>>(seed);
    } else if (isGameNameMatched(
                   {"Othello4", "Reversi4", "Othello04", "Reversi04"})) {
      state_ = newState<Othello::State<6>>(seed);
    } else if (isGameNameMatched(
                   {"Othello6", "Reversi6", "Othello06", "Reversi06"})) {
      state_ = newState<Othello::State<6>>(seed);
    } else if (isGameNameMatched({"Othello8", "Reversi8", "Othello08",
                                  "Reversi08", "Othello", "Reversi"})) {
      state_ = newState<Othello::State<8>>(seed);
    } else if (isGameNameMatched({"Othello10", "Reversi10"})) {
      state_ = newState<Othello::State<10>>(seed);
    } else if (isGameNameMatched({"Othello12", "Reversi12"})) {
      state_ = newState<Othello::State<12>>(seed);
    } else if (isGameNameMatched({"Othello14", "Reversi14"})) {
      state_ = newState<Othello::State<14>>(seed);
    } else if (isGameNameMatched({"Othello16", "Reversi16"})) {
      state_ = newState<Othello::State<16>>(seed);
    } else if (isGameNameMatched({"OthelloOpt8", "OthelloOpt", "ReversiOpt8",
                                  "ReversiOpt"})) {
      state_ = newState<Othello2::State<8>>(seed);
    } else if (isGameNameMatched({"OthelloOpt10", "ReversiOpt10"})) {
      state_ = newState<Othello2::State<10>>(seed);
    } else if (isGameNameMatched({"OthelloOpt16", "ReversiOpt16"})) {
      state_ = newState<Othello2::State<16>>(seed);
    } else if (isGameNameMatched({"GameOfTheAmazons", "Amazons"})) {
      state_ = newState<Amazons::State>(seed);
    } else if (isGameNameMatched({"ChineseCheckers"})) {
      state_ = newState<ChineseCheckers::State>(seed);
    } else if (isGameNameMatched({"Hex5pie"})) {
      state_ = newState<Hex::State<5, true>>(seed);
    } else if (isGameNameMatched({"Hex11pie"})) {
      state_ = newState<Hex::State<11, true>>(seed);
    } else if (isGameNameMatched({"Hex13pie"})) {
      state_ = newState<Hex::State<13, true>>(seed);
    } else if (isGameNameMatched({"Hex19pie"})) {
      state_ = newState<Hex::State<19, true>>(seed);
    } else if (isGameNameMatched({"Hex5"})) {
      state_ = newState<Hex::State<5, false>>(seed);
    } else if (isGameNameMatched({"Hex11"})) {
      state_ = newState<Hex::State<11, false>>(seed);
    } else if (isGameNameMatched({"Hex13"})) {
      state_ = newState<Hex::State<13, false>>(seed);
    } else if (isGameNameMatched({"Hex19"})) {
      state_ = newState<Hex::State<19, false>>(seed);
    } else if (isGameNameMatched(
                   {"Havannah5pieExt"})) {  // ext = borders, corners
      state_ = newState<Havannah::State<5, true, true>>(seed);
    } else if (isGameNameMatched({"Havannah10pieExt"})) {
      state_ = newState<Havannah::State<10, true, true>>(seed);
    } else if (isGameNameMatched({"Havannah8pieExt"})) {
      state_ = newState<Havannah::State<8, true, true>>(seed);
    } else if (isGameNameMatched({"Havannah5pie"})) {
      state_ = newState<Havannah::State<5, true, false>>(seed);
    } else if (isGameNameMatched({"Havannah8pie"})) {
      state_ = newState<Havannah::State<8, true, false>>(seed);
    } else if (isGameNameMatched({"Havannah10pie"})) {
      state_ = newState<Havannah::State<10, true, false>>(seed);
    } else if (isGameNameMatched({"Havannah5"})) {
      state_ = newState<Havannah::State<5, false, false>>(seed);
    } else if (isGameNameMatched({"Havannah8"})) {
      state_ = newState<Havannah::State<8, false, false>>(seed);
    } else if (isGameNameMatched({"Havannah10"})) {
      state_ = newState<Havannah::State<10, false, false>>(seed);
    } else if (isGameNameMatched({"Breakthrough"})) {
      state_ = newState<StateForBreakthrough<false>>(seed);
    } else if (isGameNameMatched({"BreakthroughV2"})) {
      state_ = newState<StateForBreakthrough<true>>(seed);
    } else if (gameName.rfind("Ludii", 0) == 0) {
#ifdef NO_JAVA
      throw std::runtime_error(
          "Java/JNI support has not been built in, but is required for Ludii");
#else
      std::string ludii_name = gameName.substr(5);
      Ludii::JNIUtils::InitJVM("");  // Use default /ludii/Ludii.jar path
      JNIEnv* jni_env = Ludii::JNIUtils::GetEnv();

      if (jni_env) {
		if (gameOptions.size() > 0) {
		  Ludii::LudiiGameWrapper game_wrapper(ludii_name, gameOptions);
		  for (const std::string option : gameOptions) {
	        std::cout << "Using Game Option: " << option << std::endl;
		  }
          state_ =
            newState<Ludii::LudiiStateWrapper>(seed, std::move(game_wrapper));
		} else {
		  Ludii::LudiiGameWrapper game_wrapper(ludii_name);
          state_ =
            newState<Ludii::LudiiStateWrapper>(seed, std::move(game_wrapper));
		}
      } else {
        // Probably means we couldn't find the Ludii.jar file
        throw std::runtime_error(
            "Failed to create Ludii game due to missing JNI Env!");
      }
#endif
    } else if (isGameNameMatched({"Tristannogo"})) {
      state_ = newState<StateForTristannogo>(seed);
    } else if (isGameNameMatched({"OuterOpenGomoku", "OOGomoku"})) {
      state_ = newState<StateForOOGomoku>(seed);
    } else if (isGameNameMatched({"Minishogi"})) {
      state_ = newState<StateForMinishogi<1>>(seed);
    } else if (isGameNameMatched({"MinishogiV2"})) {
      state_ = newState<StateForMinishogi<2>>(seed);
    } else if (isGameNameMatched({"Surakarta"})) {
      state_ = newState<StateForSurakarta>(seed);
    } else if (isGameNameMatched({"DiceShogi"})) {
      state_ = newState<StateForDiceshogi>(seed);
    } else if (isGameNameMatched({"BlockGo"})) {
      state_ = newState<StateForBlockGo>(seed);
    } else if (isGameNameMatched({"YINSH"})) {
      state_ = newState<StateForYinsh>(seed);
    } else if (isGameNameMatched({"GomokuSwap2", "Swap2Gomoku", "Gomoku"})) {
      state_ = newState<GomokuSwap2::State>(seed);
    } else if (isGameNameMatched({"KyotoShogi"})) {
      state_ = newState<StateForKyotoshogi>(seed);
    } else if (isGameNameMatched({"Einstein"})) {
      state_ = newState<StateForEinstein>(seed);
    } else if (isGameNameMatched({"WeakSchur_3_20"})) {  // subsets, maxNumber
      state_ = newState<weakschur::State<3, 20>>(seed);

    } else if (isGameNameMatched({"WeakSchur_4_66"})) {  // subsets, maxNumber
      state_ = newState<weakschur::State<4, 66>>(seed);
      // } else if (isGameNameMatched(gameName, {"Nogo"})) {
      //   state_ = newState<StateForNogo>();
    } else if (isGameNameMatched(
                   {"WeakSchur_5_197",
                    "WalkerSchur"})) {  // subsets, maxNumber  // is Walker
                                        // right ?   (1952! he said 197...)
      state_ = newState<weakschur::State<5, 197>>(seed);
    } else if (isGameNameMatched({"WeakSchur_3_70", "ImpossibleSchur"})) {
      state_ = newState<weakschur::State<3, 70>>(seed);
    } else if (isGameNameMatched(
                   {"WeakSchur_6_583",
                    "FabienSchur"})) {  // subsets, maxNumber  // beating F.
                                        // Teytaud et al
      state_ = newState<weakschur::State<6, 583>>(seed);
    } else if (isGameNameMatched({"WeakSchur_7_1737",
                                  "Arpad7Schur"})) {  // beating A. Rimmel et al
      state_ = newState<weakschur::State<7, 1737>>(seed);
    } else if (isGameNameMatched({"WeakSchur_8_5197",
                                  "Arpad8Schur"})) {  // beating A. Rimmel et al
      state_ = newState<weakschur::State<8, 5197>>(seed);
    } else if (isGameNameMatched({"WeakSchur_9_15315",
                                  "Arpad9Schur"})) {  // beating A. Rimmel et al
      state_ = newState<weakschur::State<9, 15315>>(seed);
    } else if (isGameNameMatched({"Chess"})) {
      state_ = newState<chess::State>(seed);
    } else {
      throw std::runtime_error("Unknown game name '" + gameName + "'");
    }

    setFeatures(outFeatures, turnFeaturesSingleChannel,
                turnFeaturesMultiChannel, geometricFeatures, history,
                randomFeatures, oneFeature);

    state_->Initialize();
  }

  template <typename T, typename... A>
  std::unique_ptr<State> newState(A&&... args) {
    auto r = std::make_unique<T>(std::forward<A>(args)...);
    r->template initializeAs<T>();
    return r;
  }

  virtual bool isOnePlayerGame() const {
    return state_->isOnePlayerGame();
  }

  void setFeatures(bool outFeatures,
                   bool turnFeaturesSingleChannel,
                   bool turnFeaturesMultiChannel,
                   bool geometricFeatures,
                   int history,
                   int randomFeatures,
                   bool oneFeature) {
    featopts.emplace_back();
    FeatureOptions& opt = featopts.back();
    opt.outFeatures = outFeatures;
    opt.turnFeaturesSingleChannel = turnFeaturesSingleChannel;
    opt.turnFeaturesMultiChannel = turnFeaturesMultiChannel;
    opt.geometricFeatures = geometricFeatures;
    opt.history = history;
    opt.randomFeatures = randomFeatures;
    opt.oneFeature = oneFeature;

    state_->setFeatures(&opt);
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

  void addPlayer(std::shared_ptr<core::ActorPlayer> player,
                 std::shared_ptr<tube::DataChannel> dc,
                 std::shared_ptr<Game> game,
                 std::shared_ptr<core::ActorPlayer> devplayer) {
    assert(dc != nullptr && !evalMode);

    players_.push_back(player);
    playerGame_.push_back(game);

    if (devplayer) {
      player = devplayer;
    }
    int seqlen = player->rnnSeqlen();

    auto addseq = [&](std::vector<int64_t> a) {
      if (seqlen) {
        a.insert(a.begin(), seqlen);
      }
      return a;
    };

    auto feat = tube::EpisodicTrajectory(
        "s", addseq(state_->GetFeatureSize()), torch::kFloat32);
    auto rnnInitialState = tube::EpisodicTrajectory(
        "rnn_initial_state", player->rnnStateSize(), torch::kFloat32);
    auto rnnStateMask = tube::EpisodicTrajectory(
        "rnn_state_mask", addseq({1}), torch::kFloat32);
    auto pi = tube::EpisodicTrajectory(
        "pi", addseq(state_->GetActionSize()), torch::kFloat32);
    auto piMask = tube::EpisodicTrajectory(
        "pi_mask", addseq(state_->GetActionSize()), torch::kFloat32);
    auto actionPi = tube::EpisodicTrajectory(
        "action_pi", addseq(state_->GetActionSize()), torch::kFloat32);
    auto v = tube::EpisodicTrajectory(
        "v", addseq({player->vOutputs()}), torch::kFloat32);
    auto predV = tube::EpisodicTrajectory(
        "pred_v", addseq({player->vOutputs()}), torch::kFloat32);
    int predicts = (predictEndState ? 2 : 0) + predictNStates;
    auto predictSize = state_->GetRawFeatureSize();
    predictSize[0] *= predicts;
    auto predictPi = tube::EpisodicTrajectory(
        "predict_pi", addseq(predictSize), torch::kFloat32);
    auto predictPiMask = tube::EpisodicTrajectory(
        "predict_pi_mask", addseq(predictSize), torch::kFloat32);

    tube::Dispatcher dispatcher(std::move(dc));
    std::vector<std::shared_ptr<tube::DataBlock>> send;
    send = {feat.buffer, pi.buffer, piMask.buffer, v.buffer, predV.buffer};
    if (predictEndState + predictNStates) {
      send.push_back(predictPi.buffer);
      send.push_back(predictPiMask.buffer);
      predictPi_.push_back(predictPi);
      predictPiMask_.push_back(predictPiMask);
    }
    if (seqlen) {
      send.push_back(rnnInitialState.buffer);
      rnnInitialState_.push_back(rnnInitialState);
      send.push_back(rnnStateMask.buffer);
      rnnStateMask_.push_back(rnnStateMask);
    }
    if (dynamic_cast<ForwardPlayer*>(&*player) != nullptr) {
      send.push_back(actionPi.buffer);
      actionPi_.push_back(actionPi);
    }
    dispatcher.addDataBlocks(send, {});

    feature_.push_back(feat);
    pi_.push_back(pi);
    piMask_.push_back(piMask);
    v_.push_back(v);
    predV_.push_back(predV);
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
  const bool evalMode;
  const int perThreadBatchSize;
  const int maxRewinds;
  const bool predictEndState;
  const int predictNStates;

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
  }

  std::optional<int> parseSpecialAction(const std::string& str);

  void step();

  void sendTrajectory();

  bool prepareForSend(int playerId);

  std::unique_ptr<State> state_;
  std::vector<std::shared_ptr<Player>> players_;
  std::vector<std::shared_ptr<Game>> playerGame_;

  std::vector<tube::EpisodicTrajectory> feature_;
  std::vector<tube::EpisodicTrajectory> rnnStateMask_;
  std::vector<tube::EpisodicTrajectory> rnnInitialState_;
  std::vector<tube::EpisodicTrajectory> pi_;
  std::vector<tube::EpisodicTrajectory> piMask_;
  std::vector<tube::EpisodicTrajectory> actionPi_;
  std::vector<tube::EpisodicTrajectory> v_;
  std::vector<tube::EpisodicTrajectory> predV_;
  std::vector<tube::EpisodicTrajectory> predictPi_;
  std::vector<tube::EpisodicTrajectory> predictPiMask_;

  std::vector<tube::Dispatcher> dispatchers_;

  std::list<FeatureOptions> featopts;

  std::vector<float> result_;

  std::mutex mutexStats_;
  EnvThread::Stats stats_;

  std::string lastAction_;
  bool hasPrintedHumanHelp_ = false;
  bool isInSingleMoveMode_ = false;
  float lastMctsValue_ = 0.0f;
  bool printMoves_ = false;
  std::string gameName_;
  std::vector<torch::Tensor> rnnState_;
};

}  // namespace core
