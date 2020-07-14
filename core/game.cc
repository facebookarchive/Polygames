/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "game.h"

#include <fmt/printf.h>

#include <torch/nn/module.h>

inline std::atomic_int threadIdCounter{0};
inline thread_local int threadId = ++threadIdCounter;

class NoiseModel : public torch::nn::Module {
 public:
  torch::nn::Conv2d input = nullptr;
  std::vector<torch::nn::Linear> hiddens;
  torch::nn::Linear output = nullptr;

  int64_t inputSize;
  std::vector<int64_t> outputShape;

  NoiseModel(std::vector<int64_t> inputs,
             std::vector<int64_t> outputs,
             size_t hidden,
             int layers = 4) {

    if (inputs.at(1) < 3 || inputs.at(2) < 3) {
      input = torch::nn::Conv2d(
          torch::nn::Conv2dOptions(inputs.at(0), 1, 3).padding(1));
      inputSize = inputs.at(1) * inputs.at(2);
    } else {
      input = torch::nn::Conv2d(torch::nn::Conv2dOptions(inputs.at(0), 1, 3));
      inputSize = (inputs.at(1) - 2) * (inputs.at(2) - 2);
    }

    int64_t s = inputSize;
    for (int i = 0; i != layers; ++i) {
      hiddens.push_back(torch::nn::Linear(s, hidden));
      s = hidden;
    }
    output = torch::nn::Linear(s, product(outputs));

    outputShape = outputs;
    outputShape.insert(outputShape.begin(), -1);
  }

  torch::Tensor forward(torch::Tensor input) {

    auto x = this->input->forward(input).view({-1, inputSize}).relu();
    for (auto& v : hiddens) {
      x = v->forward(x).relu();
    }
    x = output->forward(x);
    return x.view(outputShape);
  }
};

void Game::mainLoop() {
  if (players_.size() != (isOnePlayerGame() ? 1 : 2)) {
    std::cout << "Error: wrong number of players: " << players_.size()
              << std::endl;
    assert(false);
  }
  if (perThreadBatchSize != 0) {
    bool aHuman = std::any_of(players_.begin(), players_.end(),
                              [](const std::shared_ptr<mcts::Player>& player) {
                                return player->isHuman();
                              });
    if (aHuman && state_->stochasticReset()) {
      std::string line;
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
    }
    reset();

    for (auto& v : playerGame_) {
      if (v->state_) {
        v->state_->reset();
      }
    }

    std::vector<NoiseModel> noiseModels;
    std::vector<std::unique_ptr<torch::optim::Adam>> noiseOpt;
    std::vector<int> noiseOptSteps;
    for (auto& v : players_) {
      noiseModels.emplace_back(
          state_->GetRawFeatureSize(), state_->GetActionSize(), 64);

      noiseOpt.push_back(std::make_unique<torch::optim::Adam>(
          noiseModels.back().parameters(), torch::optim::AdamOptions(1e-3)));
      noiseOptSteps.push_back(0);
    }
    std::vector<float> noiseRunningReward;
    std::vector<float> nonoiseRunningReward;
    noiseRunningReward.resize(players_.size());
    nonoiseRunningReward.resize(players_.size());

    auto basestate = std::move(state_);

    struct MoveHistory {
      int turn = 0;
      uint64_t move = 0;
      float value = 0.0f;
      torch::Tensor shortFeat;
      bool featurized = false;

      torch::Tensor noiseTarget;
      torch::Tensor noiseTargetMask;
    };

    struct Sequence {
      std::vector<torch::Tensor> feat;
      std::vector<torch::Tensor> v;
      std::vector<torch::Tensor> pi;
      std::vector<torch::Tensor> piMask;
      torch::Tensor rnnInitialState;
      std::vector<torch::Tensor> rnnStateMask;
      std::vector<torch::Tensor> predictPi;
      std::vector<torch::Tensor> predictPiMask;
    };

    std::vector<Sequence> seqs;
    seqs.resize(players_.size());

    struct GameState {
      std::unique_ptr<State> state;
      std::vector<std::unique_ptr<State>> playerState;
      std::vector<size_t> players;
      std::vector<size_t> playersReverseMap;
      std::vector<std::vector<torch::Tensor>> feat;
      std::vector<std::vector<torch::Tensor>> pi;
      std::vector<std::vector<torch::Tensor>> piMask;
      std::vector<std::vector<std::vector<float>>> rnnStates;
      std::vector<std::vector<float>> reward;
      size_t stepindex;
      std::chrono::steady_clock::time_point start;
      std::vector<int> resignCounter;
      int drawCounter = 0;
      bool canResign = false;
      int resigned = -1;
      bool drawn = false;
      std::chrono::steady_clock::time_point prevMoveTime =
          std::chrono::steady_clock::now();
      std::vector<size_t> playerOrder;
      std::vector<MoveHistory> history;
      bool justRewound = false;
      bool justRewoundToNegativeValue = false;
      int rewindCount = 0;
      mcts::PersistentTree tree;
      std::vector<int> useNoiseModel;
      std::vector<std::vector<float>> rnnState;
      std::vector<std::vector<float>> rnnState2;

      std::vector<int> allowRandomMoves;
      bool validTournamentGame = false;
    };

    std::list<GameState> states;

    bool autoTuneBatchSize = perThreadBatchSize < 0;
    size_t ngames = autoTuneBatchSize ? 1 : size_t(perThreadBatchSize);

    int64_t startedGameCount = 0;
    int64_t completedGameCount = 0;

    std::minstd_rand rng(std::random_device{}());
    auto randint = [&](int n) {
      return std::uniform_int_distribution<int>(0, n - 1)(rng);
    };

    auto cloneState = [&](auto& state) {
      auto x = state->clone();
      auto n = dynamic_cast<State*>(x.get());
      std::unique_ptr<State> r(n);
      if (n) {
        x.release();
      }
      return r;
    };

    std::list<GameState> freeGameList;

    float runningAverageGameSteps = 0.0f;

    auto doRandomMoves = [&](GameState& gst, int n) {
      auto o = cloneState(gst.state);
      std::vector<size_t> moves;
      for (;n > 0; --n) {
        if (gst.state->terminated()) {
          break;
        }
        size_t n = randint(gst.state->GetLegalActions().size());
        moves.push_back(n);
        gst.state->forward(n);
      }
      if (gst.state->terminated()) {
        gst.state = std::move(o);
      } else {
        for (auto m : moves) {
          for (auto& x : gst.playerState) {
            if (x) {
              x->forward(m);
            }
          }
        }
        fmt::printf("Did %d random moves: '%s'\n", gst.state->getStepIdx(), gst.state->history());
      }
    };

    auto addGame = [&](auto at) {
      if (!freeGameList.empty()) {
        GameState gst = std::move(freeGameList.front());
        freeGameList.pop_front();
        return states.insert(at, std::move(gst));
      }
      ++startedGameCount;
      GameState gst;
      for (size_t i = 0; i != players_.size(); ++i) {
        gst.players.push_back(i);
      }
      std::shuffle(gst.players.begin(), gst.players.end(), rng);
      gst.playersReverseMap.resize(players_.size());
      for (size_t i = 0; i != players_.size(); ++i) {
        gst.playersReverseMap[gst.players[i]] = i;
      }
      gst.state = cloneState(basestate);
      unsigned long seed = rng();
      gst.state->newGame(seed);
      gst.playerState.resize(players_.size());
      for (size_t i = 0; i != players_.size(); ++i) {
        std::unique_ptr<State> s = nullptr;
        int index = gst.players[i];
        if (&*playerGame_[index] != this) {
          s = cloneState(playerGame_[index]->state_);
          s->newGame(seed);
        }
        gst.playerState[i] = std::move(s);
      }
      gst.feat.resize(players_.size());
      gst.pi.resize(players_.size());
      gst.piMask.resize(players_.size());
      gst.reward.resize(players_.size());
      gst.rnnState.resize(players_.size());
      gst.rnnState2.resize(players_.size());
      gst.rnnStates.resize(players_.size());
      gst.stepindex = 0;
      gst.start = std::chrono::steady_clock::now();
      gst.resignCounter.resize(players_.size());
      gst.canResign = !evalMode && players_.size() == 2 && randint(3) != 0;
      gst.validTournamentGame = true;
      gst.allowRandomMoves.resize(players_.size());
      for (auto& v : gst.allowRandomMoves) {
        v = randint(4) == 0;
      }
      gst.useNoiseModel.resize(players_.size());
      for (auto& v : gst.useNoiseModel) {
        //v = randint(8) == 0;
        v = false;
      }
      if (randint(250) == 0) {
        switch (randint(2)) {
        case 0:
          doRandomMoves(gst, randint(std::max((int)runningAverageGameSteps, 1)));
          break;
        case 1:
          doRandomMoves(gst, randint(std::max((int)runningAverageGameSteps / 10, 1)));
          break;
        case 2:
          doRandomMoves(gst, randint(std::max((int)runningAverageGameSteps / 5, 1)));
          break;
        }
        gst.validTournamentGame = false;
      }
      return states.insert(at, std::move(gst));
    };

    while (states.size() < ngames &&
           (numEpisode < 0 || startedGameCount < numEpisode)) {
      addGame(states.end());
    }

    std::vector<std::shared_ptr<mcts::MctsPlayer>> mctsPlayers;
    std::shared_ptr<mcts::MctsPlayer> devPlayer;
    for (auto& v : players_) {
      auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(v);
      if (!mctsPlayer) {
        throw std::runtime_error(
            "Cannot use perThreadBatchSize without MctsPlayer");
      }
      if (mctsPlayer->getName() == "dev") {
        devPlayer = mctsPlayer;
      }
      mctsPlayers.push_back(std::move(mctsPlayer));
    }

    std::vector<std::vector<const mcts::State*>> actStates(players_.size());
    std::vector<std::vector<const mcts::State*>> actPlayerStates(
        players_.size());
    std::vector<std::vector<GameState*>> actGameStates(players_.size());

    bool alignPlayers = true;

    // If two players are the same (pointer comparison), then they can act
    // together.
    std::vector<size_t> remapPlayerIdx(players_.size());
    for (size_t i = 0; i != players_.size(); ++i) {
      remapPlayerIdx[i] = i;
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (i != i2 && players_[i] == players_[i2]) {
          remapPlayerIdx[i] = i2;
        }
      }
    }

    std::vector<std::pair<size_t, size_t>> statePlayerSize;

    auto rewind = [&](GameState* s, int player, bool rewindToNegativeValue) {
      if (s->history.size() <= 2) {
        // fmt::printf("refusing to rewind with history size %d\n",
        // s->history.size());
        return false;
      }
      float flip = rewindToNegativeValue ? -1 : 1;
      size_t index = 0;
      for (index = s->history.size(); index;) {
        --index;
        auto& h = s->history[index];
        if (h.turn == player && h.value * flip > 0) {
          break;
        }
      }
      if (index <= 2) {
        // fmt::printf("refusing to rewind to index %d\n", index);
        return false;
      }
      if (!s->rnnStates.empty() || !s->rnnState.empty() ||
          !s->rnnState2.empty()) {
        bool rnn = false;
        for (auto& v : mctsPlayers) {
          if (v->rnnSeqlen()) {
            rnn = true;
          }
        }
        if (rnn) {
          fmt::printf("Cannot currently rewind with rnn states, sorry :(\n");
          return false;
        }
      }
      fmt::printf("rewinding from %d to index %d\n", s->history.size(), index);
      s->justRewound = true;
      s->justRewoundToNegativeValue = rewindToNegativeValue;

      auto& gst = *s;
      gst.state = cloneState(basestate);

      for (size_t i = 0; i != gst.playerState.size(); ++i) {
        auto& x = gst.playerState[i];
        if (x) {
          int player = gst.players.at(i);
          x = cloneState(playerGame_.at(player)->state_);
        }
      }

      for (auto& v : gst.feat) {
        v.clear();
      }
      for (auto& v : gst.pi) {
        v.clear();
      }
      for (auto& v : gst.piMask) {
        v.clear();
      }
      for (auto& v : gst.reward) {
        v.clear();
      }
      for (auto& v : gst.resignCounter) {
        v = 0;
      }
      gst.drawCounter = 0;
      gst.resigned = -1;
      gst.drawn = false;

      gst.history.resize(index);
      for (auto& v : gst.history) {
        v.featurized = false;
        gst.state->forward(v.move);
        for (auto& x : gst.playerState) {
          if (x) {
            x->forward(v.move);
          }
        }
      }
      return true;
    };

    double batchLr = 1.0;

    while (!states.empty() && !terminate_) {

      if (autoTuneBatchSize) {
        double batchTimeSum = 0.0;
        int batchTimeN = 0;
        for (auto& v : mctsPlayers) {
          double t = v->batchTiming();
          if (t > 0) {
            ++batchTimeN;
            batchTimeSum += t;
          }
        }
        double batchTimeAvg = batchTimeSum / batchTimeN;
        if (batchTimeAvg > 0) {
          double adjust = 1.0 - batchTimeAvg;

          size_t add = size_t(ngames / 4 * batchLr);
          if (adjust > 0.05) {
            if (ngames < 1024) {
              ++ngames;
            }
            if (1024 - ngames >= add) {
              ngames += add;
            }
          }
          if (adjust < -0.05) {
            if (ngames > 1) {
              --ngames;
            }
            if (ngames > add) {
              ngames -= add;
            }
          }
        }

        batchLr *= 0.97;

        while (states.size() < ngames &&
               (numEpisode < 0 || startedGameCount < numEpisode)) {
          addGame(states.end());
        }

        while (states.size() > ngames) {
          freeGameList.push_back(std::move(states.back()));
          states.pop_back();
        }

        {
          std::unique_lock<std::mutex> lkStats(mutexStats_);
          auto& stats_steps = stats_["Game batch size"];
          std::get<0>(stats_steps) += 1;
          std::get<1>(stats_steps) += ngames;
          std::get<2>(stats_steps) += ngames * ngames;
        }
      }

      for (auto& v : actStates) {
        v.clear();
      }
      for (auto& v : actPlayerStates) {
        v.clear();
      }
      for (auto& v : actGameStates) {
        v.clear();
      }

      for (auto i = states.begin(); i != states.end();) {
        auto* state = &*i->state;
        if (state->terminated() || i->resigned != -1 || i->drawn) {
          const auto end = std::chrono::steady_clock::now();
          const auto elapsed =
              std::chrono::duration_cast<std::chrono::seconds>(end - i->start)
                  .count();
          const size_t stepindex = i->stepindex;
          {
            std::unique_lock<std::mutex> lkStats(mutexStats_);
            auto& stats_steps = stats_["Game Duration (steps)"];
            std::get<0>(stats_steps) += 1;
            std::get<1>(stats_steps) += stepindex;
            std::get<2>(stats_steps) += stepindex * stepindex;
            auto& stats_s = stats_["Game Duration (seconds)"];
            std::get<0>(stats_s) += 1;
            std::get<1>(stats_s) += elapsed;
            std::get<2>(stats_s) += elapsed * elapsed;
          }
          if (i->drawn) {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(i->players.at(idx)) = 0;
            }
          }
          if (i->resigned != -1) {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(i->players.at(idx)) = int(idx) == i->resigned ? -1 : 1;
            }
            fmt::printf("player %d (%s) resigned : %s\n", i->resigned,
                        players_.at(i->players.at(i->resigned))->getName(),
                        state->history());
          } else {
            for (size_t idx = 0; idx != players_.size(); ++idx) {
              result_.at(i->players.at(idx)) = state->getReward(idx);
            }
            fmt::printf("game ended normally: %s\n", state->history().c_str());
          }

          runningAverageGameSteps = runningAverageGameSteps * 0.99f + state->getStepIdx() * 0.01f;

          bool doRewind = false;
          int rewindPlayer = 0;
          bool rewindToNegativeValue = false;

          for (size_t slot = 0; slot != players_.size(); ++slot) {
            size_t dstp = i->players.at(slot);
            fmt::printf("Result for %s: %g\n", players_[dstp]->getName(),
                        result_[dstp]);

            int seqlen = devPlayer->rnnSeqlen();

            auto addseq = [&](const std::vector<torch::Tensor>& src,
                              std::vector<torch::Tensor>& dst,
                              tube::EpisodicTrajectory& traj) {
              for (auto& x : src) {
                dst.push_back(x);
                if ((int)dst.size() > seqlen) {
                  throw std::runtime_error("addseq bad seqlen");
                }
                if ((int)dst.size() == seqlen) {
                  traj.pushBack(torch::stack(dst));
                  dst.clear();
                  // printf("sent sequence!\n");
                }
              }
            };

            std::vector<torch::Tensor> rewards;
            for (size_t j = 0; j != i->feat[slot].size(); ++j) {
              torch::Tensor reward = torch::zeros({1}, torch::kFloat32);
              reward[0] = result_[dstp];
              rewards.push_back(std::move(reward));
            }

            std::vector<float> piReward;
            piReward.resize(i->pi[slot].size());
            //            float reward = result_[dstp];
            //            for (size_t n = piReward.size(); n;) {
            //              --n;
            //              reward *= 0.99;
            //              reward += i->reward[slot].at(n);
            //              if (reward > 1.0f) {
            //                reward = 1.0f;
            //              }
            //              if (reward < -1.0f) {
            //                reward = -1.0f;
            //              }
            //              piReward.at(n) = reward;

            //              rewards.at(n)[0] = reward;
            //            }
            auto& seq = seqs[dstp];
            for (size_t n = 0; n != i->pi[slot].size(); ++n) {
              // i->pi[slot][n] *= result_[slot] + piReward.at(n);
            }
            // if ((mctsPlayers[dstp]->getModelId() == "dev" || result_[dstp] >
            // 0) && i->feat[slot].size() > 0) {
            if (i->feat[slot].size() > 0) {
              if (seqlen) {
                for (size_t n = 0; n != i->feat[slot].size(); ++n) {
                  if ((seq.feat.size() + n) % seqlen == seqlen - 1) {
                    auto& vec = i->rnnStates[slot].at(n);
                    torch::Tensor x = torch::empty({(int64_t)vec.size()});
                    std::memcpy(
                        x.data_ptr(), vec.data(), sizeof(float) * vec.size());
                    rnnInitialState_[dstp].pushBack(std::move(x));
                  }
                }
                addseq(i->feat[slot], seq.feat, feature_[dstp]);
                addseq(i->pi[slot], seq.pi, pi_[dstp]);
                addseq(i->piMask[slot], seq.piMask, piMask_[dstp]);
                std::vector<torch::Tensor> rnnStateMask;
                rnnStateMask.resize(i->feat[slot].size(), torch::ones({1}));
                rnnStateMask.at(0).zero_();
                addseq(rnnStateMask, seq.rnnStateMask, rnnStateMask_[dstp]);
              } else {
                for (auto& v : i->feat[slot]) {
                  feature_[dstp].pushBack(v);
                }
                for (auto& v : i->pi[slot]) {
                  pi_[dstp].pushBack(v);
                }
                for (auto& v : i->piMask[slot]) {
                  piMask_[dstp].pushBack(v);
                }
              }

              if (predictEndState || predictNStates) {
                int n = (predictEndState ? 2 : 0) + predictNStates;
                auto size = state->GetRawFeatureSize();
                size.insert(size.begin(), n);
                auto finalsize = size;
                finalsize[1] *= finalsize[0];
                finalsize.erase(finalsize.begin());
                for (size_t m = 0; m != i->history.size(); ++m) {
                  if (!i->history[m].featurized ||
                      i->history[m].turn != (int)slot) {
                    continue;
                  }
                  auto tensor = torch::zeros(size);
                  auto mask = torch::zeros(size);
                  size_t offset = 0;
                  if (predictEndState) {
                    if (state->terminated()) {
                      tensor[0].copy_(i->history.back().shortFeat);
                      mask[0].fill_(1.0f);
                    } else {
                      tensor[1].copy_(i->history.back().shortFeat);
                      mask[1].fill_(1.0f);
                    }
                    offset += 2;
                  }
                  for (int j = 0; j != predictNStates; ++j, ++offset) {
                    size_t index = m + 1 + j;
                    if (index < i->history.size()) {
                      tensor[offset].copy_(i->history[m].shortFeat);
                      mask[offset].fill_(1.0f);
                    }
                  }

                  tensor = tensor.view(finalsize);
                  mask = mask.view(finalsize);

                  if (seqlen) {
                    addseq({tensor}, seq.predictPi, predictPi_[dstp]);
                    addseq({mask}, seq.predictPiMask, predictPiMask_[dstp]);
                  } else {
                    predictPi_[dstp].pushBack(tensor);
                    predictPiMask_[dstp].pushBack(mask);
                  }
                }
              }

              // fmt::printf("result[%d] (%s) is %g\n", p,
              // players_[p]->getName(), result_[p]);

              if (seqlen) {
                addseq(rewards, seq.v, v_[dstp]);
              } else {
                for (auto& reward : rewards) {
                  v_[dstp].pushBack(std::move(reward));
                }
              }
            }

            if (mctsPlayers[dstp]->getModelId() == "dev") {
              if (result_[dstp] != 0) {
                doRewind = true;
                rewindPlayer = slot;
                rewindToNegativeValue = result_[dstp] > 0;
              }
            }

            if (i->rewindCount == 0 && i->validTournamentGame) {
              players_[dstp]->result(state, result_[dstp]);
            } else {
              players_[dstp]->forget(state);
            }

            float a = 0.9875;
            if (i->useNoiseModel.at(slot)) {
              float reward = result_[dstp];
              noiseRunningReward.at(slot) =
                  noiseRunningReward.at(slot) * a + reward * (1 - a);
              //fmt::printf("running reward for noise slot %d: %f\n", slot, noiseRunningReward.at(slot));
              //if (reward > 0) {
              if (true) {
                std::vector<torch::Tensor> inputs;
                std::vector<torch::Tensor> target;
                std::vector<torch::Tensor> targetMask;
                for (auto& v : i->history) {
                  if (v.turn == (int)slot) {
                    inputs.push_back(v.shortFeat);
                    target.push_back(v.noiseTarget);
                    targetMask.push_back(v.noiseTargetMask);
                  }
                }
                auto o = noiseModels.at(slot).forward(torch::stack(inputs));

                o = o.flatten(1);
                auto mask = torch::stack(targetMask).flatten(1);
                o = o * mask - 400 * (1 - mask);

                auto logo = o.log_softmax(1) * mask;

                auto loss =
                    -(logo * torch::stack(target).flatten(1)).sum(1).mean();

                // fmt::printf("noise loss: %g\n", loss.item<float>());

                loss.backward();

                noiseOpt.at(slot)->step();
                noiseOpt.at(slot)->zero_grad();

                if (++noiseOptSteps.at(slot) >= 15000) {
                  noiseModels.at(slot) = NoiseModel(
                      state_->GetRawFeatureSize(), state_->GetActionSize(), 64);

                  noiseOpt.at(slot) = std::make_unique<torch::optim::Adam>(
                      noiseModels.back().parameters(),
                      torch::optim::AdamOptions(1e-3));

                  noiseOptSteps.at(slot) = 0;
                }
              }
            } else {
              float reward = result_[dstp];
              nonoiseRunningReward.at(slot) =
                  nonoiseRunningReward.at(slot) * a + reward * (1 - a);
              //fmt::printf("running reward for nonoise slot %d: %f\n", slot, nonoiseRunningReward.at(slot));
            }
          }
          sendTrajectory();

          if (doRewind) {
            for (size_t slot = 0; slot != players_.size(); ++slot) {
              size_t dstp = i->players.at(slot);
              if (mctsPlayers[dstp]->wantsTournamentResult()) {
                doRewind = false;
                break;
              }
            }
          }

          ++completedGameCount;
          if (doRewind && i->rewindCount < maxRewinds &&
              rewind(&*i, rewindPlayer, rewindToNegativeValue)) {
            ++i->rewindCount;
          } else {
            i = states.erase(i);
            if (numEpisode < 0 || startedGameCount < numEpisode) {
              i = addGame(i);
            }
          }
        } else {
          i->stepindex++;
          int slot = state->getCurrentPlayer();
          auto playerIdx = i->players.at(slot);
          actStates.at(playerIdx).push_back(state);
          actPlayerStates.at(playerIdx).push_back(&*i->playerState[slot]);
          actGameStates.at(playerIdx).push_back(&*i);
          ++i;
        }
      }

      std::vector<mcts::PersistentTree*> trees;

      auto actForPlayer = [&](size_t playerIndex) {
        // Merge all identical players so they get batched together
        auto& states = actStates[playerIndex];
        if (!states.empty()) {
          statePlayerSize.clear();
          statePlayerSize.emplace_back(playerIndex, states.size());
          for (size_t i = 0; i != players_.size(); ++i) {
            if (i != playerIndex && remapPlayerIdx[i] == playerIndex) {
              auto& nstates = actStates[i];
              if (!nstates.empty()) {
                states.insert(states.end(), nstates.begin(), nstates.end());
                statePlayerSize.emplace_back(i, nstates.size());
                nstates.clear();
              }
            }
          }
          std::vector<const mcts::State*> playerActStates;
          playerActStates.resize(states.size());
          size_t offset = 0;
          for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
            size_t currentPlayerIndex = statePlayerSize[pi].first;
            size_t currentPlayerStates = statePlayerSize[pi].second;
            for (size_t i = 0; i != currentPlayerStates; ++i) {
              GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);
              int slot = gameState->playersReverseMap.at(currentPlayerIndex);
              if (gameState->playerState.at(slot)) {
                playerActStates.at(offset + i) = &*gameState->playerState[slot];
              } else {
                playerActStates.at(offset + i) = states.at(offset + i);
              }
            }
            offset += currentPlayerStates;
          }
          if (persistentTree) {
            trees.resize(states.size());
            size_t offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerIndex = statePlayerSize[pi].first;
              size_t currentPlayerStates = statePlayerSize[pi].second;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                GameState* gameState =
                    actGameStates.at(currentPlayerIndex).at(i);
                trees[offset + i] = &gameState->tree;
              }
              offset += currentPlayerStates;
            }
          }
          //          fmt::printf("thread %d act %d states %d\n", threadId,
          //          playerIndex,
          //                      states.size());
          std::vector<std::vector<float>> policyBias;
          if (false) {

            std::vector<torch::Tensor> outputs;

            size_t offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerIndex = statePlayerSize[pi].first;
              size_t currentPlayerStates = statePlayerSize[pi].second;
              std::vector<torch::Tensor> vec;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                GameState* gameState =
                    actGameStates.at(currentPlayerIndex).at(i);
                size_t slot =
                    gameState->playersReverseMap.at(currentPlayerIndex);
                if (gameState->useNoiseModel.at(slot)) {
                  State* state = (State*)states.at(offset + i);

                  torch::Tensor noiseInput =
                      torch::empty(state->GetRawFeatureSize());
                  getRawFeatureInTensor(*state, noiseInput);
                  vec.push_back(noiseInput);
                }
              }
              offset += currentPlayerStates;

              if (vec.empty()) {
                outputs.push_back(torch::Tensor());
              } else {
                outputs.push_back(noiseModels.at(currentPlayerIndex)
                                      .forward(torch::stack(vec)));
              }
            }

            offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerIndex = statePlayerSize[pi].first;
              size_t currentPlayerStates = statePlayerSize[pi].second;
              auto o = outputs.at(pi);
              size_t ooffset = 0;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                State* state = (State*)states.at(offset + i);
                GameState* gameState =
                    actGameStates.at(currentPlayerIndex).at(i);
                size_t slot =
                    gameState->playersReverseMap.at(currentPlayerIndex);
                if (gameState->useNoiseModel.at(slot)) {
                  auto x = o[ooffset++];

                  std::vector<float> out;
                  getLegalPi(
                      *state,
                      x.flatten(0).softmax(0).view(state->GetActionSize()),
                      out);
                  policyBias.push_back(std::move(out));
                } else {
                  policyBias.emplace_back();
                }
              }
              offset += currentPlayerStates;
            }
          }

          std::vector<std::vector<float>> rnnState;
          if (mctsPlayers.at(playerIndex)->rnnSeqlen()) {
            size_t offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerIndex = statePlayerSize[pi].first;
              size_t currentPlayerStates = statePlayerSize[pi].second;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                State* state = (State*)playerActStates.at(offset + i);
                GameState* gameState =
                    actGameStates.at(currentPlayerIndex).at(i);

                size_t slot =
                    gameState->playersReverseMap.at(currentPlayerIndex);

                if (gameState->rnnState.at(slot).empty()) {
                  auto shape = mctsPlayers.at(playerIndex)->rnnStateSize();
                  int64_t rnnStateSize = shape.at(0) * shape.at(1);
                  size_t n = rnnStateSize * state->GetRawFeatureSize().at(1) *
                             state->GetRawFeatureSize().at(2);
                  gameState->rnnState[slot].resize(n);
                }

                rnnState.push_back(std::move(gameState->rnnState[slot]));

                if (&*playerGame_.at(playerIndex) == this) {
                  gameState->rnnStates.at(slot).push_back(rnnState.back());
                }
              }
              offset += currentPlayerStates;
            }
          }

          auto result =
              mctsPlayers.at(playerIndex)
                  ->actMcts(playerActStates, rnnState, trees, policyBias);

          if (true) {
            offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerIndex = statePlayerSize[pi].first;
              size_t currentPlayerStates = statePlayerSize[pi].second;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                State* state = (State*)states.at(offset + i);
                GameState* gameState =
                    actGameStates.at(currentPlayerIndex).at(i);

                size_t slot =
                    gameState->playersReverseMap.at(currentPlayerIndex);

                if (gameState->allowRandomMoves.at(slot)) {
                  float x = 4.0f / std::pow(state->getStepIdx() + 10, 2.0f);
                  if (std::uniform_real_distribution<float>(0, 1.0f)(rng) < x) {
                    result.at(offset + i).bestAction = randint(state->GetLegalActions().size());
                    fmt::printf("at state '%s' - performing random move %s\n", state->history(), state->actionDescription(state->GetLegalActions().at(result.at(offset + i).bestAction)));
                    gameState->validTournamentGame = false;
                  }
                }
              }
              offset += currentPlayerStates;
            }
          }

          std::vector<std::vector<float>> nextRnnState;
          if (&*playerGame_.at(playerIndex) != this) {
            if (devPlayer->rnnSeqlen()) {

              rnnState.clear();

              offset = 0;
              for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
                size_t currentPlayerIndex = statePlayerSize[pi].first;
                size_t currentPlayerStates = statePlayerSize[pi].second;
                for (size_t i = 0; i != currentPlayerStates; ++i) {
                  State* state = (State*)states.at(offset + i);
                  GameState* gameState =
                      actGameStates.at(currentPlayerIndex).at(i);

                  size_t slot =
                      gameState->playersReverseMap.at(currentPlayerIndex);

                  if (gameState->rnnState2.at(slot).empty()) {
                    auto shape = devPlayer->rnnStateSize();
                    int64_t rnnStateSize = shape.at(0) * shape.at(1);
                    size_t n = rnnStateSize * state->GetRawFeatureSize().at(1) *
                               state->GetRawFeatureSize().at(2);
                    gameState->rnnState2[slot].resize(n);
                  }

                  rnnState.push_back(std::move(gameState->rnnState2[slot]));

                  gameState->rnnStates.at(slot).push_back(rnnState.back());
                }
                offset += currentPlayerStates;
              }

              nextRnnState = devPlayer->nextRnnState(states, rnnState);

              offset = 0;
              for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
                size_t currentPlayerIndex = statePlayerSize[pi].first;
                size_t currentPlayerStates = statePlayerSize[pi].second;
                for (size_t i = 0; i != currentPlayerStates; ++i) {
                  GameState* gameState =
                      actGameStates.at(currentPlayerIndex).at(i);

                  size_t slot =
                      gameState->playersReverseMap.at(currentPlayerIndex);

                  gameState->rnnState2[slot] =
                      std::move(nextRnnState.at(offset + i));
                }
                offset += currentPlayerStates;
              }
            }

            offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerStates = statePlayerSize[pi].second;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                State* state = (State*)playerActStates.at(offset + i);

                auto& res = result.at(offset + i);

                state->forward(res.bestAction);
              }
              offset += currentPlayerStates;
            }

          } else {
            offset = 0;
            for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
              size_t currentPlayerIndex = statePlayerSize[pi].first;
              size_t currentPlayerStates = statePlayerSize[pi].second;
              for (size_t i = 0; i != currentPlayerStates; ++i) {
                GameState* gameState =
                    actGameStates.at(currentPlayerIndex).at(i);
                auto& res = result.at(offset + i);

                for (auto& x : gameState->playerState) {
                  if (x) {
                    x->forward(res.bestAction);
                  }
                }
              }
              offset += currentPlayerStates;
            }
          }

          auto& mctsOption = mctsPlayers[playerIndex]->option();

          {
            double rps = mctsPlayers.at(playerIndex)->rolloutsPerSecond();
            std::unique_lock<std::mutex> lkStats(mutexStats_);
            auto& stats_s = stats_["Rollouts per second"];
            std::get<0>(stats_s) += 1;
            std::get<1>(stats_s) += rps;
            std::get<2>(stats_s) += rps * rps;
          }

          // Distribute results to the correct player/state and store
          // features for training.
          offset = 0;
          for (size_t pi = 0; pi != statePlayerSize.size(); ++pi) {
            size_t currentPlayerIndex = statePlayerSize[pi].first;
            size_t currentPlayerStates = statePlayerSize[pi].second;
            for (size_t i = 0; i != currentPlayerStates; ++i) {
              State* state = (State*)states.at(offset + i);

              GameState* gameState = actGameStates.at(currentPlayerIndex).at(i);

              size_t slot = gameState->playersReverseMap.at(currentPlayerIndex);

              if (!gameState->rnnState.at(slot).empty()) {
                throw std::runtime_error("rnnState is not empty error");
              }
              gameState->rnnState[slot] =
                  std::move(result.at(offset + i).rnnState);

              if (gameState->canResign) {
                float value = result.at(offset + i).rootValue;
                if (value < -0.95f) {
                  if (++gameState->resignCounter.at(slot) >= 7) {
                    gameState->resigned = int(slot);
                  }
                } else {
                  gameState->resignCounter.at(slot) = 0;
                }
                int opponent =
                    (gameState->playersReverseMap.at(currentPlayerIndex) + 1) %
                    2;
                if (value > 0.95f) {
                  ++gameState->resignCounter.at(opponent);
                } else {
                  gameState->resignCounter.at(opponent) = 0;
                }

                //                if (gameState->stepindex >= 40 && value >
                //                -0.05 && value < 0.05) {
                //                  ++gameState->drawCounter;
                //                  if (gameState->drawCounter >= 7) {
                //                    gameState->drawn = true;
                //                  }
                //                } else {
                //                  gameState->drawCounter = 0;
                //                }
              }
              bool saveForTraining = true;
              if (mctsOption.randomizedRollouts &&
                  result.at(offset + i).rollouts <
                      mctsOption.numRolloutPerThread * 1.5f) {
                saveForTraining = false;
              }
              if (saveForTraining) {
                torch::Tensor feat = getFeatureInTensor(*state);
                auto [policy, policyMask] =
                    getPolicyInTensor(*state, result.at(offset + i).mctsPolicy);
                gameState->feat.at(slot).push_back(feat);
                if (false) {
                  policy.zero_();
                  auto action = state->GetLegalActions().at(
                      result.at(offset + i).bestAction);
                  policy[action.GetX()][action.GetY()][action.GetZ()] = 1;
                }
                gameState->pi.at(slot).push_back(policy);
                gameState->piMask.at(slot).push_back(policyMask);
              }

              gameState->history.emplace_back();
              auto& h = gameState->history.back();
              h.turn = slot;
              h.move = result.at(offset + i).bestAction;
              h.value = result.at(offset + i).rootValue;
              h.featurized = saveForTraining;
              h.shortFeat = getRawFeatureInTensor(*state);

              if (gameState->useNoiseModel.at(slot)) {
                torch::Tensor t = torch::zeros(state->GetActionSize());
                torch::Tensor mask = torch::zeros(state->GetActionSize());

                auto action = state->GetLegalActions().at(h.move);

                t[action.GetX()][action.GetY()][action.GetZ()] = 1;
                for (auto& v : state->GetLegalActions()) {
                  mask[v.GetX()][v.GetY()][v.GetZ()] = 1;
                }

                h.noiseTarget = t;
                h.noiseTargetMask = mask;
              }

              if (gameState->rewindCount == 0) {
                players_[currentPlayerIndex]->recordMove(state);
              }
#ifdef OPENBW_UI
              auto x =
                  state->GetLegalActions().at(result.at(offset + i).bestAction);
              // printf("%s\n", state->stateDescription().c_str());
              printf("move player %d - %d %d %d  (value %g)\n", slot, x.GetX(),
                     x.GetY(), x.GetZ(), result.at(offset + i).rootValue);
#endif

              state->forward(result.at(offset + i).bestAction);

              state->render();

              if (saveForTraining) {
                gameState->reward[slot].push_back(state->getReward(slot));
              }

              auto now = std::chrono::steady_clock::now();
              double elapsed =
                  std::chrono::duration_cast<
                      std::chrono::duration<double, std::ratio<1, 1>>>(
                      now - gameState->prevMoveTime)
                      .count();
              gameState->prevMoveTime = now;

              {
                std::unique_lock<std::mutex> lkStats(mutexStats_);
                auto& stats_s = stats_["Move Duration (seconds)"];
                std::get<0>(stats_s) += 1;
                std::get<1>(stats_s) += elapsed;
                std::get<2>(stats_s) += elapsed * elapsed;
              }

              if (gameState->justRewound) {
                float flip =
                    gameState->justRewoundToNegativeValue ? -1.0f : 1.0f;
                if (h.value * flip < 0.0f) {
                  // fmt::printf("rewound turned negative, rewinding more!\n");
                  rewind(
                      gameState, slot, gameState->justRewoundToNegativeValue);
                } else {
                  gameState->justRewound = false;
                }
              }

              // fmt::printf("game in progress: %s\n", state->history());
            }
            offset += currentPlayerStates;
          }

          states.clear();
        }
      };

      if (alignPlayers) {
        size_t bestPlayerIdx = 0;
        size_t bestPlayerIdxSize = 0;
        for (size_t playerIdx = 0; playerIdx != actStates.size(); ++playerIdx) {
          auto& states = actStates[playerIdx];
          if (states.size() > bestPlayerIdxSize) {
            bestPlayerIdxSize = states.size();
            bestPlayerIdx = playerIdx;
          }
        }
        actForPlayer(bestPlayerIdx);
      } else {
        for (size_t playerIdx = 0; playerIdx != actStates.size(); ++playerIdx) {
          actForPlayer(playerIdx);
        }
      }
    }
  } else {

    // Warm up model. This can take several seconds, so do it before we start
    // time counting.
    for (auto& v : players_) {
      auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(v);
      if (mctsPlayer && mctsPlayer->option().totalTime) {
        std::cout << "Warming up model.\n";
        for (int i = 0; i != 10; ++i) {
          mctsPlayer->calculateValue(*state_);
        }
      }
    }

    int64_t gameCount = 0;
#ifdef DEBUG_GAME
    std::thread::id thread_id = std::this_thread::get_id();
#endif
    while ((numEpisode < 0 || gameCount < numEpisode) && !terminate_) {
      if (terminate_) {
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ": terminating, "
                  << "game " << this << ", " << gameCount << " / " << numEpisode
                  << " games played" << std::endl;
#endif
        break;
      }
// std::unique_lock<std::mutex> lk(terminateMutex_);
#ifdef DEBUG_GAME
      std::cout << "Thread " << thread_id << ", game " << this
                << ": not terminating - run another iteration. " << std::endl;
#endif
      bool aHuman =
          std::any_of(players_.begin(), players_.end(),
                      [](const std::shared_ptr<mcts::Player>& player) {
                        return player->isHuman();
                      });
      if (aHuman && state_->stochasticReset()) {
        std::string line;
        std::cout << "Random outcome ?" << std::endl;
        std::cin >> line;
        state_->forcedDice = std::stoul(line, nullptr, 0);
      }
      auto randint = [&](int n) {
        return std::uniform_int_distribution<int>(0, n - 1)(rng_);
      };
      reset();
      int stepindex = 0;
      auto start = std::chrono::steady_clock::now();
      resignCounter_.resize(players_.size());
      canResign_ = !evalMode && players_.size() == 2 && randint(6) != 0;
      resigned_ = -1;
      while (!state_->terminated() && resigned_ == -1) {
        stepindex += 1;
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ", game " << this << ": step "
                  << stepindex << std::endl;
#endif
        step();
        if (isInSingleMoveMode_) {
          std::cout << lastMctsValue_ << "\n";
          state_->printLastAction();
          std::exit(0);
        }
        if (printMoves_) {
          std::cout << "MCTS value: " << lastMctsValue_ << "\n";
          std::cout << "Made move: " << state_->lastMoveString() << std::endl;
        }
      }
      auto end = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
      {
        std::unique_lock<std::mutex> lkStats(mutexStats_);
        auto& stats_steps = stats_["Game Duration (steps)"];
        std::get<0>(stats_steps) += 1;
        std::get<1>(stats_steps) += stepindex;
        std::get<2>(stats_steps) += stepindex * stepindex;
        auto& stats_s = stats_["Game Duration (seconds)"];
        std::get<0>(stats_s) += 1;
        std::get<1>(stats_s) += elapsed;
        std::get<2>(stats_s) += elapsed * elapsed;
      }
#ifdef DEBUG_GAME
      std::cout << "Thread " << thread_id << ", game " << this << ": game "
                << gameCount << " / " << numEpisode << " ended; " << stepindex
                << " steps, " << (static_cast<float>(stepindex) / elapsed)
                << " steps per second" << std::endl;
#endif
      if (!lastAction_.empty() && aHuman) {
        std::cout << "\n#Last Action: " << lastAction_ << "\n\n";
        state_->printCurrentBoard();
      }
      if (std::any_of(players_.begin(), players_.end(),
                      [](const std::shared_ptr<mcts::Player>& player) {
                        return player->isTP();
                      })) {
        state_->errPrintCurrentBoard();
      } /*else {
        state_->printCurrentBoard();
      }*/

      if (resigned_ != -1) {
        for (size_t idx = 0; idx != players_.size(); ++idx) {
          result_.at(idx) = int(idx) == resigned_ ? 1 : -1;
        }
      } else {
        result_[0] = state_->getReward(0);
        if (players_.size() > 1) {
          result_[1] = state_->getReward(1);
        }
      }

      if (!evalMode) {
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ", game " << this
                  << ": sending trajectory... " << std::endl;
#endif
        setReward(*state_, resigned_);
        sendTrajectory();
#ifdef DEBUG_GAME
        std::cout << "Thread " << thread_id << ", game " << this
                  << ": trajectory sent... " << std::endl;
#endif
      }

      ++gameCount;
    }
#ifdef DEBUG_GAME
    std::cout << "Thread " << thread_id << ", game " << this
              << ": exiting main loop" << std::endl;
#endif
  }
}

std::optional<int> Game::parseSpecialAction(const std::string& str) {
  if (str == "-1" || str == "undo" || str == "u") {
    std::cout << "Undoing the last move\n";
    state_->undoLastMoveForPlayer(state_->getCurrentPlayer());
    return -1;
  } else if (str == "exit") {
    std::exit(0);
  } else if (str == "m" || str == "manual") {
    bool resume = false;
    auto playerString = [&](int index) {
      std::string str;
      auto& player = players_.at(index);
      if (std::dynamic_pointer_cast<mcts::MctsPlayer>(player)) {
        str += "MctsPlayer";
      } else if (std::dynamic_pointer_cast<HumanPlayer>(player)) {
        str += "HumanPlayer";
      } else {
        str += typeid(player).name();
      }
      return str;
    };
    auto specialAction = [&](const std::string& str) -> std::optional<int> {
      if (str == "singlemovemode" || str == "sm") {
        isInSingleMoveMode_ = true;
        return -1;
      } else if (str.substr(0, 3) == "set") {
        state_->setStateFromStr(str.substr(4));
        return -1;
      } else if (str == "r" || str == "reset") {
        state_->reset();
        return -1;
      } else if (str == "u" || str == "undo") {
        state_->undoLastMove();
        return -1;
      } else if (str == "c" || str == "continue") {
        resume = true;
        return -1;
      } else if (str == "swap") {
        std::next_permutation(players_.begin(), players_.end());
        for (size_t i = 0; i != players_.size(); ++i) {
          std::cout << "Player " << i << " is now " << playerString(i) << "\n";
        }
        return -1;
      } else if (str == "printmoves") {
        printMoves_ = true;
        return -1;
      } else if (str == "printvalue") {
        auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(
            players_.at(state_->getCurrentPlayer()));
        if (!mctsPlayer) {
          for (auto& v : players_) {
            mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(v);
            if (mctsPlayer) {
              break;
            }
          }
        }
        if (mctsPlayer) {
          std::cout << "NN Value: " << mctsPlayer->calculateValue(*state_)
                    << "\n";
        } else {
          std::cout << "NN Value: 0\n";
        }
      }
      return std::nullopt;
    };
    std::cout
        << "\nEntering moves manually. Enter 'r' or 'reset' to reset the "
           "board, 'u' or 'undo' to undo the last move, 'c' or 'continue' to "
           "continue play, or 'swap' to swap the turn order of the players\n\n";
    while (!state_->terminated()) {
      int index = -1;
      while (index == -1) {
        std::cout << "Enter a move for player " << state_->getCurrentPlayer()
                  << " (" << playerString(state_->getCurrentPlayer()) << ")\n";
        index = state_->humanInputAction(specialAction);
        if (resume) {
          return -1;
        }
      }

      if (!lastAction_.empty()) {
        std::cout << "\nLast Action: " << lastAction_ << "\n\n";
      }
      std::cout << " applying action... " << std::endl;
      auto action = state_->GetLegalActions().at(index);
      lastAction_ = state_->actionDescription(action);
      if (!state_->isStochastic()) {
        state_->forward(action.GetIndex());
      } else {
        // auto backup_state = state_->clone();
        std::string line;
        std::cout << "Random outcome ?" << std::endl;
        std::cin >> line;
        state_->forcedDice = std::stoul(line, nullptr, 0);
        state_->forward(action.GetIndex());
      }
    }
    return -1;
  }
  return std::nullopt;
}

/* virtual */ tube::EnvThread::Stats Game::get_stats() {
  std::unique_lock<std::mutex> lkStats(mutexStats_);
  return stats_;
}

void Game::step() {
  auto playerIdx = state_->getCurrentPlayer();
  auto& player = players_.at(playerIdx);
  // std::cout << "board" << std::endl;
  // state_->printCurrentBoard();
  if (player->isTP()) {
    // auto TPplayer = std::dynamic_pointer_cast<TPPlayer>(player);
    assert(!state_->isStochastic());
    auto index = state_->TPInputAction();
    auto action = state_->GetLegalActions().at(index);
    lastAction_ = state_->actionDescription(action);
    state_->forward(index);
  } else if (player->isHuman()) {
    if (!hasPrintedHumanHelp_) {
      std::cout << "\nEnter a move for the human player. Enter 'u' or 'undo' "
                   "to undo your previous move, 'm' or 'manual' to enter moves "
                   "manually for all players.\n\n";
      hasPrintedHumanHelp_ = true;
    }
    auto humanPlayer = std::dynamic_pointer_cast<HumanPlayer>(player);
    if (!lastAction_.empty()) {
      std::cout << "\nLast Action: " << lastAction_ << "\n\n";
    }

    std::cout << "History: " << state_->history() << "\n";

    int index = state_->humanInputAction(
        std::bind(&Game::parseSpecialAction, this, std::placeholders::_1));
    if (index == -1) {
      return step();
    }
    std::cout << " applying action... " << std::endl;
    auto action = state_->GetLegalActions().at(index);
    lastAction_ = state_->actionDescription(action);
    if (!state_->isStochastic()) {
      state_->forward(action.GetIndex());
    } else {
      // auto backup_state = state_->clone();
      std::string line;
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
      state_->forward(action.GetIndex());
    }
  } else {
    auto mctsPlayer = std::dynamic_pointer_cast<mcts::MctsPlayer>(player);
    auto rnnShape = mctsPlayer->rnnStateSize();
    int64_t rnnStateSize = 0;
    if (!rnnShape.empty()) {
      rnnStateSize = rnnShape.at(0) * rnnShape.at(1);
    }
    if (rnnStateSize) {
      if (rnnState_.size() <= playerIdx) {
        rnnState_.resize(playerIdx + 1);
      }
      if (rnnState_.at(playerIdx).empty()) {
        size_t n = rnnStateSize * state_->GetRawFeatureSize().at(1) *
                   state_->GetRawFeatureSize().at(2);
        rnnState_[playerIdx].resize(n);
      }
    }
    mcts::MctsResult result;
    if (rnnStateSize) {
      result = mctsPlayer->actMcts(*state_, rnnState_.at(playerIdx));
      rnnState_.at(playerIdx) = std::move(result.rnnState);
    } else {
      result = mctsPlayer->actMcts(*state_);
    }
    lastMctsValue_ = result.rootValue;

    if (canResign_ && result.rootValue < -0.95f) {
      if (++resignCounter_.at(playerIdx) >= 2) {
        resigned_ = playerIdx;
      }
    }

    // store feature for training
    if (!evalMode) {
      torch::Tensor feat = getFeatureInTensor(*state_);
      auto [policy, policyMask] = getPolicyInTensor(*state_, result.mctsPolicy);
      feature_[playerIdx].pushBack(std::move(feat));
      pi_[playerIdx].pushBack(std::move(policy));
      piMask_[playerIdx].pushBack(std::move(policyMask));
    }

    // std::cout << ">>>>actual act" << std::endl;
    _Action action = state_->GetLegalActions().at(result.bestAction);
    lastAction_ = state_->actionDescription(action);
    bool noHuman =
        std::none_of(players_.begin(), players_.end(),
                     [](const std::shared_ptr<mcts::Player>& player) {
                       return player->isHuman();
                     });
    if (!state_->isStochastic()) {
      if (!noHuman) {
        std::cout << "Performing action "
                  << state_->performActionDescription(
                         state_->GetLegalActions().at(result.bestAction))
                  << "\n";
      }
    } else if (!noHuman) {
      std::string line;
      std::cout << "Performing action "
                << state_->performActionDescription(
                       state_->GetLegalActions().at(result.bestAction))
                << "\n";
      std::cout << "Random outcome ?" << std::endl;
      std::cin >> line;
      state_->forcedDice = std::stoul(line, nullptr, 0);
    }
    state_->forward(result.bestAction);
  }
}

void Game::setReward(const State& state, int resigned) {
  for (int i = 0; i < (int)players_.size(); ++i) {
    assert(v_[i].len() <= pi_[i].len() && pi_[i].len() == feature_[i].len());
    while (v_[i].len() < pi_[i].len()) {
      torch::Tensor reward = torch::zeros({1}, torch::kFloat32);
      reward[0] = resigned == -1 ? state.getReward(i) : i == resigned ? -1 : 1;
      v_[i].pushBack(std::move(reward));
    }
  }
}

void Game::sendTrajectory() {
  for (int i = 0; i < (int)players_.size(); ++i) {
    assert(v_[i].len() == pi_[i].len() && pi_[i].len() == feature_[i].len());
    assert(pi_[i].len() == piMask_[i].len());
    int errcode;
    while (prepareForSend(i)) {
      // ignore error codes from the dispatcher
      errcode = dispatchers_[i].dispatchNoReply();
      switch (errcode) {
      case tube::Dispatcher::DISPATCH_ERR_DC_TERM:
#ifdef DEBUG_GAME
        std::cout << "game " << this << ", sendTrajectory: "
                  << "attempt to dispatch through"
                  << " a terminated data channel " << std::endl;
#endif
        break;
      case tube::Dispatcher::DISPATCH_ERR_NO_SLOT:
#ifdef DEBUG_GAME
        std::cout << "game " << this << ": sendTrajectory: "
                  << "no slots available to dispatch" << std::endl;
#endif
        break;
      case tube::Dispatcher::DISPATCH_NOERR:
        break;
      }
    }
    assert(v_[i].len() == 0);
    assert(pi_[i].len() == 0);
    assert(piMask_[i].len() == 0);
    assert(feature_[i].len() == 0);
  }
}

bool Game::prepareForSend(int playerId) {
  if (feature_[playerId].prepareForSend()) {
    bool b = pi_[playerId].prepareForSend();
    b &= piMask_[playerId].prepareForSend();
    b &= v_[playerId].prepareForSend();
    if (predictEndState + predictNStates) {
      b &= predictPi_[playerId].prepareForSend();
      b &= predictPiMask_[playerId].prepareForSend();
    }
    if (!rnnInitialState_.empty()) {
      b &= rnnInitialState_[playerId].prepareForSend();
      b &= rnnStateMask_[playerId].prepareForSend();
    }
    if (!b) {
      throw std::runtime_error("prepareForSend mismatch 1");
    }
    return true;
  }
  bool b = pi_[playerId].prepareForSend();
  b |= piMask_[playerId].prepareForSend();
  b |= v_[playerId].prepareForSend();
  if (predictEndState + predictNStates) {
    b |= predictPi_[playerId].prepareForSend();
    b |= predictPiMask_[playerId].prepareForSend();
  }
  if (!rnnInitialState_.empty()) {
    b |= rnnInitialState_[playerId].prepareForSend();
    b |= rnnStateMask_[playerId].prepareForSend();
  }
  if (b) {
    throw std::runtime_error("prepareForSend mismatch 2");
  }
  return false;
}
