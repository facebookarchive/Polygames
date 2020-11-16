/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>

#include "actor.h"
#include "common/threads.h"
#include "forward_player.h"
#include "game.h"
#include "model_manager.h"

namespace py = pybind11;

using namespace core;

PYBIND11_MODULE(polygames, m) {

  m.def("init_threads", &threads::init);

  py::class_<Game, tube::EnvThread, std::shared_ptr<Game>>(m, "Game")
      .def(py::init<std::string, std::vector<std::string>, int, int, bool, bool, bool, bool, bool, int,
                    int, bool, int, int, bool, int>())
      .def("add_player", &Game::addPlayer, py::keep_alive<1, 2>())
      .def("add_eval_player", &Game::addEvalPlayer)
      .def("add_human_player", &Game::addHumanPlayer)
      .def("add_tp_player", &Game::addTPPlayer)
      .def("get_raw_feat_size", &Game::getRawFeatSize)
      .def("get_feat_size", &Game::getFeatSize)
      .def("is_one_player_game", &Game::isOnePlayerGame)
      .def("set_features", &Game::setFeatures)
      .def("get_action_size", &Game::getActionSize)
      .def("get_result", &Game::getResult);

  py::class_<Actor, std::shared_ptr<Actor>>(m, "Actor")
      .def(py::init<std::shared_ptr<tube::DataChannel>,
                    const std::vector<int64_t>&,  // featSize
                    const std::vector<int64_t>&,  // actionSize
                    const std::vector<int64_t>&,  // rnnStateSize
                    int,                          // rnnSeqlen
                    bool,                         // logitValue
                    bool,                         // useValue
                    bool,                         // usePolicy
                    std::shared_ptr<ModelManager>>());

  py::class_<HumanPlayer, std::shared_ptr<HumanPlayer>>(m, "HumanPlayer")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>());

  py::class_<TPPlayer, std::shared_ptr<TPPlayer>>(m, "TPPlayer")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>());

  py::class_<ActorPlayer, std::shared_ptr<ActorPlayer>>(m, "ActorPlayer")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>())
      .def("set_actor", &ForwardPlayer::setActor, py::keep_alive<1, 2>())
      .def("set_name", &ForwardPlayer::setName);

  py::class_<ForwardPlayer, ActorPlayer, std::shared_ptr<ForwardPlayer>>(
      m, "ForwardPlayer")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>());

  py::class_<ModelManager, std::shared_ptr<ModelManager>>(m, "ModelManager")
      .def(py::init<int, const std::string&, int, int, const std::string&, int,
                    int>())
      .def("get_train_channel", &ModelManager::getTrainChannel)
      .def("get_act_channel", &ModelManager::getActChannel)
      .def("update_model", &ModelManager::updateModel)
      .def("buffer_size", &ModelManager::bufferSize)
      .def("buffer_full", &ModelManager::bufferFull)
      .def("buffer_num_sample", &ModelManager::bufferNumSample)
      .def("buffer_num_add", &ModelManager::bufferNumAdd)
      .def("sample", &ModelManager::sample)
      .def("start", &ModelManager::start)
      .def("test_act", &ModelManager::testAct)
      .def("set_is_tournament_opponent", &ModelManager::setIsTournamentOpponent)
      .def("add_tournament_model", &ModelManager::addTournamentModel)
      .def("set_dont_request_model_updates",
           &ModelManager::setDontRequestModelUpdates)
      .def("start_server", &ModelManager::startServer)
      .def("start_client", &ModelManager::startClient)
      .def("start_replay_buffer_server", &ModelManager::startReplayBufferServer)
      .def("start_replay_buffer_client", &ModelManager::startReplayBufferClient)
      .def("remote_sample", &ModelManager::remoteSample)
      .def("set_find_batch_size_max_ms", &ModelManager::setFindBatchSizeMaxMs)
      .def("set_find_batch_size_max_bs", &ModelManager::setFindBatchSizeMaxBs);

  py::class_<SampleResult>(m, "SampleResult").def("get", &SampleResult::get);
}
