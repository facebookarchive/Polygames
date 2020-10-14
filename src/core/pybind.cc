/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>

#include "actor.h"
#include "game.h"
#include "tube/src_cpp/channel_assembler.h"

namespace py = pybind11;

PYBIND11_MODULE(polygames, m) {

  py::class_<Game, tube::EnvThread, std::shared_ptr<Game>>(m, "Game")
      //      .def(py::init<std::string, int, int, bool>())
      .def(py::init<std::string, int, int, bool, bool, bool, bool, int, int,
                    bool, int>())
      .def("add_player", &Game::addPlayer, py::keep_alive<1, 2>())
      .def("add_eval_player", &Game::addEvalPlayer)
      .def("add_human_player", &Game::addHumanPlayer)
      .def("add_tp_player", &Game::addTPPlayer)
      .def("get_feat_size", &Game::getFeatSize)
      .def("get_raw_feat_size", &Game::getRawFeatSize)
      .def("is_one_player_game", &Game::isOnePlayerGame)
      .def("set_features", &Game::setFeatures)
      .def("get_action_size", &Game::getActionSize)
      .def("get_result", &Game::getResult);

  py::class_<Actor, mcts::Actor, std::shared_ptr<Actor>>(m, "Actor")
      .def(py::init<std::shared_ptr<tube::DataChannel>,
                    const std::vector<int64_t>&,  // featSize
                    const std::vector<int64_t>&,  // actionSize
                    bool,                         // useValue
                    bool,                         // usePolicy
                    std::shared_ptr<tube::ChannelAssembler>>());

  py::class_<HumanPlayer, std::shared_ptr<HumanPlayer>>(m, "HumanPlayer")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>());

  py::class_<TPPlayer, std::shared_ptr<TPPlayer>>(m, "TPPlayer")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>());
}
