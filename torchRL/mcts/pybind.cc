/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>

#include "mcts/mcts.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts, m) {
  using namespace mcts;

  py::class_<MctsPlayer, std::shared_ptr<MctsPlayer>>(m, "MctsPlayer")
      .def(py::init<const MctsOption&>(),
           py::call_guard<py::gil_scoped_release>())
      .def("add_actor", &MctsPlayer::addActor, py::keep_alive<1, 2>())
      .def("set_name", &MctsPlayer::setName);

  // to enforce subclass of Actor has to use shared_ptr for ownership
  py::class_<Actor, std::shared_ptr<Actor>>(m, "Actor");

  py::class_<MctsOption>(m, "MctsOption")
      .def(py::init<>())
      .def(py::init<const MctsOption&>())
      .def_readwrite("use_mcts", &MctsOption::useMcts)
      .def_readwrite("puct", &MctsOption::puct)
      .def_readwrite("sample_before_step_idx", &MctsOption::sampleBeforeStepIdx)
      .def_readwrite("num_rollout_per_thread", &MctsOption::numRolloutPerThread)
      .def_readwrite("seed", &MctsOption::seed)
      .def_readwrite("virtual_loss", &MctsOption::virtualLoss)
      .def_readwrite("store_state_in_node", &MctsOption::virtualLoss)
      .def_readwrite("use_value_prior", &MctsOption::useValuePrior)
      .def_readwrite("time_ratio", &MctsOption::timeRatio)
      .def_readwrite("total_time", &MctsOption::totalTime)
      .def_readwrite("randomized_rollouts", &MctsOption::randomizedRollouts)
      .def_readwrite("sampling_mcts", &MctsOption::samplingMcts)
      .def_readwrite(
          "move_select_use_mcts_value", &MctsOption::moveSelectUseMctsValue);
}
