/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "channel_assembler.h"
#include "context.h"
#include "data_channel.h"

// for testing
#include "test/test_producer.h"

namespace py = pybind11;
using namespace tube;

PYBIND11_MODULE(tube, m) {
  py::class_<DataChannel, std::shared_ptr<DataChannel>>(m, "DataChannel")
      .def(py::init<std::string, int, int>())
      .def_readonly("name", &DataChannel::name)
      .def("terminate", &DataChannel::terminate)
      .def("get_input",
           &DataChannel::getInput,
           py::call_guard<py::gil_scoped_release>())
      .def("set_reply",
           &DataChannel::setReply,
           py::call_guard<py::gil_scoped_release>());

  using BufferState = ReplayBuffer::SerializableState;

  py::class_<ChannelAssembler, std::shared_ptr<ChannelAssembler>>(
      m, "ChannelAssembler")
      .def(py::init<int,
                    int,
                    const std::vector<std::string>&,
                    int,
                    int,
                    const std::string&,
                    int,
                    int>())
      .def("get_train_channel", &ChannelAssembler::getTrainChannel)
      .def("get_act_channels", &ChannelAssembler::getActChannels)
      .def("update_model", &ChannelAssembler::updateModel)
      .def("buffer_size", &ChannelAssembler::bufferSize)
      .def("buffer_full", &ChannelAssembler::bufferFull)
      .def("buffer_num_sample", &ChannelAssembler::bufferNumSample)
      .def("buffer_num_add", &ChannelAssembler::bufferNumAdd)
      .def_property("buffer",
                    &ChannelAssembler::getBufferState,
                    &ChannelAssembler::setBufferState,
                    py::return_value_policy::move
                    // getter
                    // py::cpp_function(
                    //[](ChannelAssembler& casm) {
                    //    return casm.getBufferState();
                    //},//,
                    // py::return_value_policy::move
                    //),
                    // setter
                    // py::cpp_function(
                    //[](ChannelAssember& casm,
                    //   const BufferState& bufferState) {
                    //    casm.setBufferState(bufferState);
                    //}
                    //)
                    )
      .def("sample", &ChannelAssembler::sample)
      .def("start", &ChannelAssembler::start)
      .def("test_act", &ChannelAssembler::testAct)
      .def("set_is_tournament_opponent",
           &ChannelAssembler::setIsTournamentOpponent)
      .def("add_tournament_model", &ChannelAssembler::addTournamentModel)
      .def("set_dont_request_model_updates",
           &ChannelAssembler::setDontRequestModelUpdates)
      .def("start_server", &ChannelAssembler::startServer)
      .def("start_client", &ChannelAssembler::startClient)
      .def("start_replay_buffer_server",
           &ChannelAssembler::startReplayBufferServer)
      .def("start_replay_buffer_client",
           &ChannelAssembler::startReplayBufferClient)
      .def("remote_sample", &ChannelAssembler::remoteSample)
      .def("loss", &ChannelAssembler::loss);

  py::class_<ChannelAssembler::SampleResult>(m, "SampleResult")
      .def("get", &ChannelAssembler::SampleResult::get);

  py::class_<BufferState>(m, "ReplayBuffer")
      .def(py::pickle(
          // __getstate__
          [](const BufferState& bufferState) {
            return py::make_tuple(bufferState.capacity,
                                  bufferState.size,
                                  bufferState.nextIdx,
                                  bufferState.rngState,
                                  bufferState.buffer);
          },
          // __setstate__
          [](const py::tuple& t) {
            constexpr size_t kExpectedTupleSize = 5;
            if (t.size() != kExpectedTupleSize) {
              std::ostringstream oss;
              oss << "Error unpickling ReplayBuffer: expected "
                  << kExpectedTupleSize << " elements, got " << t.size();
              throw std::runtime_error(oss.str());
            }
            using RawBuffer = std::unordered_map<std::string, torch::Tensor>;
            BufferState bufferState;
            bufferState.capacity = t[0].cast<int>();
            bufferState.size = t[1].cast<int>();
            bufferState.nextIdx = t[2].cast<int>();
            bufferState.rngState = t[3].cast<std::string>();
            bufferState.buffer = t[4].cast<RawBuffer>();
            return bufferState;
          }))
      .def_readonly("size", &BufferState::size)
      .def_readonly("capacity", &BufferState::capacity)
      .def_property_readonly("is_full", [](const BufferState& bufferState) {
        return bufferState.size == bufferState.capacity;
      });

  py::class_<EnvThread, std::shared_ptr<EnvThread>>(m, "EnvThread");

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("push_env_thread", &Context::pushEnvThread, py::keep_alive<1, 2>())
      .def("start", &Context::start)
      .def("terminated", &Context::terminated)
      .def("get_stats_str", &Context::getStatsStr);

  // for testing
  py::class_<ProducerThread, EnvThread, std::shared_ptr<ProducerThread>>(
      m, "ProducerThread")
      .def(py::init<int, std::shared_ptr<DataChannel>>());

  py::class_<DualDispatchThread,
             EnvThread,
             std::shared_ptr<DualDispatchThread>>(m, "DualDispatchThread")
      .def(py::init<int,
                    int,
                    std::shared_ptr<DataChannel>,
                    std::shared_ptr<DataChannel>>());
}
