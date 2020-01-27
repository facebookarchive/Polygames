CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(csp_vkms)

set(CMAKE_CXX_STANDARD 17)

if (${Torch_FOUND})
  include_directories(${TORCH_INCLUDE_DIRS})
else()
  find_package(Torch REQUIRED)
  include_directories(${TORCH_INCLUDE_DIRS})
endif()

find_package( PythonInterp 3.7 REQUIRED )
find_package( PythonLibs 3.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

set(CORE_DIR ../../core)
set(TORCHRL_DIR ../../torchRL)

include_directories(
  ${CORE_DIR}
  ${TORCHRL_DIR}
)

set(TORCHRL_SRC
  ${TORCHRL_DIR}/mcts/mcts.cc
  ${TORCHRL_DIR}/mcts/node.cc
  ${TORCHRL_DIR}/tube/src_cpp/data_channel.cc
  ${TORCHRL_DIR}/tube/src_cpp/replay_buffer.cc
)

add_executable(benchmark_csp_vkms
  ${CORE_DIR}/game.cc
  ${CORE_DIR}/state.cc
  ${TORCHRL_SRC}
  ../../games/minesweeper.cc
  csp_vkms.cc
)

target_link_libraries(benchmark_csp_vkms
  ${CMAKE_THREAD_LIBS_INIT}
  ${TORCH_LIBRARIES}
  fmt
)

