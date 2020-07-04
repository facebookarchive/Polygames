CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(choufleur)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE RelWithDebInfo)
# endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-register -fPIC -march=native -O3 -Wfatal-errors")

OPTION(PYTORCH12 "Is PyTorch >= 1.2" OFF)
OPTION(PYTORCH15 "Is PyTorch >= 1.5" OFF)
IF(PYTORCH15)
    ADD_DEFINITIONS(-DPYTORCH15 -DPYTORCH12)
ELSEIF(PYTORCH12)
    ADD_DEFINITIONS(-DPYTORCH12)
ENDIF()

execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

find_package(Boost COMPONENTS system)
if( Boost_FOUND )
include_directories( ${Boost_INCLUDE_DIRS})
endif()

option(WITH_LUDII "Include LUDII support" ON)

if(WITH_LUDII)
  find_package(JNI)
  if (JNI_FOUND)
      include_directories( ${JNI_INCLUDE_DIRS})
  else()
      message(STATUS "Java not found, LUDII support will not be included")
      add_definitions(-DNO_JAVA)
  endif()
else()
  add_definitions(-DNO_JAVA)
endif()

message(STATUS "Adding PyTorch compilation flags: ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(torchRL/third_party/fmt/include)
add_subdirectory(torchRL/third_party/fmt)

# add_subdirectory(torchRL)
add_subdirectory(torchRL/third_party/pybind11)
add_subdirectory(torchRL/tube)
add_subdirectory(torchRL/mcts)

set(SOURCES
  games/gomoku_swap2.cc
  games/othello_opt.cc
  games/mastermind_state.cc
  games/amazons.cc
  games/breakthrough.cc
  games/chess.cc
  games/chinesecheckers.cc
  games/tristan_nogo.cc
  games/yinsh.cc
  games/minesweeper.cc
  games/weakschur/SchurMatrix.cpp
  games/weakschur/SchurVector.cpp
  games/weakschur/WeakSchur.cpp)

if (JNI_FOUND)
  list(APPEND SOURCES
    games/ludii/jni_utils.cc
    games/ludii/ludii_game_wrapper.cc
    games/ludii/ludii_state_wrapper.cc)
endif()

add_library(_games
  ${SOURCES})
target_include_directories(_games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/torchRL)
target_include_directories(_games SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_games PUBLIC ${PYTHON_INCLUDE_DIRS})

pybind11_add_module(polygames core/pybind.cc core/game.cc core/state.cc)

target_include_directories(polygames PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/torchRL)
target_link_libraries(polygames PUBLIC _tube _mcts _games)

if (JNI_FOUND)
  target_include_directories(_games PUBLIC ${JNI_INCLUDE_DIRS})
  target_link_libraries(polygames PUBLIC ${JNI_LIBRARIES})
endif()

# add Minesweeper benchmarks
add_subdirectory(games/minesweeper_csp_vkms)

# tests
add_executable(test_state core/test_state.cc core/state.cc)
target_link_libraries(test_state PUBLIC _tube _mcts _games ${JNI_LIBRARIES})

enable_testing()

add_test(NAME test_replay_buffer
    COMMAND ${PYTHON_EXECUTABLE} -m test_replay_buffer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)

set_tests_properties(test_replay_buffer
    PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}")

