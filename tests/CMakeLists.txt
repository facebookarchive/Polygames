cmake_minimum_required( VERSION 3.3 )
project( polygames-tests )
set(CMAKE_CXX_STANDARD 17)

execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package( PythonInterp 3.7 REQUIRED )
find_package( PythonLibs 3.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

find_package (Threads)

find_package( GTest REQUIRED )
include_directories( ${GTEST_INCLUDE_DIRS} )

include_directories(
 ../games
 ../torchRL
 ../torchRL/third_party/fmt/include
 ../torchRL/tube/src_cpp
 )

add_executable( polygames-tests
 ../core/game.cc
 ../core/state.cc
 ../torchRL/mcts/mcts.cc
 ../torchRL/mcts/node.cc
 ../torchRL/tube/src_cpp/data_channel.cc
 ../torchRL/tube/src_cpp/replay_buffer.cc
 tests.cc
 )

target_link_libraries( polygames-tests 
    ${CMAKE_THREAD_LIBS_INIT}
    ${GTEST_LIBRARIES}
    ${TORCH_LIBRARIES}
    )

install (TARGETS polygames-tests DESTINATION bin)

