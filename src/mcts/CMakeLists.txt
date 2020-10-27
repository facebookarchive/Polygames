cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# lib for other c++ programs
add_library(_mcts
  node.cc
  mcts.cc
  storage.cc
)
target_link_libraries(_mcts PUBLIC pthread)
target_include_directories(_mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_include_directories(_mcts PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_mcts PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_mcts PUBLIC ${TORCH_LIBRARIES} _tube)

# pybind lib
pybind11_add_module(mcts pybind.cc)
target_link_libraries(mcts PUBLIC libpolygames)

# tests
#add_executable(test_mcts test.cc)
#target_link_libraries(test_mcts PUBLIC _mcts)

