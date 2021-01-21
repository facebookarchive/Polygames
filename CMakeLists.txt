CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(polygames)

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE RelWithDebInfo)
# endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsized-deallocation -O3 -ffast-math")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch 1.2 REQUIRED)
message(STATUS "Torch VERSION: ${Torch_VERSION}")
add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})

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

add_subdirectory(src)

# add Minesweeper benchmarks
add_subdirectory(src/games/minesweeper_csp_vkms)

# tests
add_executable(test_state src/core/test_state.cc src/core/state.cc)
target_link_libraries(test_state PUBLIC _tube _mcts _games ${JNI_LIBRARIES})

enable_testing()

add_test(NAME test_replay_buffer
    COMMAND ${PYTHON_EXECUTABLE} -m test_replay_buffer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)

set_tests_properties(test_replay_buffer
    PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}")

