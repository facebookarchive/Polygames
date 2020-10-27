 

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(third_party)
include_directories(third_party/fmt/include)
add_subdirectory(third_party/fmt)

# add_subdirectory(torchRL)
add_subdirectory(third_party/pybind11)
add_subdirectory(tube)
add_subdirectory(mcts)

file(GLOB _zstd_SOURCES third_party/zstd/lib/common/*.c third_party/zstd/lib/compress/*.c third_party/zstd/lib/decompress/*.c)
add_library(_zstd OBJECT ${_zstd_SOURCES})

target_include_directories(_zstd PUBLIC third_party/zstd/lib third_party/zstd/lib/common)

find_path(IBV_INCLUDE_DIR infiniband/verbs.h)
find_library(IBV_LIBRARY ibverbs)

add_library(libpolygames SHARED "")

add_library(_distributed OBJECT
  distributed/network.cc
  distributed/distributed.cc
)
target_include_directories(_distributed SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

if (IBV_INCLUDE_DIR AND IBV_LIBRARY)
  message(STATUS "Found ibverbs: ${IBV_INCLUDE_DIR}/infiniband/verbs.h ${IBV_LIBRARY}")
  target_sources(_distributed PRIVATE
    distributed/ib.cc)
  target_include_directories(_distributed SYSTEM PUBLIC ${IBV_INCLUDE_DIR})
  target_link_libraries(libpolygames PUBLIC ${IBV_LIBRARY})
else()
  message(STATUS "ibverbs NOT found, InfiniBand support will be disabled!")
  target_sources(_distributed PRIVATE
    distributed/rdma_nop.cc)
endif()

add_library(_common OBJECT
  common/thread_id.cc
  common/threads.cc
  )

set(_games_SOURCES
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
  list(APPEND _games_SOURCES
    games/ludii/jni_utils.cc
    games/ludii/ludii_game_wrapper.cc
    games/ludii/ludii_state_wrapper.cc)
endif()

add_library(_games
  ${_games_SOURCES})
target_include_directories(_games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/torchRL)
target_include_directories(_games SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_games PUBLIC ${PYTHON_INCLUDE_DIRS})

target_sources(libpolygames PRIVATE
  core/game.cc
  core/state.cc
  core/replay_buffer.cc
  core/model_manager.cc
  $<TARGET_OBJECTS:_zstd>
  $<TARGET_OBJECTS:_distributed>
  $<TARGET_OBJECTS:_common>
)

target_include_directories(libpolygames PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/torchRL)
target_link_libraries(libpolygames PUBLIC _tube _mcts _games)
set_target_properties(libpolygames PROPERTIES PREFIX "")

if (JNI_FOUND)
  target_include_directories(_games PUBLIC ${JNI_INCLUDE_DIRS})
  target_link_libraries(libpolygames PUBLIC ${JNI_LIBRARIES})
endif()


pybind11_add_module(polygames
  core/pybind.cc
)
target_link_libraries(polygames PUBLIC libpolygames)
