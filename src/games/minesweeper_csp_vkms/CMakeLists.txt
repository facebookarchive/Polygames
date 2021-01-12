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

set(SRC_DIR ../..)

add_executable(benchmark_csp_vkms
  ${SRC_DIR}/core/state.cc
  ${SRC_DIR}/games/minesweeper.cc
  csp_vkms.cc
)
target_include_directories(benchmark_csp_vkms PUBLIC ${SRC_DIR})
target_link_libraries(benchmark_csp_vkms
  ${CMAKE_THREAD_LIBS_INIT}
  ${TORCH_LIBRARIES}
  fmt
)
