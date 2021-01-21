cmake_minimum_required( VERSION 3.3 )
project( polygames-tests )
set(CMAKE_CXX_STANDARD 17)

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
include_directories(${TORCH_INCLUDE_DIRS})

find_package( PythonInterp 3.7 REQUIRED )
find_package( PythonLibs 3.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )

find_package (Threads)

find_package( GTest REQUIRED )
include_directories( ${GTEST_INCLUDE_DIRS} )

find_package(JNI REQUIRED)
include_directories( ${JNI_INCLUDE_DIRS})

add_subdirectory(../src build-libpolygames)

add_executable( polygames-tests
 tests.cc

 # Include your tests here.
 connectfour-tests.cc
 havannah-state-tests.cc
 havannah-tests.cc
 hex-state-tests.cc
 hex-tests.cc
 ludii-game-tests.cc
 )

target_link_libraries( polygames-tests 
    libpolygames
    python3
    ${CMAKE_THREAD_LIBS_INIT}
    ${GTEST_LIBRARIES}
    ${JNI_LIBRARIES}
    ${TORCH_LIBRARIES}
    )

install (TARGETS polygames-tests DESTINATION bin)

