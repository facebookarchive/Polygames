cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# set(CMAKE_PREFIX_PATH
#     /private/home/hengyuan/miniconda3/envs/pytorch-c/lib/python3.7/site-packages/torch)
# find_package(Torch REQUIRED)

execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)

find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)


# lib for other c++ programs
add_library(_tube src_cpp/data_channel.cc src_cpp/replay_buffer.cc src_cpp/reqrepserver.cpp)
target_include_directories(_tube PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_tube PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_tube PUBLIC ${TORCH_LIBRARIES} ${TorchPath}/lib/libtorch_python.so fmt zmq)

# tests
add_executable(test_data_channel src_cpp/test/test_data_channel.cc)
target_link_libraries(test_data_channel _tube)

# pybind lib
pybind11_add_module(tube src_cpp/pybind.cc)
target_link_libraries(tube PUBLIC _tube)
