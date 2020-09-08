cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# set(CMAKE_PREFIX_PATH
#     /private/home/hengyuan/miniconda3/envs/pytorch-c/lib/python3.7/site-packages/torch)
# find_package(Torch REQUIRED)

find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

# lib for other c++ programs
add_library(_tube
  src_cpp/data_channel.cc
)
target_include_directories(_tube PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_tube PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_tube PUBLIC ${TORCH_LIBRARIES} fmt zmq)

#target_include_directories(_tube PUBLIC ../third_party ../third_party/zstd/lib ../third_party/zstd/lib/common)

# tests
#add_executable(test_data_channel src_cpp/test/test_data_channel.cc)
#target_link_libraries(test_data_channel _tube)

# pybind lib
pybind11_add_module(tube src_cpp/pybind.cc)
target_link_libraries(tube PUBLIC libpolygames)
