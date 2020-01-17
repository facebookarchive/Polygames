# tube

### build
we need to build pytorch from source first. To do that, follow the instruction here 
https://github.com/pytorch/pytorch#from-source. 

Instruction for building PyTorch on devfair:
```
module purge
module load cudnn/v7.4-cuda.10.0
module load cuda/10.0

# create a fresh conda environment with python3
conda create --name [your env name] python=3.7

conda activate [your env name] # Or source activate [your env name], depending on conda version.

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda100

# clone the repo
# Note: put the repo onto /scratch partition for MUCH FASTER building speed. 
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# set cuda arch list so that the built binary can be run on both pascal and volta
TORCH_CUDA_ARCH_LIST="6.0;7.0" python setup.py install
```

To build this repo:

```
mkdir build
cd build
cmake ..
make
```

### run
Note that we need to set the following before running any multi-threading 
program that uses torch::Tensor. Otherwise a simple tensor operation will
use all cores by default.
```
export OMP_NUM_THREADS=1
```


Todo:

1. coding convention & clang-format file
2. split into .h and .cc properly

Todo++:
1. improve performance by changing to lock-free structure
2. multi-buffer
3. we should be ablt to test environment in C++ alone


To explain:

1. pybind shared_ptr
2. pybind keep_alive
3. torch accessor can still go out of bound, i.e. extremely wired segfault, we need to think about it
