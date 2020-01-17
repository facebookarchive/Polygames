Unit tests for games

```
conda activate pypg
# or: source activate pypg

conda install gtest
# if necessary

mkdir build
cd build
cmake ..
make -j
./polygames-test
```

