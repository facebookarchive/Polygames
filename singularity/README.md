# Polygames singularity image


Build an image with the following command (from polygames root directory):
```bash
singularity build singularity/polygames.simg singularity/polygames.def
```
This can take up to 45min.

Once the image is built, you can run it with:
```bash
singularity shell --nv /checkpoint/polygames/polygames.simg
```
The `--nv` parameters gives access to the GPUs. Do not worry about the warnings: `awk: warning: escape sequence`.

In the image, you can compile polygames with (please remove build if it already exists):
```bash
cd polygames
mkdir build
cd build
cmake ..
make
```

The image can be used to run any command:
```bash
singularity run --nv singularity/polygames.simg python -m pypolygames
```
