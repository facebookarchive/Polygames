# Polygames for Nix users

[Nix](https://nixos.org/) is a package manager that can be intalled on any
Linux distribution (you just need root permissions to create the `/nix`
directory). Alternatively, NixOS is a Linux distribution based on Nix.

Polygames is quite easy to use with Nix, as explained below.


## Configure your system (NixOS)

- With NixOS, if you want to run Polygames on CPU, you have nothing to
  configure.

- If you want to run Polygames on CUDA devices, check that the Nvidia driver is
  enable in `/etc/nixos/configuration.nix` (don't forget to rebuild and reboot
  the system, if necessary):

    ```
      services.xserver.videoDrivers = [ "nvidia" ];
      nixpkgs.config.allowUnfree = true;
    ```


## Configure your system (Nix + Linux)

- Install Nix:

    ```
    curl https://nixos.org/releases/nix/latest/install | sh
    echo "source $HOME/.nix-profile/etc/profile.d/nix.sh" >> $HOME/.bashrc
    source $HOME/.bashrc
    ```

- If you want to run Polygames on CUDA devices, install the Nvidia driver using
  your Linux distribution.


## Build & run Polygames

- Activate the binary cache (optional if you like compiling for hours):

    ```
    nix-env -iA nixpkgs.cachix
    cachix use polygames
    ```

    > Warning: this cache provides pre-built binaries for CPU and for Nvidia
    > 418.74 only but you can [build your own binary
    > cache](README.md#build-you-own-binary-cache).

- Clone the repo:

    ```
    git clone https://github.com/facebookincubator/polygames.git
    cd Polygames
    ```

- Open a nix-shell (CPU):

    ```
    nix-shell nix/shell-cpu.nix
    ```

- Open a nix-shell (CUDA):

    - Find the version of your Nvidia driver, for example with `nvidia-settings
      --version`.

    - Open a nix-shell using the driver version:

        ```
        nix-shell nix/shell-cuda.nix --argstr nvidiaVersion 418.74
        ```

- Build Polygames:

    ```
    mkdir build
    cd build
    cmake -DPYTORCH12=ON ..
    make -j4
    cd ..
    ```

- Run Polygames (CPU):

    ```
    python -m pypolygames train --game_name="Connect4" --device=cpu
    ```

- Run Polygames (CUDA):

    ```
    python -m pypolygames train --game_name="Connect4" --device=cuda:0
    ```

## Build you own binary cache

When you open a nix-shell for the first time, Nix/NixOS fetches and builds all
the required dependencies. [This cachix repo](https://polygames.cachix.org/)
provides some pre-built dependencies that should greatly speed-up the
installation.

However, if your Nvidia driver version is not in the cache, you have to build
CUDA/Pytorch/etc. You can upload the built binaries to reuse them on other
machines:

- Create a (free) account on [cachix](https://cachix.org/).

- Create a repo on cachix.

- Push your binaries to this repo:

    ```
    find /nix/store -maxdepth 1 -name "*pytorch*" -exec cachix push <my-cachix-repo> {} \;
    ```

- Use your repo when you install/configure Polygames on a new machine:

    ```
    cachix use <my-cachix-repo>
    ```

