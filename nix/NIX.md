
# Run Polygames using Nix


## Setup for NixOS

- check that the nvidia driver is enable in `/etc/nixos/configuration.nix`:

```
  services.xserver.videoDrivers = [ "nvidia" ];
  nixpkgs.config.allowUnfree = true;
```


## Setup for Nix on Linux

- install the nvidia driver

- install Nix:

```
curl https://nixos.org/releases/nix/latest/install | sh
echo "source $HOME/.nix-profile/etc/profile.d/nix.sh" >> $HOME/.bashrc
source $HOME/.bashrc
```


## Build & run Polygames

- activate binary cache (optional if you like compiling for hours):

```
nix-env -iA nixpkgs.cachix
cachix use polygames
```

- get Polygames:

```
git clone https://github.com/facebookincubator/polygames.git
cd Polygames
```

- run a nix-shell (one of these lines);

```
nix-shell nix/shell-cuda.nix
nix-shell nix/shell-cpu.nix
```

- build Polygames:

```
mkdir build
cd build
cmake -DPYTORCH12=ON ..
make -j4
cd ..
```

- run Polygames:

```
python -m pypolygames train --game_name="Connect4" --device=cuda:0
python -m pypolygames train --game_name="Connect4" --device=cpu
```

