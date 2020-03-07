FROM debian:buster
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get install -y git build-essential curl sudo
RUN useradd -ms /bin/bash -G sudo myuser
RUN echo "myuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER myuser
ENV USER=myuser

# get Polygames
WORKDIR /home/myuser
#RUN git clone https://github.com/facebookincubator/polygames.git
RUN git clone https://github.com/juliendehos/Polygames.git --branch=nix
WORKDIR /home/myuser/Polygames

# install Nix
RUN curl https://nixos.org/releases/nix/latest/install | sh
#RUN echo "source /home/myuser/.nix-profile/etc/profile.d/nix.sh" >> /home/myuser/.bashrc
#RUN source /home/myuser/.bashrc
ENV PATH="/home/myuser/.nix-profile/bin/:$PATH"

# activate cachix repo
RUN nix-env -iA nixpkgs.cachix
RUN cachix use polygames

# build Polygames
RUN mkdir /home/myuser/Polygames/build
WORKDIR /home/myuser/Polygames/build
RUN nix-shell ../nix/shell-cpu.nix --run "cmake -DPYTORCH12=ON .. ; make -j4"

# tests
WORKDIR /home/myuser/Polygames/
RUN nix-shell nix/shell-cpu.nix --run "pytest pypolygames --durations=10 --verbose"

# unit-tests
RUN mkdir /home/myuser/Polygames/tests/build
WORKDIR /home/myuser/Polygames/tests/build
RUN nix-shell ../../nix/shell-cpu.nix --run "cmake .. ; make -j4 ; ./polygames-tests"

# docker build -t polygames .

