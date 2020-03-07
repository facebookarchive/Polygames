# This Dockerfile configures a Debian system with Nix, builds Polygames and
# runs some tests. To build this docker image, run:
# `docker build -t polygames .`


###############################################################################
# Initialize the docker image. You can ignore this when installing on a real
# system.
###############################################################################
FROM debian:buster
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get install -y git curl sudo xz-utils
RUN useradd -ms /bin/bash -G sudo myuser
RUN echo "myuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER myuser
ENV USER="myuser"
ENV HOME="/home/myuser"


###############################################################################
# Install Nix.
###############################################################################
RUN curl https://nixos.org/releases/nix/latest/install | sh

# For docker only:
ENV PATH="$HOME/.nix-profile/bin/:$PATH"

# On a real system, yau have to run these two lines instead:
#RUN echo "source $HOME/.nix-profile/etc/profile.d/nix.sh" >> $HOME/.bashrc
#RUN source $HOME/.bashrc


###############################################################################
# Activate the cachix repo.
###############################################################################
RUN nix-env -iA nixpkgs.cachix
RUN cachix use polygames


###############################################################################
# Get Polygames.
###############################################################################
WORKDIR $HOME
RUN git clone https://github.com/facebookincubator/polygames.git
WORKDIR $HOME/Polygames

# On a real system with CUDA devices, you have to run `./nix/get-nvidia.sh`
# here.


###############################################################################
# Build Polygames.
###############################################################################
RUN mkdir $HOME/Polygames/build
WORKDIR $HOME/Polygames/build
RUN nix-shell ../nix/shell-cpu.nix --run "cmake -DPYTORCH12=ON .. ; make -j4"


###############################################################################
# Run unit-tests.
###############################################################################
RUN mkdir $HOME/Polygames/tests/build
WORKDIR $HOME/Polygames/tests/build
RUN nix-shell ../../nix/shell-cpu.nix --run "cmake .. ; make -j4 ; ./polygames-tests"


###############################################################################
# Run tests.
###############################################################################
WORKDIR $HOME/Polygames/
RUN nix-shell nix/shell-cpu.nix --run "pytest pypolygames --durations=10 --verbose"


