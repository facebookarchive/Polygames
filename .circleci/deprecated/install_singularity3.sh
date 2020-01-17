#!/bin/bash
# updated from singularityhub example

singularity_version="${singularity_version:-3.2.1}"

sudo sed -i -e 's/^Defaults\tsecure_path.*$//' /etc/sudoers

# singularity ##################################################################

echo "Installing singularity deps"
sudo apt-get update 
sudo apt-get install -y build-essential \
                                  wget \
                                  git \
                                  jq \
                                  squashfs-tools \
                                  libtool \
                                  uuid-dev \
                                  libssl-dev \
                                  libgpgme11-dev \
                                  libseccomp-dev \
                                  libarchive-dev \
                                  pkg-config

# go --version
# added libarchive-dev for singularity 2.6.1


echo "Installing go"
which go  # already here?
echo "Go $GOPATH"

sudo rm -rvf /usr/local/go/
export VERSION=1.12 OS=linux ARCH=amd64 && \
./mconfig && \
    make -C ./builddir && \
    sudo make -C ./builddir install
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz

#export PATH=$PATH:/usr/local/go/bin && \
#    sudo mkdir -p /go && \
#    sudo chmod -R 7777 /go && \
#    export GOPATH=/go

echo "Installing singularity"

export VERSION=3.2.1 && # adjust this as necessary \
    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz -O ~/singularity.tar.gz && \
    cd ~ && \
    tar -xzf singularity.tar.gz && cd singularity

./mconfig && \
    make -C ./builddir && \
    sudo make -C ./builddir install
