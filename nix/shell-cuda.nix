let

  jsonPath = ../nvidia.json;
  hasJson = builtins.pathExists jsonPath;
  json = builtins.fromJSON (builtins.readFile jsonPath);

  overlay = self: super:
  {
    linuxPackages = super.linuxPackages //
    {
      nvidia_x11 = (super.linuxPackages.nvidia_x11.override {
      }).overrideAttrs(oldAttrs: rec {
        version = json.nvidiaVersion;
        name = "nvidia-${version}";
        src = super.fetchurl {
          url = "http://download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run";
          sha256 = json.nvidiaSha256;
        };
        useGLVND = true;
      });
    };
  };

  rev = "dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b";
  channel = fetchTarball "https://github.com/NixOS/nixpkgs/archive/${rev}.tar.gz";
  config = {
    allowUnfree = true;
    cudaSupport = true;
    packageOverrides = pkgs: {
      cudatoolkit = pkgs.cudatoolkit_10;
      cudnn = pkgs.cudnn_cudatoolkit_10;
    };
  };
  pkgs = import channel {
    overlays = if hasJson then [overlay] else [];
    inherit config;
  };

  python = pkgs.python3;
  pytorch = pkgs.python3Packages.pytorchWithCuda;
  pybind11 = pkgs.pybind11;

  tensorboardX = pkgs.python3Packages.buildPythonPackage rec {
    pname = "tensorboardX";
    version = "1.8";
    src = fetchTarball "https://github.com/lanpa/tensorboardX/archive/v1.8.tar.gz";
    propagatedBuildInputs = with pkgs.python3Packages; [
      six
      protobuf
      numpy
    ];
    doCheck = false;
  };

in pkgs.mkShell {
  name = "Polygames-cuda";
  src = ./.;

  buildInputs = [
    pkgs.boost
    pkgs.cmake
    pkgs.cudatoolkit
    pkgs.cudnn
    pkgs.czmq
    pkgs.gtest
    pkgs.linuxPackages.nvidia_x11
    pkgs.python3Packages.pytest
    pytorch
    tensorboardX
  ];

  shellHook = ''
      export CFLAGS="-I${pybind11}/include -I${pytorch}/${python.sitePackages}/torch/include -I${pytorch}/${python.sitePackages}/torch/include/torch/csrc/api/include"
      export CXXFLAGS=$CFLAGS
      export LDFLAGS="-L${pytorch}/${python.sitePackages}/torch/lib -L$out/${python.sitePackages} -L${pkgs.cudatoolkit}/lib"
      export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
      export PYTHONPATH="$PYTHONPATH:build:build/torchRL/mcts:build/torchRL/tube"
      export OMP_NUM_THREADS=1
  '';
}

