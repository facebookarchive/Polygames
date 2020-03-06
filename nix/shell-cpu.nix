let
  rev = "dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b";
  channel = fetchTarball "https://github.com/NixOS/nixpkgs/archive/${rev}.tar.gz";
  config = { allowUnfree = true; };
  pkgs = import channel { inherit config; };

  python = pkgs.python3;
  pytorch = pkgs.python3Packages.pytorchWithoutCuda;
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
  name = "Polygames-cpu";
  src = ./.;

  buildInputs = [
    pkgs.cmake
    pkgs.czmq
    pkgs.gtest
    pkgs.python3Packages.pytest
    pytorch
    tensorboardX
  ];

  shellHook = ''
      export CFLAGS="-I${pybind11}/include -I${pytorch}/${python.sitePackages}/torch/include -I${pytorch}/${python.sitePackages}/torch/include/torch/csrc/api/include"
      export CXXFLAGS=$CFLAGS
      export LDFLAGS="-L${pytorch}/${python.sitePackages}/torch/lib -L$out/${python.sitePackages}"
      export PYTHONPATH="$PYTHONPATH:build:build/torchRL/mcts:build/torchRL/tube"
      export OMP_NUM_THREADS=1
  '';
}

