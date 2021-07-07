#!/bin/bash

set -eu

SUDOCMD=sudo
if [[ $(whoami) = "root" ]]; then
  SUDOCMD=
fi

HALIDE_URL=https://github.com/halide/Halide/releases/download/v8.0.0/halide-linux-64-gcc53-800-65c26cba6a3eca2d08a0bccf113ca28746012cc3.tgz
ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-linux-x64-1.4.0.tgz

IONKIT_URL=https://github.com/fixstars/ion-kit.git
IONKIT_CHECKOUT=501277472af52998982cb9860d597f2c659a8a44

INSTALL_ROOT=~/local
HALIDE_DIR=$INSTALL_ROOT/halide/
ONNXRUNTIME_DIR=$INSTALL_ROOT/onnxruntime/
IONKIT_DIR=$INSTALL_ROOT/ion-kit/
IONKIT_BUILD_DIR=./.tmp/ion-kit

$SUDOCMD apt update
$SUDOCMD apt install -y \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    ninja-build

mkdir -p $HALIDE_DIR
curl -L $HALIDE_URL | tar zx --strip-components=1 -C $HALIDE_DIR

mkdir -p $ONNXRUNTIME_DIR
curl -L $ONNXRUNTIME_URL | tar zx --strip-components=1 -C $ONNXRUNTIME_DIR

rm -rf $IONKIT_BUILD_DIR
mkdir -p $IONKIT_BUILD_DIR
git clone $IONKIT_URL $IONKIT_BUILD_DIR
cd $IONKIT_BUILD_DIR
git checkout $IONKIT_CHECKOUT
mkdir build
cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=$IONKIT_DIR -DHALIDE_ROOT=$HALIDE_DIR -DONNXRUNTIME_ROOT=$ONNXRUNTIME_DIR ../
cmake --build .
cmake --build . --target install
rm -rf $IONKIT_BUILD_DIR


echo "export LD_LIBRARY_PATH=${HALIDE_DIR}/bin:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${IONKIT_DIR}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
