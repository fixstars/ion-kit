# ion-kit
A framework to compile user-defined pipeline.

## Depedencies
* [Halide (v8.0.0)](https://github.com/halide/Halide/releases/tag/v8.0.0)
* doxygen
* sphinx

## Build
### 1. Install Halide
#### a. Using a binary release
```sh
curl -L https://github.com/halide/Halide/releases/download/v8.0.0/halide-linux-64-gcc53-800-65c26cba6a3eca2d08a0bccf113ca28746012cc3.tgz | tar zx <path-to-halide-install>
```

#### b. Build from source
##### 2.b.1. Build and install LLVM
Halide release_2019_08_27 requires LLVM 7.0-9.0. In the following, assume install LLVM-9.0.

```sh
git clone https://github.com/llvm/llvm-project.git -b release/9.x --depth=1
cd llvm-project
mkdir build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<path-to-llvm-install> -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;Hexagon;PowerPC;AMDGPU;RISCV" -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF ../llvm
cmake --build . --target install
```

##### 2.b.2 Build and install Halide
```sh
git clone https://github.com/halide/Halide.git -b v8.0.0 --depth=1
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=<path-to-halide-install> -DLLVM_DIR=<path-to-llvm-install>/lib/cmake/llvm/ -DLLVM_PACKAGE_VERSION=90 -DHALIDE_ENABLE_RTTI=ON -DWITH_APPS=OFF ..
cmake --build . --target install
```

### 2. Install onnxruntime (Optional, if you use ion-bb-dnn)
If you use only cpu, you can get binary release.

```sh
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-linux-x64-1.4.0.tgz | tar zx -C <path-to-onnxruntime-install>
```

### 3. Place additional building blocks (Appendix)
If you want to use additional `ion-bb-xx` directories, place them directly under ` `ion-kit`  directory.


### 4. Build
```sh
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=<path-to-ion-kit-install> -DHALIDE_ROOT=<path-to-halide-install> -DONNXRUNTIME_ROOT=<path-to-onnxruntime-root> ../
cmake --build .
```

### 5. Install
```sh
cmake --build . --target install
```

### 6. Run examples
```sh
ctest
```

## CMake variables
| Variable          | Type   | Descriotion                                                               |
| ----------------- | ------ | ------------------------------------------------------------------------- |
| HALIDE_ROOT       | Path   | Path to Halide install directory.                                         |
| ION_TARGET_BB_ALL | ON/OFF | Enable to buld all building blocks. (Default: ON)                         |
| ION_TARGET_BBS    | String | The building blocks of target to build. (This overrides ION_BUILD_ALL_BB) |
| ION_BUILD_DOC     | ON/OFF | Enable to bulid documents. (Default: ON)                                  |
| ION_BUILD_TEST    | ON/OFF | Enable to bulid tests. (Default: ON)                                      |
| ION_BUILD_EXAMPLE | ON/OFF | Enable to bulid examples. (Default: ON)                                   |
| WITH_CUDA         | ON/OFF | Enable CUDA with buliding examples. (Default: ON)                         |

## Authors
The ion-kit is an open-source project created by Fixstars Corporation and its subsidiary companies including Fixstars Solutions Inc, Fixstars Autonomous Technologies.

## Remark
This source code is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
