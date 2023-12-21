# ion-kit
A framework to compile user-defined pipeline. Current support:
  * Linux
  * Windows
  * MacOS

## Depedencies
* [Halide (v16.0.0)](https://github.com/halide/Halide/releases/tag/v16.0.0)
* llvm
* doxygen
* sphinx
* ninja (unix)
* msvc (windows)

## Build
### 1. Install LLVM
#### a. Using a binary release (Preferred for all systems)
Find latest binary release [here](https://github.com/llvm/llvm-project/releases/tag/llvmorg-16.0.1) for your supported system.
#### b. Install with Package Managers
##### 1.b.1. Install with Brew (MacOS)
```
brew install llvm@16
```

### 2. Install Halide
#### a. Using a binary release (Preferred for all systems)
Find latest binary release [here](https://github.com/halide/Halide/releases/tag/v16.0.0) for your supported system.
```sh
curl -L https://github.com/halide/Halide/releases/download/v12.0.1/Halide-12.0.1-x86-64-linux-5dabcaa9effca1067f907f6c8ea212f3d2b1d99a.tar.gz | tar zx <path-to-halide-install>
```
#### b. Install with Package Managers
##### 2.b.1. Install with Brew (MacOS)
```
brew install Halide@16
```
Use `brew info` to install and check missing dependencies (including `llvm16`)

#### c. Build from source
##### 2.c.1. Build and install LLVM
Halide v16.0.1 requires LLVM 16. In the following, assume install LLVM-16.0.

```sh
git clone https://github.com/llvm/llvm-project.git -b release/16.x --depth=1
cd llvm-project
mkdir build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<path-to-llvm-install> -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;Hexagon;PowerPC;AMDGPU;RISCV" -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF ../llvm
cmake --build . --target install
```

##### 2.c.2 Build and install Halide
```sh
git clone https://github.com/halide/Halide.git -b v16.0.0 --depth=1
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=<path-to-halide-install> -DLLVM_DIR=<path-to-llvm-install>/lib/cmake/llvm/ -DHALIDE_ENABLE_RTTI=ON -DWITH_APPS=OFF ..
cmake --build . --target install
```

### 2. Install onnxruntime (Optional, if you use ion-bb-dnn)
**Not currently supported on MacOS.**  If you use only cpu, you can get binary release.  
Please visit official onnxruntime [github](https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3) for latest release.

```sh
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-linux-x64-1.4.0.tgz | tar zx -C <path-to-onnxruntime-install>
```
* Please note, latest version of onnxruntime with GPU only supports `CUDA 1.8`
* `libcudnn8` will also be needed if you run with GPU, please install with:
```
sudo apt-get install libcudnn8
```

### 3. Place additional building blocks (Appendix)
If you want to use additional `ion-bb-xx` directories, place them directly under ` `ion-kit`  directory.

### 4. Install OpenCV
#### a. Using Package Managers
##### 4.a.1 Linux
```
sudo apt install libopencv-dev
```
##### 4.a.2 MacOS
```
brew install opencv
```
#### b. Using Installers
##### 4.b.1 Windows
Downalod and install latest from [OpenCV](https://sourceforge.net/projects/opencvlibrary/files/)

### 5. Install Generators
#### a. Using Package Managers
##### 4.a.1 Linux
```
sudo apt-get install ninja-build
```
##### 4.a.2 MacOS
```
brew install ninja
```
##### 4.a.2 Windows
Please install the latest version of Microsoft Visual Studio.


### 6. Build
#### a. Unix
```sh
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=<path-to-ion-kit-install> -DHalide_DIR=<path-to-HalideConfig.cmake> -DONNXRUNTIME_ROOT=<path-to-onnxruntime-root> -DOPENCV_DIR=<path-to-opencv-cmake> ../
cmake --build .
```
#### b. Windows
```
cmake -G "Visual Studio 17 2022" -A x64 -DHalide_DIR=<path-to-HalideConfig.cmake> -DOpenCV_DIR=<path-to-opencv-cmake> ../
cmake --build . --config Release
```

### 7. Install
```sh
cmake --build . --target install
```

### 8. Run examples
```sh
ctest
```

## CMake variables
| Variable          | Type   | Descriotion                                                               |
| ----------------- | ------ | ------------------------------------------------------------------------- |
| ION_BUILD_ALL_BB  | ON/OFF | Enable to buld all building blocks. (Default: ON)                         |
| ION_BBS_TO_BUILD  | String | The building blocks of target to build. (This overrides ION_BUILD_ALL_BB) |
| ION_BUILD_DOC     | ON/OFF | Enable to bulid documents. (Default: ON)                                  |
| ION_BUILD_TEST    | ON/OFF | Enable to bulid tests. (Default: ON)                                      |
| ION_BUILD_EXAMPLE | ON/OFF | Enable to bulid examples. (Default: ON)                                   |
| WITH_CUDA         | ON/OFF | Enable CUDA with buliding examples. (Default: ON)                         |

## Authors
The ion-kit is an open-source project created by Fixstars Corporation and its subsidiary companies including Fixstars Solutions Inc, Fixstars Autonomous Technologies.

## Remark
This source code is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
