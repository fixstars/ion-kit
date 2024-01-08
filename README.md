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

### 1. Install
Please follow the instructions provided for your preferred platform.
* [Linux](INSTALL-LINUX.md)
* [Windows](INSTALL-WINDOWS.md)
* [MacOS](INSTALL-MACOS.md)

### 2. Build
#### a. Unix
```sh
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=<path-to-ion-kit-install> -DHalide_DIR=<path-to-HalideConfig.cmake> -DHalideHelpers_DIR=<path-to-halide-helpers> -DONNXRUNTIME_ROOT=<path-to-onnxruntime-root> -DOPENCV_DIR=<path-to-opencv-cmake> ../
cmake --build .
```
#### b. Windows
```
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DHalideHelpers_DIR=<path-to-halide-helpers> -DHalide_DIR=<path-to-HalideConfig.cmake> -DOpenCV_DIR=<path-to-opencv-cmake> -D ION_BUILD_ALL_BB=OFF -DION_BBS_TO_BUILD="ion-bb-core;ion-bb-image-processing;ion-bb-sgm;ion-bb-image-io" ../
cmake --build . --config Release
```

### 3. Install
```sh
cmake --build . --target install
```

### 4. Run examples
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
