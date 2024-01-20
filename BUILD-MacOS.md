## 1. Setup dependencies

Here is the list of dependent software.

- Mandatory
  - [Halide (v16.0.0)](https://github.com/halide/Halide/releases/tag/v16.0.0)

For Halide, please find latest binary release [here](https://github.com/halide/Halide/releases).

```sh
curl -sL https://github.com/halide/Halide/releases/download/v16.0.0/Halide-16.0.0-arm-64-osx-1e963ff817ef0968cc25d811a25a7350c8953ee6.tar.gz | tar zx
```

## 2. Build

Here is CMake variables
| Variable          | Type   | Descriotion                                                               |
| ----------------- | ------ | ------------------------------------------------------------------------- |
| ION_BUILD_DOC     | ON/OFF | Enable to bulid documents. (Default: ON)                                  |
| ION_BUILD_TEST    | ON/OFF | Enable to bulid tests. (Default: ON)                                      |
| ION_BUILD_EXAMPLE | ON/OFF | Enable to bulid examples. (Default: ON)                                   |

Under the ion-kit source tree, run following command.
`HALIDE_PATH` is the installation directory path of Halide.

```sh
mkdir build && cd build
cmake -D Halide_DIR=${HALIDE_PATH}/lib/cmake/Halide -D HalideHelpers_DIR=${HALIDE_PATH}/lib/cmake/HalideHelpers -DCMAKE_BUILD_TYPE=Release -D ION_BUILD_TEST=ON -D ION_BUILD_EXAMPLE=ON ..
cmake --build .
ctest
```
