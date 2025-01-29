## 1. Setup dependencies

Here is the list of dependent software.

- Mandatory
  - [Halide (v17.0.1)](https://github.com/halide/Halide/releases/tag/v17.0.1)

For Halide, please find latest binary release [here](https://github.com/halide/Halide/releases).

```sh
curl -sL https://github.com/halide/Halide/releases/download/v17.0.1/Halide-17.0.1-x86-64-osx-52541176253e74467dabc42eeee63d9a62c199f6.tar.gz | tar zx
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
