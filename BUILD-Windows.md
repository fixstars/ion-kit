## 1. Setup dependencies

Here is the list of dependent software.

- Mandatory
  - [Halide (v16.0.0)](https://github.com/halide/Halide/releases/tag/v16.0.0)
- Optional
  - [libjpeg](https://libjpeg-turbo.org/)
  - [libpng](http://www.libpng.org/)
  - [zlib](https://www.zlib.net/)

For Halide, please find latest binary release [here](https://github.com/halide/Halide/releases) and extract it.

We recommend to setup libjpeg-turbo, libpng and zlib by [vcpkg](https://vcpkg.io/).

In root folder of `ion-kit` run:

```sh
vcpkg install
```

## 2. Build

Here is CMake variables
| Variable          | Type   | Descriotion                                                               |
| ----------------- | ------ | ------------------------------------------------------------------------- |
| ION_BUILD_DOC     | ON/OFF | Enable to bulid documents. (Default: ON)                                  |
| ION_BUILD_TEST    | ON/OFF | Enable to bulid tests. (Default: ON)                                      |
| ION_BUILD_EXAMPLE | ON/OFF | Enable to bulid examples. (Default: ON)                                   |

Under the ion-kit source tree, run following command.
Please notice `VCPKG_PATH` is your local path to the vcpkg installation directory.

```sh
mkdir build && cd build
run: cmake -G "Visual Studio 16 2019" -A x64 -D VCPKG_TARGET_TRIPLET=x64-windows-static -D CMAKE_TOOLCHAIN_FILE=${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake -D Halide_DIR=${HALIDE_PATH}/lib/cmake/Halide-D HalideHelpers_DIR=${HALIDE_PATH}/lib/cmake/HalideHelpers -D ION_BUILD_TEST=ON -D ION_BUILD_EXAMPLE=ON ..
cmake --build . --config Release
ctest
```
