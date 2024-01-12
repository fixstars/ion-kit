[![Linux](https://github.com/fixstars/ion-kit/workflows/Linux/badge.svg)](https://github.com/fixstars/ion-kit/actions?query=workflow%3ALinux)
[![MacOS](https://github.com/fixstars/ion-kit/workflows/MacOS/badge.svg)](https://github.com/fixstars/ion-kit/actions?query=workflow%3AMacOS)
[![Windows](https://github.com/fixstars/ion-kit/workflows/Windows/badge.svg)](https://github.com/fixstars/ion-kit/actions?query=workflow%3AWindows)

# ion-kit
ion-kit is a graph based image processing framework based on Halide.
User can build pipeline using ion-kit API composing building blocks and compile it into static library or run just in time.

## Depedencies
* [Halide (v16.0.0)](https://github.com/halide/Halide/releases/tag/v16.0.0)
* [libjpeg](https://libjpeg-turbo.org/)
* [libpng](http://www.libpng.org/)
* [zlib](https://www.zlib.net/)

## Quick start

```c++
int main() {
    int32_t min0 = 0, extent0 = 2, min1 = 0, extent1 = 2, v = 1;

    // ion::Builder is fundamental class to build a graph.
    ion::Builder b;

    // Load ion building block module. User can create own BB module using ion::BuildingBlock class.
    b.with_bb_module("ion-bb-test");

    // Set the target architecture you will compile to. Here just use host architecture.
    b.set_target(ion::get_host_target());

    // Create simple graph consists from two nodes.
    //
    // test_producer -> test_consumer -> r (dummy output)
    //
    ion::Node n;
    n = b.add("test_producer").set_param(ion::Param("v", 41));
    n = b.add("test_consumer")(n["output"], &min0, &extent0, &min1, &extent1, &v);

    // Allocate dummy output. At least one output is required to run the pipeline.
    auto r = ion::Buffer<int32_t>::make_scalar();

    // Bind output with test_consumer "output" port.
    n["output"].bind(r);

    // Run the pipeline. Internally, it is compiled into native code just in time called as a function.
    b.run();
}
```

Compile it.

## Build from scratch
Please follow the instructions provided for your preferred platform.
* [Linux](INSTALL-LINUX.md)
* [Windows](INSTALL-WINDOWS.md)
* [MacOS](INSTALL-MACOS.md)

## CMake variables
| Variable          | Type   | Descriotion                                                               |
| ----------------- | ------ | ------------------------------------------------------------------------- |
| ION_BUILD_DOC     | ON/OFF | Enable to bulid documents. (Default: ON)                                  |
| ION_BUILD_TEST    | ON/OFF | Enable to bulid tests. (Default: ON)                                      |
| ION_BUILD_EXAMPLE | ON/OFF | Enable to bulid examples. (Default: ON)                                   |
| ION_BUNDLE_HALIDE | ON/OFF | Bundle Halide when packaging. (Default: OFF)                              |
| ION_ENABLE_HALIDE_FPGA_BACKEND | ON/OFF | Enable experimental FPGA backend. (Default: OFF)             |

## Authors
The ion-kit is an open-source project created by Fixstars Corporation and its subsidiary companies including Fixstars Solutions Inc, Fixstars Autonomous Technologies.

## Remark
This source code is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
