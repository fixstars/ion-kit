[![Linux](https://github.com/fixstars/ion-kit/workflows/Linux/badge.svg)](https://github.com/fixstars/ion-kit/actions?query=workflow%3ALinux)
[![MacOS](https://github.com/fixstars/ion-kit/workflows/MacOS/badge.svg)](https://github.com/fixstars/ion-kit/actions?query=workflow%3AMacOS)
[![Windows](https://github.com/fixstars/ion-kit/workflows/Windows/badge.svg)](https://github.com/fixstars/ion-kit/actions?query=workflow%3AWindows)

# ion-kit
The ion-kit is a graph based data processing framework.
User can define data processing algorithm in Halide as a "Building Block" (BB), then form a processing pipeline as a directed acyclic graph (DAG) combining BBs.
The pipeline can be ran just in time or compiled into binary object.

## Depedencies
* [Halide (v16.0.0)](https://github.com/halide/Halide/releases/tag/v16.0.0)
* [libjpeg](https://libjpeg-turbo.org/)
* [libpng](http://www.libpng.org/)
* [zlib](https://www.zlib.net/)

## Quick start

```c++
#include <ion/ion.h>

struct MyFilter : ion::BuildingBlock<MyFilter> {
    // This Building Block takes 1 input, 1 output and 1 parameter.
    ion::Input<Halide::Func> input{"input", Int(32), 1};
    ion::Output<Halide::Func> output{"output", Int(32), 1};
    ion::GeneratorParam<int32_t> v{"v", 0};

    void generate() {
        Halide::Var i;

        // Increment input elements by value specified as "v"
        output(i) = input(i) + v;
    }
};
ION_REGISTER_BUILDING_BLOCK(MyFilter, my_filter);

int main() {
    int32_t v = 1;

    // ion::Builder is the fundamental class to build a graph.
    ion::Builder b;

    // Set the target architecture same as host.
    b.set_target(ion::get_host_target());

    auto size = 4;

    ion::Buffer<int32_t> input{size};
    input.fill(0);

    // Create sequential graph.
    //
    // input -> my_filter (1st) -> my_filter (2nd) -> output
    //

    // Builder::add() creates Node object from Building Block.
    ion::Node n1 = b.add("my_filter");

    // Input is set by calling Node::operator().
    n1(input);

    // Parameter can be set by Node::set_param();
    n1.set_param(ion::Param("v", 40));

    // Method chain can be used to make it simple.
    auto n2 = b.add("my_filter")(n1["output"]).set_param(ion::Param("v", 2));

    // Bind output buffer.
    ion::Buffer<int32_t> output{size};
    output.fill(0);
    n2["output"].bind(output);

    // Run the pipeline.
    b.run();

    // Expected output is "42 42 42 42"
    for (int i=0; i<size; ++i) {
        std::cout << output(i) << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

Assuming [release binary](https://github.com/fixstars/ion-kit/releases) is extracted in <ion-kit-PATH>.

```bash
$ c++ -std=c++17 -fno-rtti main.cc -o main -I <ion-kit-PATH>/include -L <ion-kit-PATH>/lib -lion-core -lHalide && LD_LIBRARY_PATH=<ion-kit-PATH>/lib ./main
42 42 42 42
```

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
