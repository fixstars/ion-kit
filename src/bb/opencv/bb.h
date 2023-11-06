#ifndef ION_BB_OPENCV_BB_H
#define ION_BB_OPENCV_BB_H

#include <ion/ion.h>

class MedianBlur : public ion::BuildingBlock<MedianBlur> {
public:
    GeneratorParam<int32_t> ksize{"ksize", 3};
    Input<Halide::Func> input{"input", UInt(8), 3};
    Input<int32_t> width{"width", 0};
    Input<int32_t> height{"height", 0};
    Output<Halide::Func> output{"output", UInt(8), 3};

    void generate() {
        using namespace Halide;
        Func in;
        in(_) = input(_);
        in.compute_root();
        std::vector<ExternFuncArgument> params{in, cast<int32_t>(3), width, height, static_cast<int>(ksize)};
        Func median_blur;
        median_blur.define_extern("median_blur", params, UInt(8), 3);
        median_blur.compute_root();
        output(c, x, y) = median_blur(c, x, y);
    }

    void schedule() {
        output.compute_root();
    }

private:
    Halide::Var c, x, y;
};
ION_REGISTER_BUILDING_BLOCK(MedianBlur, opencv_median_blur);

class Display : public ion::BuildingBlock<Display> {
public:
    GeneratorParam<int32_t> idx{"idx", 0};
    Input<Halide::Func> input{"input", UInt(8), 3};
    Input<int32_t> width{"width", 0};
    Input<int32_t> height{"height", 0};

    Output<int> output{"output"};
    void generate() {
        using namespace Halide;
        Func in;
        in(_) = input(_);
        in.compute_root();
        std::vector<ExternFuncArgument> params = {in, width, height, static_cast<int>(idx)};
        Func display;
        display.define_extern("display", params, Int(32), 0);
        display.compute_root();
        output() = display();
    }

    void schedule() {
    }

private:
    Halide::Var x, y;
};
ION_REGISTER_BUILDING_BLOCK(Display, opencv_display);

#endif // ION_BB_OPENCV_BB_H
