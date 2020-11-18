#ifndef ION_BB_DEMO_BB_H
#define ION_BB_DEMO_BB_H

#include <ion/ion.h>

#include "bb_sgm.h"
//#include "bb_dnn.h"

#include <cmath>

namespace ion {
namespace bb {
namespace demo {

class Schedule : public ion::BuildingBlock<Schedule> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Schedule"};
    GeneratorParam<std::string> gc_description{"gc_description", "This applies various scheduling."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,network"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> output_name{"output_name", ""};
    GeneratorParam<bool> output_replace{"output_replace", false};
    GeneratorInput<Halide::Func> input{"input"};
    GeneratorOutput<Halide::Func> output{"output"};

    void generate() {
        using namespace Halide;
        Func f(static_cast<std::string>(output_name));
        f(_) = input(_);
        f.compute_root();

        if (get_target().has_gpu_feature()) {
            if (f.args().size() == 1) {
                Var i = f.args()[0];
                Var block, thread;
                f.split(i, block, thread, 64);
                f.gpu_blocks(block).gpu_threads(thread);
            } else {
                Var x = f.args()[f.args().size() - 2];
                Var y = f.args()[f.args().size() - 1];
                Var xo, yo, xi, yi;
                f.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            }
        }

        if (static_cast<bool>(output_replace)) {
            output = f;
        } else {
            output(_) = f(_);
        }
    }
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Schedule, demo_schedule);

namespace ion {
namespace bb {
namespace demo {

class BayerMap {
public:
    enum class Pattern {
        RGGB,
        BGGR,
        GRBG,
        GBRG
    };

    static const std::map<std::string, Pattern> enum_map;

    // R: 0
    // G: 1
    // B: 2
    static Halide::Expr get_color(Pattern pat, Halide::Expr x, Halide::Expr y) {
        return Halide::select(
            y % 2 == 0,
            Halide::select(
                x % 2 == 0,
                get_color(pat, 0),
                get_color(pat, 1)),
            Halide::select(
                x % 2 == 0,
                get_color(pat, 2),
                get_color(pat, 3)));
    }

private:
    static const int bayer_map[4][4];

    static int get_color(Pattern pat, int pos) {
        return bayer_map[static_cast<int>(pat)][pos];
    }
};

const std::map<std::string, BayerMap::Pattern> BayerMap::enum_map{
    {"RGGB", BayerMap::Pattern::RGGB},
    {"BGGR", BayerMap::Pattern::BGGR},
    {"GRBG", BayerMap::Pattern::GRBG},
    {"GBRG", BayerMap::Pattern::GBRG}};

const int BayerMap::bayer_map[4][4]{
    {0, 1, 1, 2},  // RGGB
    {2, 1, 1, 0},  // BGGR
    {1, 0, 2, 1},  // GRBG
    {1, 2, 0, 1}   // GBRG
};

class Luminance {
public:
    enum class Method {
        Max,
        Average,
        SimpleY,
        Y
    };

    static const std::map<std::string, Method> enum_map;

    static Halide::Expr calc(Method method, Halide::Expr r, Halide::Expr g, Halide::Expr b) {
        switch (method) {
        case Method::Max:
            return Halide::max(r, g, b);
        case Method::Average:
            return (r + g + b) / 3;
        case Method::SimpleY:
            return (r * 3 + g * 12 + b) / 16;
        case Method::Y:
            return r * 0.2126f + g * 0.7152f + b * 0.0722f;  // BT.709
        default:
            internal_error << "Unknown Luminance method";
        }
    }
};

const std::map<std::string, Luminance::Method> Luminance::enum_map{
    {"Max", Luminance::Method::Max},
    {"Average", Luminance::Method::Average},
    {"SimpleY", Luminance::Method::SimpleY},
    {"Y", Luminance::Method::Y}};

Halide::Expr select_by_color(Halide::Expr color, Halide::Expr r_value, Halide::Expr g_value, Halide::Expr b_value) {
    return Halide::select(
        color == 0,
        r_value,
        Halide::select(
            color == 1,
            g_value,
            b_value));
}

class BayerOffset : public BuildingBlock<BayerOffset> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerOffset"};
    GeneratorParam<std::string> gc_description{"gc_description", "Offset values of bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<float> offset_r{"offset_r"};
    GeneratorInput<float> offset_g{"offset_g"};
    GeneratorInput<float> offset_b{"offset_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        BayerMap::Pattern bayer_pattern = BayerMap::Pattern::RGGB;
        output(x, y) = Halide::clamp(input(x, y) - select_by_color(BayerMap::get_color(bayer_pattern, x, y), offset_r, offset_g, offset_b), 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.align_bounds(x, 2).align_bounds(y, 2);

        if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi, xii, yii;
            output.gpu_tile(x, y, xi, yi, 32, 32).tile(xi, yi, xii, yii, 2, 2).unroll(xii).unroll(yii);
        } else {
            Halide::Var xi, yi;
            output.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi).vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x, y;
};

class BayerWhiteBalance : public BuildingBlock<BayerWhiteBalance> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerWhiteBalance"};
    GeneratorParam<std::string> gc_description{"gc_description", "Gain values of bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<float> gain_r{"gain_r"};
    GeneratorInput<float> gain_g{"gain_g"};
    GeneratorInput<float> gain_b{"gain_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        BayerMap::Pattern bayer_pattern = BayerMap::Pattern::RGGB;
        output(x, y) = Halide::clamp(input(x, y) * select_by_color(BayerMap::get_color(bayer_pattern, x, y), gain_r, gain_g, gain_b), 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.align_bounds(x, 2).align_bounds(y, 2);

        if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi, xii, yii;
            output.gpu_tile(x, y, xi, yi, 32, 32).tile(xi, yi, xii, yii, 2, 2).unroll(xii).unroll(yii);
        } else {
            Halide::Var xi, yi;
            output.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi).vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x, y;
};

class BayerDemosaicSimple : public BuildingBlock<BayerDemosaicSimple> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerDemosaicSimple"};
    GeneratorParam<std::string> gc_description{"gc_description", "Demosaic bayer image by simple algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,arithmetic"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map(x => x / 2).concat([3]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 3};

    void generate() {

        Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, width}, {0, height}});
        BayerMap::Pattern bayer_pattern = BayerMap::Pattern::RGGB;
        switch (bayer_pattern) {
        case BayerMap::Pattern::RGGB:
            output(x, y, c) = select_by_color(
                c,
                input_wrapper(x * 2, y * 2),
                (input_wrapper(x * 2 + 1, y * 2) + input_wrapper(x * 2, y * 2 + 1)) / 2,
                input_wrapper(x * 2 + 1, y * 2 + 1));
            break;
        case BayerMap::Pattern::BGGR:
            output(x, y, c) = select_by_color(
                c,
                input_wrapper(x * 2 + 1, y * 2 + 1),
                (input_wrapper(x * 2 + 1, y * 2) + input_wrapper(x * 2, y * 2 + 1)) / 2,
                input_wrapper(x * 2, y * 2));
            break;
        case BayerMap::Pattern::GRBG:
            output(x, y, c) = select_by_color(
                c,
                input_wrapper(x * 2 + 1, y * 2),
                (input_wrapper(x * 2, y * 2) + input_wrapper(x * 2 + 1, y * 2 + 1)) / 2,
                input_wrapper(x * 2 + 1, y * 2));
            break;
        case BayerMap::Pattern::GBRG:
            output(x, y, c) = select_by_color(
                c,
                input_wrapper(x * 2, y * 2 + 1),
                (input_wrapper(x * 2, y * 2) + input_wrapper(x * 2 + 1, y * 2 + 1)) / 2,
                input_wrapper(x * 2 + 1, y * 2));
            break;
        default:
            internal_error << "Unknown BayerMap pattern";
        }
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).bound(c, 0, 3).unroll(c);

        if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x{"x"}, y{"y"}, c{"c"};
};

template<typename X, int32_t D>
class GammaCorrection : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Gamma correction."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<float> gamma{"gamma"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), D};

    void generate() {
        output(Halide::_) = Halide::pow(input(Halide::_), gamma);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (D == 3) {
            Halide::Var c = output.args()[2];
            output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        }

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 8);
        } else {
            output.vectorize(x, this->natural_vector_size(Halide::Float(32)));
            output.parallel(y, 16);
        }
        output.compute_root();
#endif
    }
};

class GammaCorrection3D : public GammaCorrection<GammaCorrection3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "GammaCorrection3D"};
};

class LensShadingCorrectionLinear : public BuildingBlock<LensShadingCorrectionLinear> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "LensShadingCorrectionLinear"};
    GeneratorParam<std::string> gc_description{"gc_description", "Correct lens shading."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<float> slope_r{"slope_r"};
    GeneratorInput<float> slope_g{"slope_g"};
    GeneratorInput<float> slope_b{"slope_b"};
    GeneratorInput<float> offset_r{"offset_r"};
    GeneratorInput<float> offset_g{"offset_g"};
    GeneratorInput<float> offset_b{"offset_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        BayerMap::Pattern bayer_pattern = BayerMap::Pattern::RGGB;
        Halide::Expr center_x, center_y, r2;

        center_x = width / Halide::cast<float>(2.f);
        center_y = height / Halide::cast<float>(2.f);
        r2 = ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) / (center_x * center_x + center_y * center_y);

        output(x, y) = input(x, y) * select_by_color(
                                         BayerMap::get_color(bayer_pattern, x, y),
                                         r2 * slope_r + offset_r,
                                         r2 * slope_g + offset_g,
                                         r2 * slope_b + offset_b);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.align_bounds(x, 2).align_bounds(y, 2);

        if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi, xii, yii;
            output.gpu_tile(x, y, xi, yi, 32, 32).tile(xi, yi, xii, yii, 2, 2).unroll(xii).unroll(yii);
        } else {
            Halide::Var xi, yi;
            output.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi).vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }
        output.compute_root();
#endif
    }

private:
    Halide::Var x, y;
};

class CalcLuminance : public BuildingBlock<CalcLuminance> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CalcLuminance"};
    GeneratorParam<std::string> gc_description{"gc_description", "Calc luminance of image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,arithmetic"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.slice(0, -1) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        Luminance::Method luminance_method = Luminance::Method::Y;
        output(x, y) = Luminance::calc(luminance_method, input(x, y, 0), input(x, y, 1), input(x, y, 2));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x, y;
};

template<typename X, int32_t D>
class ResizeBilinear : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Resize image by bilinear algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<float> scale{"scale", 1.f};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), D};

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}});

        Halide::Expr map_x, map_y, x0, y0, x1, y1, x_coef, y_coef;

        map_x = (x + 0.5f) / scale - 0.5f;
        map_y = (y + 0.5f) / scale - 0.5f;

        x0 = Halide::cast<int32_t>(Halide::floor(map_x));
        y0 = Halide::cast<int32_t>(Halide::floor(map_y));
        x1 = x0 + 1;
        y1 = y0 + 1;
        x_coef = map_x - x0;
        y_coef = map_y - y0;

        output(x, y, Halide::_) = (input_wrapper(x0, y0, Halide::_) * (1 - x_coef) + input_wrapper(x1, y0, Halide::_) * x_coef) * (1 - y_coef) +
                                  (input_wrapper(x0, y1, Halide::_) * (1 - x_coef) + input_wrapper(x1, y1, Halide::_) * x_coef) * y_coef;
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (D == 3) {
            Halide::Var c = output.args()[2];
            output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        }

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class ResizeBilinear3D : public ResizeBilinear<ResizeBilinear3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeBilinear3D"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.slice(0, -1).map(x => Math.floor(x * parseFloat(v.scale))).concat(v.input.slice(-1)) }}))"};
};

template<typename X, typename T>
class BayerDownscale : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Downscale bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map(x => Math.floor(x / parseInt(v.downscale_factor))) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorParam<int32_t> downscale_factor{"downscale_factor", 1};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 2};

    void generate() {
        Halide::Var x;
        Halide::Var y;
        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, input_width}, {0, input_height}});
        output(x, y) = input_wrapper(x / 2 * 2 * downscale_factor + x % 2, y / 2 * 2 * downscale_factor + y % 2);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class BayerDownscaleUInt16 : public BayerDownscale<BayerDownscaleUInt16, uint16_t> {
    GeneratorParam<std::string> gc_title{"gc_title", "BayerDownscaleUInt16"};
};

class NormalizeRawImage : public BuildingBlock<NormalizeRawImage> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize RAW"};
    GeneratorParam<std::string> gc_description{"gc_description", "Normalize raw image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<uint8_t> bit_width{"bit_width", 10};
    GeneratorParam<uint8_t> bit_shift{"bit_shift", 6};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint16_t>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 2};

    void generate() {
        Halide::Var x;
        Halide::Var y;
        output(Halide::_) = Halide::cast<float>(input(Halide::_) >> Halide::cast<uint8_t>(bit_shift)) / ((1 << bit_width) - 1);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

template<typename X, typename T, int32_t D>
class MergeImage : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Merge images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "output_width,output_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};
    GeneratorParam<int32_t> output_width{"output_width", 0};
    GeneratorParam<int32_t> output_height{"output_height", 0};

    GeneratorParam<int32_t> input1_left{"input1_left", 0};
    GeneratorParam<int32_t> input1_top{"input1_top", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<T>(), D};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Var x, y;

        Halide::Func input1_wrapper;

        input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{0, input1_width}, {0, input1_height}});

        output(x, y, Halide::_) = Halide::select(
            x >= input1_left && x < input1_left + input1_width && y >= input1_top && y < input1_top + input1_height,
            input1_wrapper(x - input1_left, y - input1_top, Halide::_),
            input0(x, y, Halide::_));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (D == 3) {
            Halide::Var c = output.args()[2];
            output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        }

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class MergeImage3DFloat : public MergeImage<MergeImage3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "MergeImage3DFloat"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.output_width), parseInt(v.output_height), v.input0[2]] }}))"};
};

class Tile2Images3DArrayFloat : public BuildingBlock<Tile2Images3DArrayFloat> {
    // Currently available only 2x1 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile2Images3DArrayFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[0] * 2, v.input[1], v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 4};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        Halide::Var x, y, c, idx;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, input_width}, {0, input_height}});

        output(x, y, c) = Halide::select(
            x < input_width,
            input_wrapper(x, y, c, 0),
            input_wrapper(x - input_width, y, c, 1));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile2Images3DArrayUInt8HWC : public BuildingBlock<Tile2Images3DArrayUInt8HWC> {
    // Currently available only 2x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile2Images3DArrayUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[0], v.input[1] * 2, v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 4};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        Halide::Var x, y, c;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{Halide::Expr(), Halide::Expr()}, {0, input_width}, {0, input_height}});

        output(c, x, y) = Halide::select(
            x < input_width,
            input_wrapper(c, x, y, 0),
            input_wrapper(c, x - input_width, y, 1));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile4Images3DFloat : public BuildingBlock<Tile4Images3DFloat> {
    // Currently available only 2x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile4Images3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [Math.max(v.input0[0], v.input1[0], v.input2[0], v.input3[0]) * 2, Math.max(v.input0[1], v.input1[1], v.input2[1], v.input3[1]) * 2, Math.max(v.input0[2], v.input1[2], v.input2[2], v.input3[2])] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        Halide::Var x, y;

        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{0, input3_width}, {0, input3_height}});

        Halide::Expr size_x = std::max<int32_t>({input0_width, input1_width, input2_width, input3_width});
        Halide::Expr size_y = std::max<int32_t>({input0_height, input1_height, input2_height, input3_height});

        output(x, y, Halide::_) = Halide::select(
            y < size_y,
            Halide::select(
                x < size_x,
                input0_wrapper(x, y, Halide::_),
                input1_wrapper(x - size_x, y, Halide::_)),
            Halide::select(
                x < size_x,
                input2_wrapper(x, y - size_y, Halide::_),
                input3_wrapper(x - size_x, y - size_y, Halide::_)));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile4Images3DUInt8HWC : public BuildingBlock<Tile4Images3DUInt8HWC> {
    // Currently available only 2x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile4Images3DUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [Math.max(v.input0[0], v.input1[0], v.input2[0], v.input3[0]), Math.max(v.input0[1], v.input1[1], v.input2[1], v.input3[1]) * 2, Math.max(v.input0[2], v.input1[2], v.input2[2], v.input3[2]) * 2] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        Halide::Var x, y, c;

        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{Halide::Expr(), Halide::Expr()}, {0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{Halide::Expr(), Halide::Expr()}, {0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{Halide::Expr(), Halide::Expr()}, {0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{Halide::Expr(), Halide::Expr()}, {0, input3_width}, {0, input3_height}});

        Halide::Expr size_x = std::max<int32_t>({input0_width, input1_width, input2_width, input3_width});
        Halide::Expr size_y = std::max<int32_t>({input0_height, input1_height, input2_height, input3_height});

        output(c, x, y) = Halide::select(
            y < size_y,
            Halide::select(
                x < size_x,
                input0_wrapper(c, x, y),
                input1_wrapper(c, x - size_x, y)),
            Halide::select(
                x < size_x,
                input2_wrapper(c, x, y - size_y),
                input3_wrapper(c, x - size_x, y - size_y)));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile4Images3DArrayFloat : public BuildingBlock<Tile4Images3DArrayFloat> {
    // Currently available only 2x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile4Images3DArrayFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[0] * 2, v.input[1] * 2, v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 4};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        Halide::Var x, y, c, idx;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, input_width}, {0, input_height}});

        output(x, y, c) = Halide::select(
            y < input_height,
            Halide::select(
                x < input_width,
                input_wrapper(x, y, c, 0),
                input_wrapper(x - input_width, y, c, 1)),
            Halide::select(
                x < input_width,
                input_wrapper(x, y - input_height, c, 2),
                input_wrapper(x - input_width, y - input_height, c, 3)));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile4Images3DArrayUInt8HWC : public BuildingBlock<Tile4Images3DArrayUInt8HWC> {
    // Currently available only 2x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile4Images3DArrayUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[0], v.input[1] * 2, v.input[2] * 2] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 4};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        Halide::Var x, y, c;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{Halide::Expr(), Halide::Expr()}, {0, input_width}, {0, input_height}});

        output(c, x, y) = Halide::select(
            y < input_height,
            Halide::select(
                x < input_width,
                input_wrapper(c, x, y, 0),
                input_wrapper(c, x - input_width, y, 1)),
            Halide::select(
                x < input_width,
                input_wrapper(c, x, y - input_height, 2),
                input_wrapper(c, x - input_width, y - input_height, 3)));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile6Images3DFloat : public BuildingBlock<Tile6Images3DFloat> {
    // Currently available only 3x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile6Images3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [Math.max(v.input0[0], v.input1[0], v.input2[0], v.input3[0], v.input4[0], v.input5[0]) * 3, Math.max(v.input0[1], v.input1[1], v.input2[1], v.input3[1], v.input4[1], v.input5[1]) * 2, Math.max(v.input0[2], v.input1[2], v.input2[2], v.input3[2], v.input4[2], v.input5[2])] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height,input4_width,input4_height,input5_width,input5_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorParam<int32_t> input4_width{"input4_width", 0};
    GeneratorParam<int32_t> input4_height{"input4_height", 0};
    GeneratorParam<int32_t> input5_width{"input5_width", 0};
    GeneratorParam<int32_t> input5_height{"input5_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input4{"input4", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input5{"input5", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        Halide::Var x, y;

        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{0, input3_width}, {0, input3_height}});
        Halide::Func input4_wrapper = Halide::BoundaryConditions::constant_exterior(input4, 0, {{0, input4_width}, {0, input4_height}});
        Halide::Func input5_wrapper = Halide::BoundaryConditions::constant_exterior(input5, 0, {{0, input5_width}, {0, input5_height}});

        Halide::Expr size_x = std::max<int32_t>({input0_width, input1_width, input2_width, input3_width, input4_width, input5_width});
        Halide::Expr size_y = std::max<int32_t>({input0_height, input1_height, input2_height, input3_height, input4_height, input5_height});

        output(x, y, Halide::_) = Halide::select(
            y < size_y,
            Halide::select(
                x < size_x,
                input0_wrapper(x, y, Halide::_),
                Halide::select(
                    x < size_x * 2,
                    input1_wrapper(x - size_x, y, Halide::_),
                    input2_wrapper(x - size_x * 2, y, Halide::_))),
            Halide::select(
                x < size_x,
                input3_wrapper(x, y - size_y, Halide::_),
                Halide::select(
                    x < size_x * 2,
                    input4_wrapper(x - size_x, y - size_y, Halide::_),
                    input5_wrapper(x - size_x * 2, y - size_y, Halide::_))));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile6Images3DUInt8HWC : public BuildingBlock<Tile6Images3DUInt8HWC> {
    // Currently available only 3x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile6Images3DUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [Math.max(v.input0[0], v.input1[0], v.input2[0], v.input3[0], v.input4[0], v.input5[0]), Math.max(v.input0[1], v.input1[1], v.input2[1], v.input3[1], v.input4[1], v.input5[1]) * 3, Math.max(v.input0[2], v.input1[2], v.input2[2], v.input3[2], v.input4[2], v.input5[2]) * 2] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height,input4_width,input4_height,input5_width,input5_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorParam<int32_t> input4_width{"input4_width", 0};
    GeneratorParam<int32_t> input4_height{"input4_height", 0};
    GeneratorParam<int32_t> input5_width{"input5_width", 0};
    GeneratorParam<int32_t> input5_height{"input5_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input4{"input4", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input5{"input5", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        Halide::Var x, y, c;

        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{Halide::Expr(), Halide::Expr()}, {0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{Halide::Expr(), Halide::Expr()}, {0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{Halide::Expr(), Halide::Expr()}, {0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{Halide::Expr(), Halide::Expr()}, {0, input3_width}, {0, input3_height}});
        Halide::Func input4_wrapper = Halide::BoundaryConditions::constant_exterior(input4, 0, {{Halide::Expr(), Halide::Expr()}, {0, input4_width}, {0, input4_height}});
        Halide::Func input5_wrapper = Halide::BoundaryConditions::constant_exterior(input5, 0, {{Halide::Expr(), Halide::Expr()}, {0, input5_width}, {0, input5_height}});

        Halide::Expr size_x = std::max<int32_t>({input0_width, input1_width, input2_width, input3_width, input4_width, input5_width});
        Halide::Expr size_y = std::max<int32_t>({input0_height, input1_height, input2_height, input3_height, input4_height, input5_height});

        output(c, x, y) = Halide::select(
            y < size_y,
            Halide::select(
                x < size_x,
                input0_wrapper(c, x, y),
                Halide::select(
                    x < size_x * 2,
                    input1_wrapper(c, x - size_x, y),
                    input2_wrapper(c, x - size_x * 2, y))),
            Halide::select(
                x < size_x,
                input3_wrapper(c, x, y - size_y),
                Halide::select(
                    x < size_x * 2,
                    input4_wrapper(c, x - size_x, y - size_y),
                    input5_wrapper(c, x - size_x * 2, y - size_y))));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile6Images3DArrayFloat : public BuildingBlock<Tile6Images3DArrayFloat> {
    // Currently available only 2x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile6Images3DArrayFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[0] * 3, v.input[1] * 2, v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 4};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        Halide::Var x, y, c, idx;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, input_width}, {0, input_height}});

        output(x, y, c) = Halide::select(
            y < input_height,
            Halide::select(
                x < input_width,
                input_wrapper(x, y, c, 0),
                Halide::select(
                    x < input_width * 2,
                    input_wrapper(x - input_width, y, c, 1),
                    input_wrapper(x - input_width * 2, y, c, 2))),
            Halide::select(
                x < input_width,
                input_wrapper(x, y - input_height, c, 3),
                Halide::select(
                    x < input_width * 2,
                    input_wrapper(x - input_width, y - input_height, c, 4),
                    input_wrapper(x - input_width * 2, y - input_height, c, 5))));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

class Tile6Images3DArrayUInt8HWC : public BuildingBlock<Tile6Images3DArrayUInt8HWC> {
    // Currently available only 3x2 layout
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Tile6Images3DArrayUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Tile images."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[0], v.input[1] * 3, v.input[2] * 2] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 4};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        Halide::Var x, y, c;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{Halide::Expr(), Halide::Expr()}, {0, input_width}, {0, input_height}});

        output(c, x, y) = Halide::select(
            y < input_height,
            Halide::select(
                x < input_width,
                input_wrapper(c, x, y, 0),
                Halide::select(
                    x < input_width * 2,
                    input_wrapper(c, x - input_width, y, 1),
                    input_wrapper(c, x - input_width * 2, y, 2))),
            Halide::select(
                x < input_width,
                input_wrapper(c, x, y - input_height, 3),
                Halide::select(
                    x < input_width * 2,
                    input_wrapper(c, x - input_width, y - input_height, 4),
                    input_wrapper(c, x - input_width * 2, y - input_height, 5))));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            Halide::Var xi, yi;
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

template<typename X, typename T, int32_t D>
class Pack : public BuildingBlock<X> {
    static_assert(D < 4, "D must be less than 4.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.concat([1]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D + 1};

    void generate() {
        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input(Halide::_),
            0);
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Pack3DFloat : public Pack<Pack3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack3DFloat"};
};

class Pack2Images3DFloat : public BuildingBlock<Pack2Images3DFloat> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack2Images3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => Math.max(x, v.input1[i])).concat([2]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 4};

    void generate() {
        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{0, input1_width}, {0, input1_height}});

        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input0_wrapper(Halide::_),
            input1_wrapper(Halide::_));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Pack2Images3DUInt8HWC : public BuildingBlock<Pack2Images3DUInt8HWC> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack2Images3DUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => Math.max(x, v.input1[i])).concat([2]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 4};

    void generate() {
        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{Halide::Expr(), Halide::Expr()}, {0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{Halide::Expr(), Halide::Expr()}, {0, input1_width}, {0, input1_height}});

        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input0_wrapper(Halide::_),
            input1_wrapper(Halide::_));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Pack4Images3DFloat : public BuildingBlock<Pack4Images3DFloat> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack4Images3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => Math.max(x, v.input1[i], v.input2[i], v.input3[i])).concat([4]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 4};

    void generate() {
        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{0, input3_width}, {0, input3_height}});

        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input0_wrapper(Halide::_),
            Halide::select(
                idx == 1,
                input1_wrapper(Halide::_),
                Halide::select(
                    idx == 2,
                    input2_wrapper(Halide::_),
                    input3_wrapper(Halide::_))));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Pack4Images3DUInt8HWC : public BuildingBlock<Pack4Images3DUInt8HWC> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack4Images3DUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => Math.max(x, v.input1[i], v.input2[i], v.input3[i])).concat([4]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 4};

    void generate() {
        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{Halide::Expr(), Halide::Expr()}, {0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{Halide::Expr(), Halide::Expr()}, {0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{Halide::Expr(), Halide::Expr()}, {0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{Halide::Expr(), Halide::Expr()}, {0, input3_width}, {0, input3_height}});

        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input0_wrapper(Halide::_),
            Halide::select(
                idx == 1,
                input1_wrapper(Halide::_),
                Halide::select(
                    idx == 2,
                    input2_wrapper(Halide::_),
                    input3_wrapper(Halide::_))));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Pack6Images3DFloat : public BuildingBlock<Pack6Images3DFloat> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack6Images3DFloat"};
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => Math.max(x, v.input1[i], v.input2[i], v.input3[i], v.input4[i], v.input5[i])).concat([6]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height,input4_width,input4_height,input5_width,input5_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorParam<int32_t> input4_width{"input4_width", 0};
    GeneratorParam<int32_t> input4_height{"input4_height", 0};
    GeneratorParam<int32_t> input5_width{"input5_width", 0};
    GeneratorParam<int32_t> input5_height{"input5_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input4{"input4", Halide::type_of<float>(), 3};
    GeneratorInput<Halide::Func> input5{"input5", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 4};

    void generate() {
        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{0, input3_width}, {0, input3_height}});
        Halide::Func input4_wrapper = Halide::BoundaryConditions::constant_exterior(input4, 0, {{0, input4_width}, {0, input4_height}});
        Halide::Func input5_wrapper = Halide::BoundaryConditions::constant_exterior(input5, 0, {{0, input5_width}, {0, input5_height}});

        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input0_wrapper(Halide::_),
            Halide::select(
                idx == 1,
                input1_wrapper(Halide::_),
                Halide::select(
                    idx == 2,
                    input2_wrapper(Halide::_),
                    Halide::select(
                        idx == 3,
                        input3_wrapper(Halide::_),
                        Halide::select(
                            idx == 4,
                            input4_wrapper(Halide::_),
                            input5_wrapper(Halide::_))))));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Pack6Images3DUInt8HWC : public BuildingBlock<Pack6Images3DUInt8HWC> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Pack6Images3DUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Pack data to array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => Math.max(x, v.input1[i], v.input2[i], v.input3[i], v.input4[i], v.input5[i])).concat([6]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height,input2_width,input2_height,input3_width,input3_height,input4_width,input4_height,input5_width,input5_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorParam<int32_t> input2_width{"input2_width", 0};
    GeneratorParam<int32_t> input2_height{"input2_height", 0};
    GeneratorParam<int32_t> input3_width{"input3_width", 0};
    GeneratorParam<int32_t> input3_height{"input3_height", 0};
    GeneratorParam<int32_t> input4_width{"input4_width", 0};
    GeneratorParam<int32_t> input4_height{"input4_height", 0};
    GeneratorParam<int32_t> input5_width{"input5_width", 0};
    GeneratorParam<int32_t> input5_height{"input5_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input2{"input2", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input3{"input3", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input4{"input4", Halide::type_of<uint8_t>(), 3};
    GeneratorInput<Halide::Func> input5{"input5", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 4};

    void generate() {
        Halide::Func input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, {{Halide::Expr(), Halide::Expr()}, {0, input0_width}, {0, input0_height}});
        Halide::Func input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, {{Halide::Expr(), Halide::Expr()}, {0, input1_width}, {0, input1_height}});
        Halide::Func input2_wrapper = Halide::BoundaryConditions::constant_exterior(input2, 0, {{Halide::Expr(), Halide::Expr()}, {0, input2_width}, {0, input2_height}});
        Halide::Func input3_wrapper = Halide::BoundaryConditions::constant_exterior(input3, 0, {{Halide::Expr(), Halide::Expr()}, {0, input3_width}, {0, input3_height}});
        Halide::Func input4_wrapper = Halide::BoundaryConditions::constant_exterior(input4, 0, {{Halide::Expr(), Halide::Expr()}, {0, input4_width}, {0, input4_height}});
        Halide::Func input5_wrapper = Halide::BoundaryConditions::constant_exterior(input5, 0, {{Halide::Expr(), Halide::Expr()}, {0, input5_width}, {0, input5_height}});

        output(Halide::_, idx) = Halide::select(
            idx == 0,
            input0_wrapper(Halide::_),
            Halide::select(
                idx == 1,
                input1_wrapper(Halide::_),
                Halide::select(
                    idx == 2,
                    input2_wrapper(Halide::_),
                    Halide::select(
                        idx == 3,
                        input3_wrapper(Halide::_),
                        Halide::select(
                            idx == 4,
                            input4_wrapper(Halide::_),
                            input5_wrapper(Halide::_))))));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

template<typename X, typename T, int32_t D>
class Concat : public BuildingBlock<X> {
    static_assert(D > 0, "D must be greater than 0.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Concat array."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,arithmetic"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.slice(0, -1).map((x, i) => Math.max(x, v.input1[i])).concat([v.input0.slice(-1)[0] + v.input1.slice(-1)[0]]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input0_length{"input0_length", 1};
    GeneratorParam<int32_t> input1_length{"input1_length", 1};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<T>(), D};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        output(Halide::_, idx) = Halide::select(
            idx < input0_length,
            input0(Halide::_, idx),
            Halide::select(
                idx < input0_length + input1_length,
                input1(Halide::_, idx - input0_length),
                0));
    }

    void schedule() {
    }

private:
    Halide::Var idx;
};

class Concat4DFloat : public Concat<Concat4DFloat, float, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Concat4DFloat"};
};

template<typename X, typename T, int32_t D>
class CropImage : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Crop image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> left{"left", 0};
    GeneratorParam<int32_t> top{"top", 0};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Var x, y;

        output_tmp = Halide::Func{static_cast<std::string>(gc_prefix) + "output_tmp"};

        output_tmp(x, y, Halide::_) = input(x + left, y + top, Halide::_);

        output(Halide::_) = output_tmp(Halide::_);
    }

    void schedule() {
        Halide::Var x = output_tmp.args()[0];
        Halide::Var y = output_tmp.args()[1];

        output_tmp.bound(x, 0, width).bound(y, 0, height);
        output_tmp.compute_root();
    }

private:
    Halide::Func output_tmp;
};

class CropImage3DFloat : public CropImage<CropImage3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CropImage3DFloat"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), v.input[2]] }}))"};
};

template<typename X, typename T, int32_t D>
class ShiftImage : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Shift image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "output_width,output_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};
    GeneratorParam<int32_t> output_width{"output_width", 0};
    GeneratorParam<int32_t> output_height{"output_height", 0};

    GeneratorParam<int32_t> shift_x{"shift_x", 0};
    GeneratorParam<int32_t> shift_y{"shift_y", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Var x, y;

        output(x, y, Halide::_) = input(x - shift_x, y - shift_y, Halide::_);
    }

    void schedule() {
    }
};

class ShiftImage3DFloat : public ShiftImage<ShiftImage3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ShiftImage3DFloat"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.output_width), parseInt(v.output_height), v.input[2]] }}))"};
};

template<typename X, typename T, int32_t D>
class PaddingImage : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Padding image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height,output_width,output_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};
    GeneratorParam<int32_t> output_width{"output_width", 0};
    GeneratorParam<int32_t> output_height{"output_height", 0};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, width}, {0, height}});

        output(Halide::_) = input_wrapper(Halide::_);
    }

    void schedule() {
    }
};

class PaddingImage3DFloat : public PaddingImage<PaddingImage3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "PaddingImage3DFloat"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.output_width), parseInt(v.output_height), v.input[2]] }}))"};
};

template<typename X, typename T, int32_t D>
class FitImageToCenter : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Fit image to center."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.output_width), parseInt(v.output_height), v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height,output_width,output_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorParam<int32_t> output_width{"output_width", 0};
    GeneratorParam<int32_t> output_height{"output_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, input_width}, {0, input_height}});
        Halide::Expr left = Halide::cast<int32_t>((output_width - input_width) / 2);
        Halide::Expr top = Halide::cast<int32_t>((output_height - input_height) / 2);

        output(x, y, Halide::_) = input_wrapper(x - left, y - top, Halide::_);
    }

    void schedule() {
    }
};

class FitImageToCenter3DFloat : public FitImageToCenter<FitImageToCenter3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FitImageToCenter3DFloat"};
};

class FitImageToCenter3DUInt8 : public FitImageToCenter<FitImageToCenter3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FitImageToCenter3DUInt8"};
};

template<typename X, typename T>
class ReorderImageHWC2CHW : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Reorder image from HWC to CHW."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[1], v.input[2], v.input[0]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 3};

    void generate() {
        Halide::Var x, y, c;
        output(x, y, c) = input(c, x, y);
    }

    void schedule() {
    }
};

class ReorderImageHWC2CHWFloat : public ReorderImageHWC2CHW<ReorderImageHWC2CHWFloat, float> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderImageHWC2CHWFloat"};
};

template<typename X, typename T>
class ReorderImageCHW2HWC : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Reorder image from CHW to HWC."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [v.input[2], v.input[0], v.input[1]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 3};

    void generate() {
        Halide::Var x, y, c;
        output(c, x, y) = input(x, y, c);
    }

    void schedule() {
    }
};

class ReorderImageCHW2HWCFloat : public ReorderImageCHW2HWC<ReorderImageCHW2HWCFloat, float> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderImageCHW2HWCFloat"};
};

template<typename X, typename T>
class ReorderColorChannel : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Reorder color channel (RGB <-> BGR)."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 3};

    void generate() {
        Halide::Var x, y, c;
        output(c, x, y) = select(c == 0, input(2, x, y),
                                 c == 1, input(1, x, y),
                                 input(0, x, y));
    }

    void schedule() {
    }
};

class ReorderColorChannelFloat : public ReorderColorChannel<ReorderColorChannelFloat, float> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderColorChannelFloat"};
};

class ReorderColorChannelUInt8 : public ReorderColorChannel<ReorderColorChannelUInt8, uint8_t> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderColorChannelUInt8"};
};

class MonoToColorUInt8HWC : public BuildingBlock<MonoToColorUInt8HWC> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "MonoToColorUInt8HWC"};
    GeneratorParam<std::string> gc_description{"gc_description", "Convert mono image to color image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3].concat(v.input) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        Halide::Var c;

        output(c, Halide::_) = input(Halide::_);
    }

    void schedule() {
    }

private:
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::BayerOffset, demo_bayer_offset);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::BayerWhiteBalance, demo_bayer_white_balance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::BayerDemosaicSimple, demo_bayer_demosaic_simple);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::GammaCorrection3D, demo_gamma_correction_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::LensShadingCorrectionLinear, demo_lens_shading_correction_linear);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::CalcLuminance, demo_calc_luminance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ResizeBilinear3D, demo_resize_bilinear_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::BayerDownscaleUInt16, demo_bayer_downscale_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::NormalizeRawImage, demo_normalize_raw_image);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MergeImage3DFloat, demo_merge_image_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile2Images3DArrayFloat, demo_tile_2images_3d_array_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile2Images3DArrayUInt8HWC, demo_tile_2images_3d_array_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile4Images3DFloat, demo_tile_4images_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile4Images3DUInt8HWC, demo_tile_4images_3d_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile4Images3DArrayFloat, demo_tile_4images_3d_array_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile4Images3DArrayUInt8HWC, demo_tile_4images_3d_array_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile6Images3DFloat, demo_tile_6images_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile6Images3DUInt8HWC, demo_tile_6images_3d_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile6Images3DArrayFloat, demo_tile_6images_3d_array_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Tile6Images3DArrayUInt8HWC, demo_tile_6images_3d_array_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack3DFloat, demo_pack_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack2Images3DFloat, demo_pack_2images_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack2Images3DUInt8HWC, demo_pack_2images_3d_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack4Images3DFloat, demo_pack_4images_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack4Images3DUInt8HWC, demo_pack_4images_3d_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack6Images3DFloat, demo_pack_6images_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Pack6Images3DUInt8HWC, demo_pack_6images_3d_uint8_hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::Concat4DFloat, demo_concat_4d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::CropImage3DFloat, demo_crop_image_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ShiftImage3DFloat, demo_shift_image_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::PaddingImage3DFloat, demo_padding_image_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::FitImageToCenter3DFloat, demo_fit_image_to_center_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::FitImageToCenter3DUInt8, demo_fit_image_to_center_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ReorderImageHWC2CHWFloat, demo_reorder_image_hwc2chw_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ReorderImageCHW2HWCFloat, demo_reorder_image_chw2hwc_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ReorderColorChannelFloat, demo_reorder_color_channel_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::ReorderColorChannelUInt8, demo_reorder_color_channel_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MonoToColorUInt8HWC, demo_mono_to_color_uint8_hwc);

namespace ion {
namespace bb {
namespace demo {

template<typename X, typename T, int D>
class Multiply : public ion::BuildingBlock<X> {
public:
    GeneratorParam<std::string> gc_description{"gc_description", "This multiplies specified value."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,arithmetic"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};
    GeneratorParam<T> value{"value", 1};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        using namespace Halide;
        output(_) = input(_) * value;
    }
};

class MultiplyU8x2 : public Multiply<MultiplyU8x2, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Multiply U8x2"};
};

class MultiplyU8x3 : public Multiply<MultiplyU8x3, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Multiply U8x3"};
};

class MultiplyU16x2 : public Multiply<MultiplyU16x2, uint16_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Multiply U16x2"};
};

class MultiplyU16x3 : public Multiply<MultiplyU16x3, uint16_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Multiply U16x3"};
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MultiplyU8x2, demo_multiply_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MultiplyU8x3, demo_multiply_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MultiplyU16x2, demo_multiply_u16x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::MultiplyU16x3, demo_multiply_u16x3);

namespace ion {
namespace bb {
namespace demo {

class IMX219 : public ion::BuildingBlock<IMX219> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "IMX219"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures IMX219 image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "ouput,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3264, 2464] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<int32_t> index{"index", 0};
    GeneratorParam<bool> force_sim_mode{"force_sim_mode", false};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;
        std::string url_str = url;
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {url_buf, cast<int32_t>(index), cast<bool>(force_sim_mode)};
        Func v4l2_imx219(static_cast<std::string>(gc_prefix) + "v4l2_imx219");
        v4l2_imx219.define_extern("ion_bb_demo_v4l2_imx219", params, type_of<uint16_t>(), 2);
        v4l2_imx219.compute_root();

        Var x, y;
        output(x, y) = v4l2_imx219(x, y);
    }
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::IMX219, demo_imx219);

namespace ion {
namespace bb {
namespace demo {

class D435 : public ion::BuildingBlock<D435> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "D435"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures D435 stereo image and depth."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "ouput,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output_l: [1280, 720], output_r: [1280, 720], output_d: [1280, 720] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorOutput<Halide::Func> output_l{"output_l", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output_r{"output_r", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output_d{"output_d", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;
        Func realsense_d435_frameset(static_cast<std::string>(gc_prefix) + "realsense_d435_frameset");
        realsense_d435_frameset.define_extern("ion_bb_demo_realsense_d435_frameset", {}, type_of<uint64_t>(), 0);
        realsense_d435_frameset.compute_root();

        Func realsense_d435_infrared(static_cast<std::string>(gc_prefix) + "realsense_d435_infrared");
        realsense_d435_infrared.define_extern("ion_bb_demo_realsense_d435_infrared", {realsense_d435_frameset}, {type_of<uint8_t>(), type_of<uint8_t>()}, 2);
        realsense_d435_infrared.compute_root();

        Func realsense_d435_depth(static_cast<std::string>(gc_prefix) + "realsense_d435_depth");
        realsense_d435_depth.define_extern("ion_bb_demo_realsense_d435_depth", {realsense_d435_frameset}, type_of<uint16_t>(), 2);
        realsense_d435_depth.compute_root();

        output_l(_) = realsense_d435_infrared(_)[0];
        output_r(_) = realsense_d435_infrared(_)[1];
        output_d(_) = realsense_d435_depth(_);
    }
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::D435, demo_d435);

namespace ion {
namespace bb {
namespace demo {

class GUIDisplay : public ion::BuildingBlock<GUIDisplay> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "GUI Display"};
    GeneratorParam<std::string> gc_description{"gc_description", "This renders RGB image on GUI window."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "ouput,display"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: []  }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self,assume_compute_root"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<int32_t> idx{"idx", 0};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<int> output{"output"};

    void generate() {
        using namespace Halide;
        Func in;
        in(c, x, y) = input(c, x, y);
        in.compute_root();
        if (get_target().has_gpu_feature()) {
            Var xo, yo, xi, yi;
            in.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        } else {
            in.parallel(y);
        }
        std::vector<ExternFuncArgument> params = {in, static_cast<int>(width), static_cast<int>(height), static_cast<int>(idx)};
        Func display(static_cast<std::string>(gc_prefix) + "display");
        display.define_extern("ion_bb_demo_gui_display", params, Int(32), 0);
        display.compute_root();
        output() = display();
    }

    void schedule() {
    }

private:
    Halide::Var c, x, y;
};
}  // namespace demo
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::demo::GUIDisplay, demo_gui_display);

#endif
