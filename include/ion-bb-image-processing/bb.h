#ifndef ION_BB_IMAGE_PROCESSING_BB_H
#define ION_BB_IMAGE_PROCESSING_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace image_processing {

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

class ColorDifference {
public:
    enum class Method {
        PerChannel,
        Average
    };

    static const std::map<std::string, Method> enum_map;

    static Halide::Expr calc(Method method, Halide::Expr r0, Halide::Expr g0, Halide::Expr b0, Halide::Expr r1, Halide::Expr g1, Halide::Expr b1) {
        switch (method) {
        case Method::PerChannel:
            return (r1 - r0) * (r1 - r0) + (g1 - g0) * (g1 - g0) + (b1 - b0) * (b1 - b0);
        case Method::Average: {
            Halide::Expr average0 = (r0 + g0 + b0) / 3;
            Halide::Expr average1 = (r1 + g1 + b1) / 3;
            return (average1 - average0) * (average1 - average0);
        }
        default:
            internal_error << "Unknown ColorDifference method";
        }

        return Halide::Expr();
    }
};

const std::map<std::string, ColorDifference::Method> ColorDifference::enum_map{
    {"PerChannel", ColorDifference::Method::PerChannel},
    {"Average", ColorDifference::Method::Average}};

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

        return Halide::Expr();
    }
};

const std::map<std::string, Luminance::Method> Luminance::enum_map{
    {"Max", Luminance::Method::Max},
    {"Average", Luminance::Method::Average},
    {"SimpleY", Luminance::Method::SimpleY},
    {"Y", Luminance::Method::Y}};

class BoundaryConditions {
public:
    enum class Method {
        RepeatEdge,
        RepeatImage,
        MirrorImage,
        MirrorInterior,
        Zero
    };

    static const std::map<std::string, Method> enum_map;

    static Halide::Func calc(Method method, Halide::Func f, Halide::Expr width, Halide::Expr height) {
        internal_assert(f.dimensions() >= 2) << "Bad func for BoundaryConditions";

        Halide::Region region(f.dimensions(), {Halide::Expr(), Halide::Expr()});
        region[0] = {0, width};
        region[1] = {0, height};

        switch (method) {
        case Method::RepeatEdge:
            return Halide::BoundaryConditions::repeat_edge(f, region);
        case Method::RepeatImage:
            return Halide::BoundaryConditions::repeat_image(f, region);
        case Method::MirrorImage:
            return Halide::BoundaryConditions::mirror_image(f, region);
        case Method::MirrorInterior:
            return Halide::BoundaryConditions::mirror_interior(f, region);
        case Method::Zero:
            return Halide::BoundaryConditions::constant_exterior(f, 0, region);
        default:
            internal_error << "Unknown BoundaryCondition method";
        }

        return Halide::Func();
    }
};

const std::map<std::string, BoundaryConditions::Method> BoundaryConditions::enum_map{
    {"RepeatEdge", BoundaryConditions::Method::RepeatEdge},
    {"RepeatImage", BoundaryConditions::Method::RepeatImage},
    {"MirrorImage", BoundaryConditions::Method::MirrorImage},
    {"MirrorInterior", BoundaryConditions::Method::MirrorInterior},
    {"Zero", BoundaryConditions::Method::Zero}};

// value range is [0, 1]
// lut_size is actual LUT size - 1
// If LUT func has 257 elements, set 256 to lut_size.
Halide::Expr lut_interpolation_float(Halide::Func lut, Halide::Expr value, int32_t lut_size) {
    Halide::Expr index0, index1, diff, coef;
    index0 = Halide::cast(Halide::Int(32), Halide::floor(value * lut_size));
    index1 = Halide::min(index0 + 1, lut_size);
    diff = lut(index1) - lut(index0);
    coef = value - index0;
    return lut(index0) + coef * diff;
}

class BayerOffset : public BuildingBlock<BayerOffset> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerOffset"};
    GeneratorParam<std::string> gc_description{"gc_description", "Offset values of bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorInput<float> offset_r{"offset_r"};
    GeneratorInput<float> offset_g{"offset_g"};
    GeneratorInput<float> offset_b{"offset_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        output(x, y) = Halide::clamp(input(x, y) - Halide::mux(BayerMap::get_color(static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), x, y), {offset_r, offset_g, offset_b}), 0.f, 1.f);
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

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorInput<float> gain_r{"gain_r"};
    GeneratorInput<float> gain_g{"gain_g"};
    GeneratorInput<float> gain_b{"gain_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        output(x, y) = Halide::clamp(input(x, y) * Halide::mux(BayerMap::get_color(static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), x, y), {gain_r, gain_g, gain_b}), 0.f, 1.f);
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
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map(x => x / 2).concat([3]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 3};

    void generate() {
        Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, width}, {0, height}});
        switch (static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern))) {
        case BayerMap::Pattern::RGGB:
            output(x, y, c) = Halide::mux(
                c,
                {input_wrapper(x * 2, y * 2),
                 (input_wrapper(x * 2 + 1, y * 2) + input_wrapper(x * 2, y * 2 + 1)) / 2,
                 input_wrapper(x * 2 + 1, y * 2 + 1)});
            break;
        case BayerMap::Pattern::BGGR:
            output(x, y, c) = Halide::mux(
                c,
                {input_wrapper(x * 2 + 1, y * 2 + 1),
                 (input_wrapper(x * 2 + 1, y * 2) + input_wrapper(x * 2, y * 2 + 1)) / 2,
                 input_wrapper(x * 2, y * 2)});
            break;
        case BayerMap::Pattern::GRBG:
            output(x, y, c) = Halide::mux(
                c,
                {input_wrapper(x * 2 + 1, y * 2),
                 (input_wrapper(x * 2, y * 2) + input_wrapper(x * 2 + 1, y * 2 + 1)) / 2,
                 input_wrapper(x * 2 + 1, y * 2)});
            break;
        case BayerMap::Pattern::GBRG:
            output(x, y, c) = Halide::mux(
                c,
                {input_wrapper(x * 2, y * 2 + 1),
                 (input_wrapper(x * 2, y * 2) + input_wrapper(x * 2 + 1, y * 2 + 1)) / 2,
                 input_wrapper(x * 2 + 1, y * 2)});
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

class BayerDemosaicLinear : public BuildingBlock<BayerDemosaicLinear> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerDemosaicLinear"};
    GeneratorParam<std::string> gc_description{"gc_description", "Demosaic bayer image by linear algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.concat([3]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 3};

    void generate() {
        split(x, y, c) = Halide::select(c == BayerMap::get_color(static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), x, y), input(x, y), 0);
        split_mirror = Halide::BoundaryConditions::mirror_interior(split, {{0, width}, {0, height}});

        Halide::Buffer<float> rb_coef(3, 3);
        rb_coef.set_min(-1, -1);
        rb_coef(-1, -1) = 0.25f;
        rb_coef(0, -1) = 0.5f;
        rb_coef(1, -1) = 0.25f;
        rb_coef(-1, 0) = 0.5f;
        rb_coef(0, 0) = 1.f;
        rb_coef(1, 0) = 0.5f;
        rb_coef(-1, 1) = 0.25f;
        rb_coef(0, 1) = 0.5f;
        rb_coef(1, 1) = 0.25f;

        Halide::Buffer<float> g_coef(3, 3);
        g_coef.set_min(-1, -1);
        g_coef(-1, -1) = 0.f;
        g_coef(0, -1) = 0.25f;
        g_coef(1, -1) = 0.f;
        g_coef(-1, 0) = 0.25f;
        g_coef(0, 0) = 1.f;
        g_coef(1, 0) = 0.25f;
        g_coef(-1, 1) = 0.f;
        g_coef(0, 1) = 0.25f;
        g_coef(1, 1) = 0.f;

        sum(x, y, c) += split_mirror(x + r.x, y + r.y, c) * Halide::select(c == 1, g_coef(r.x, r.y), rb_coef(r.x, r.y));
        output(x, y, c) = sum(x, y, c);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        sum.update().reorder(c, r.x, r.y, x, y).unroll(c).unroll(r.x).unroll(r.y);

        if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
            sum.compute_at(output, xi);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
            sum.compute_at(output, x);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x{"x"}, y{"y"}, c{"c"};
    Halide::Func split{"split"};
    Halide::Func split_mirror{"split_mirror"};
    Halide::Func sum{"sum"};
    Halide::RDom r{-1, 3, -1, 3, "r"};
};

class BayerDemosaicFilter : public BuildingBlock<BayerDemosaicFilter> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerDemosaicFilter"};
    GeneratorParam<std::string> gc_description{"gc_description", "Demosaic bayer image by filter algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.concat([3]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 3};

    void generate() {
        // Generate filters
        std::vector<float> lpf{1 / 16.f, 2 / 16.f, 3 / 16.f, 4 / 16.f, 3 / 16.f, 2 / 16.f, 1 / 16.f};
        std::vector<float> hpf{-1 / 16.f, 2 / 16.f, -3 / 16.f, 4 / 16.f, -3 / 16.f, 2 / 16.f, -1 / 16.f};
        Halide::Buffer<float> fc1(7, 7), fc2v(7, 7), fc2h(7, 7), fl(7, 7);
        fc1.set_min(-3, -3);
        fc2v.set_min(-3, -3);
        fc2h.set_min(-3, -3);
        fl.set_min(-3, -3);
        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                float fc1_value = hpf[x] * hpf[y];
                float fc2v_value = lpf[x] * hpf[y];
                float fc2h_value = hpf[x] * lpf[y];
                fc1(x - 3, y - 3) = fc1_value;
                fc2v(x - 3, y - 3) = fc2v_value;
                fc2h(x - 3, y - 3) = fc2h_value;
                fl(x - 3, y - 3) = (x == 3 && y == 3 ? 1 : 0) - fc1_value - fc2v_value - fc2h_value;
            }
        }

        input_mirror = Halide::BoundaryConditions::mirror_interior(input, {{0, width}, {0, height}});
        f_c1_shifted(x, y) += input_mirror(x + r.x, y + r.y) * fc1(r.x, r.y);
        f_c2v_shifted(x, y) += input_mirror(x + r.x, y + r.y) * fc2v(r.x, r.y);
        f_c2h_shifted(x, y) += input_mirror(x + r.x, y + r.y) * fc2h(r.x, r.y);
        f_l(x, y) += input_mirror(x + r.x, y + r.y) * fl(r.x, r.y);

        Halide::Expr c1_cond, c2v_cond, c2h_cond;

        BayerMap::Pattern pattern = static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern));
        if (pattern == BayerMap::Pattern::RGGB ||
            pattern == BayerMap::Pattern::BGGR) {
            c1_cond = x % 2 != y % 2;
        } else {
            c1_cond = x % 2 == y % 2;
        }
        if (pattern == BayerMap::Pattern::RGGB ||
            pattern == BayerMap::Pattern::GRBG) {
            c2v_cond = y % 2 == 1;
        } else {
            c2v_cond = y % 2 == 0;
        }
        if (pattern == BayerMap::Pattern::RGGB ||
            pattern == BayerMap::Pattern::GBRG) {
            c2h_cond = x % 2 == 1;
        } else {
            c2h_cond = x % 2 == 0;
        }

        f_c1(x, y) = f_c1_shifted(x, y) * Halide::select(c1_cond, -1, 1);
        f_c2v(x, y) = f_c2v_shifted(x, y) * Halide::select(c2v_cond, -1, 1);
        f_c2h(x, y) = f_c2h_shifted(x, y) * Halide::select(c2h_cond, -1, 1);
        f_c2(x, y) = f_c2v(x, y) + f_c2h(x, y);

        output(x, y, c) = Halide::clamp(
            Halide::mux(c,
                        {f_l(x, y) + f_c1(x, y) + f_c2(x, y),
                         f_l(x, y) - f_c1(x, y),
                         f_l(x, y) + f_c1(x, y) - f_c2(x, y)}),
            0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        f_c1_shifted.compute_with(f_l, x);
        f_c2v_shifted.compute_with(f_l, x);
        f_c2h_shifted.compute_with(f_l, x);
        Halide::Stage sum_stage = f_l.update().unroll(r.x).unroll(r.y);
        f_c1_shifted.update().unroll(r.x).unroll(r.y).compute_with(sum_stage, r.x);
        f_c2v_shifted.update().unroll(r.x).unroll(r.y).compute_with(sum_stage, r.x);
        f_c2h_shifted.update().unroll(r.x).unroll(r.y).compute_with(sum_stage, r.x);

        if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 8);

            f_c1_shifted.compute_at(output, xi);
            f_c2v_shifted.compute_at(output, xi);
            f_c2h_shifted.compute_at(output, xi);
            f_l.compute_at(output, xi);
            f_c1.compute_at(output, xi);
            f_c2.compute_at(output, xi);
        } else {
            f_c1_shifted.compute_at(output, x);
            f_c2v_shifted.compute_at(output, x);
            f_c2h_shifted.compute_at(output, x);
            f_l.compute_at(output, x);
            f_c1.compute_at(output, x);
            f_c2.compute_at(output, x);

            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x{"x"}, y{"y"}, c{"c"};
    Halide::RDom r{-3, 7, -3, 7, "r"};
    Halide::Func input_mirror{"input_mirror"};
    Halide::Func f_c1{"f_c1"};
    Halide::Func f_c2{"f_c2"};
    Halide::Func f_c2v{"f_c2v"};
    Halide::Func f_c2h{"f_c2h"};
    Halide::Func f_c1_shifted{"f_c1_shifted"};
    Halide::Func f_c2v_shifted{"f_c2v_shifted"};
    Halide::Func f_c2h_shifted{"f_c2h_shifted"};
    Halide::Func f_l{"f_l"};
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
        output(Halide::_) = Halide::clamp(Halide::fast_pow(input(Halide::_), gamma), 0.f, 1.f);
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

class GammaCorrection2D : public GammaCorrection<GammaCorrection2D, 2> {
    GeneratorParam<std::string> gc_title{"gc_title", "GammaCorrection2D"};
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

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
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

        output(x, y) = input(x, y) * Halide::mux(
                                         BayerMap::get_color(static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), x, y),
                                         {r2 * slope_r + offset_r,
                                          r2 * slope_g + offset_g,
                                          r2 * slope_b + offset_b});
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

class LensShadingCorrectionLUT : public BuildingBlock<LensShadingCorrectionLUT> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "LensShadingCorrectionLUT"};
    GeneratorParam<std::string> gc_description{"gc_description", "Correct lens shading."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> lut_r{"lut_r", Halide::Float(32), 1};
    GeneratorInput<Halide::Func> lut_g{"lut_g", Halide::Float(32), 1};
    GeneratorInput<Halide::Func> lut_b{"lut_b", Halide::Float(32), 1};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        Halide::Expr center_x, center_y, r2;

        center_x = width / Halide::cast<float>(2.f);
        center_y = height / Halide::cast<float>(2.f);
        r2 = ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) / (center_x * center_x + center_y * center_y);

        output(x, y) = input(x, y) * Halide::mux(
                                         BayerMap::get_color(static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), x, y),
                                         {lut_interpolation_float(lut_r, input(x, y), 256),
                                          lut_interpolation_float(lut_g, input(x, y), 256),
                                          lut_interpolation_float(lut_b, input(x, y), 256)});
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

class ColorMatrix : public BuildingBlock<ColorMatrix> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ColorMatrix"};
    GeneratorParam<std::string> gc_description{"gc_description", "Apply color matrix."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorInput<Halide::Func> matrix{"matrix", Halide::Float(32), 2};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 3};

    void generate() {
        sum(x, y, c) += input(x, y, r) * matrix(r, c);
        output(x, y, c) = Halide::clamp(sum(x, y, c), 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        sum.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        sum.update().unroll(r).unroll(c);
        sum.compute_at(output, c);

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
    Halide::Func sum{"sum"};
    Halide::RDom r{0, 3, "r"};
};

class CalcLuminance : public BuildingBlock<CalcLuminance> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CalcLuminance"};
    GeneratorParam<std::string> gc_description{"gc_description", "Calc luminance of image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.slice(0, -1) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<Luminance::Method> luminance_method { "luminance_method", Luminance::Method::Average, Luminance::enum_map };
    GeneratorParam<int32_t> luminance_method{"luminance_method", 3, 0, 3};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        output(x, y) = Luminance::calc(static_cast<Luminance::Method>(static_cast<int32_t>(luminance_method)), input(x, y, 0), input(x, y, 1), input(x, y, 2));
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

class BilateralFilter2D : public BuildingBlock<BilateralFilter2D> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BilateralFilter2D"};
    GeneratorParam<std::string> gc_description{"gc_description", "Bilateral filter."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> window_size{"window_size", 2};  // window_size=2 -> 5x5 window
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<float> coef_color{"coef_color"};
    GeneratorInput<float> coef_space{"coef_space"};
    GeneratorInput<Halide::Func> sigma{"sigma", Halide::Float(32), 2};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 2};

    void generate() {
        Halide::Func input_mirror = Halide::BoundaryConditions::mirror_interior(input, {{0, width}, {0, height}, {0, 3}});
        Halide::Expr color_diff, weight;

        r = {-window_size, window_size * 2 + 1, -window_size, window_size * 2 + 1, "r"};

        color_diff = (input_mirror(x + r.x, y + r.y) - input_mirror(x, y)) * (input_mirror(x + r.x, y + r.y) - input_mirror(x, y));
        sigma_inv(x, y) = 1 / sigma(x, y);
        weight = Halide::exp(-(color_diff * coef_color + (r.x * r.x + r.y * r.y) * coef_space) * sigma_inv(x, y));
        weight_sum(x, y) += weight;
        image_sum(x, y) += input_mirror(x + r.x, y + r.y) * weight;

        output(x, y) = image_sum(x, y) / weight_sum(x, y);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        image_sum.compute_with(weight_sum, x);
        image_sum.update().compute_with(weight_sum.update(), r.x);
        if (window_size <= 3) {
            weight_sum.update().unroll(r.x).unroll(r.y);
            image_sum.update().unroll(r.x).unroll(r.y);
        }

        if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 8);

            sigma_inv.compute_at(output, xi);
            weight_sum.compute_at(output, xi);
            image_sum.compute_at(output, xi);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32)));
            output.parallel(y);

            sigma_inv.compute_at(output, x);
            weight_sum.compute_at(output, x);
            image_sum.compute_at(output, x);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x{"x"}, y{"y"};
    Halide::RDom r;
    Halide::Func sigma_inv{"sigma_inv"};
    Halide::Func weight_sum{"weight_sum"};
    Halide::Func image_sum{"image_sum"};
};

class BilateralFilter3D : public BuildingBlock<BilateralFilter3D> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BilateralFilter3D"};
    GeneratorParam<std::string> gc_description{"gc_description", "Bilateral filter."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    // GeneratorParam<ColorDifference::Method> color_difference_method { "color_difference_method", ColorDifference::Method::Average, ColorDifference::enum_map };
    GeneratorParam<int32_t> color_difference_method{"color_difference_method", 0, 0, 1};
    GeneratorParam<int32_t> window_size{"window_size", 2};  // window_size=2 -> 5x5 window
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<float> coef_color{"coef_color"};
    GeneratorInput<float> coef_space{"coef_space"};
    GeneratorInput<Halide::Func> sigma{"sigma", Halide::Float(32), 2};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), 3};

    void generate() {
        Halide::Func input_mirror = Halide::BoundaryConditions::mirror_interior(input, {{0, width}, {0, height}, {0, 3}});
        Halide::Expr color_diff, weight;

        r = {-window_size, window_size * 2 + 1, -window_size, window_size * 2 + 1, "r"};

        color_diff = ColorDifference::calc(
            static_cast<ColorDifference::Method>(static_cast<int32_t>(color_difference_method)),
            input_mirror(x, y, 0),
            input_mirror(x, y, 1),
            input_mirror(x, y, 2),
            input_mirror(x + r.x, y + r.y, 0),
            input_mirror(x + r.x, y + r.y, 1),
            input_mirror(x + r.x, y + r.y, 2));
        sigma_inv(x, y) = 1 / sigma(x, y);
        weight = Halide::exp(-(color_diff * coef_color + (r.x * r.x + r.y * r.y) * coef_space) * sigma_inv(x, y));
        weight_sum(x, y) += weight;
        image_sum(x, y, c) += input_mirror(x + r.x, y + r.y, c) * weight;

        output(x, y, c) = image_sum(x, y, c) / weight_sum(x, y);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        image_sum.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        image_sum.update().reorder(c, r.x, r.y, x, y).unroll(c);
        output.reorder(c, x, y).bound(c, 0, 3).unroll(c);

        image_sum.compute_with(weight_sum, x);
        image_sum.update().compute_with(weight_sum.update(), r.x);
        if (window_size <= 3) {
            weight_sum.update().unroll(r.x).unroll(r.y);
            image_sum.update().unroll(r.x).unroll(r.y);
        }

        if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 8);

            sigma_inv.compute_at(output, xi);
            weight_sum.compute_at(output, xi);
            image_sum.compute_at(output, xi);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32)));
            output.parallel(y);

            sigma_inv.compute_at(output, x);
            weight_sum.compute_at(output, x);
            image_sum.compute_at(output, x);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Var x{"x"}, y{"y"}, c{"c"};
    Halide::RDom r;
    Halide::Func sigma_inv{"sigma_inv"};
    Halide::Func weight_sum{"weight_sum"};
    Halide::Func image_sum{"image_sum"};
};

template<typename X, int32_t D>
class Convolution : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Image convolution."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    // GeneratorParam<BoundaryConditions::Method> boundary_conditions_method { "boundary_conditions_method", BoundaryConditions::Method::Zero, BoundaryConditions::enum_map };
    GeneratorParam<int32_t> boundary_conditions_method{"boundary_conditions_method", 0, 0, 4};
    GeneratorParam<int32_t> window_size{"window_size", 2};  // window_size=2 -> 5x5 window
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> kernel{"kernel", Halide::Float(32), 2};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), D};

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = BoundaryConditions::calc(static_cast<BoundaryConditions::Method>(static_cast<int32_t>(boundary_conditions_method)), input, width, height);

        r = {-window_size, window_size * 2 + 1, -window_size, window_size * 2 + 1, "r"};
        sum(x, y, Halide::_) += input(x + r.x, y + r.y, Halide::_) * kernel(r.x + window_size, r.y + window_size, Halide::_);
        output(x, y, Halide::_) = sum(x, y, Halide::_);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (D == 3) {
            Halide::Var c = output.args()[2];
            output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
            sum.update().reorder(c, r.x, r.y, x, y).unroll(c);
        }

        if (window_size <= 3) {
            sum.update().unroll(r.x).unroll(r.y);
        }

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
            sum.compute_at(output, xi);
        } else {
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
            sum.compute_at(output, x);
        }

        output.compute_root();
#endif
    }

private:
    Halide::Func sum{"sum"};
    Halide::RDom r;
};

class Convolution2D : public Convolution<Convolution2D, 2> {
    GeneratorParam<std::string> gc_title{"gc_title", "Convolution2D"};
};

class Convolution3D : public Convolution<Convolution3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "Convolution3D"};
};

template<typename X, int32_t D>
class LensDistortionCorrectionLUT : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    // Input image is scaled so that r2 range is [0, 1]
    // p1, p2 is not supported
    // LUT(r2) = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) / output_scale
    // fx = fy = sqrt(width^2 + height^2)
public:
    GeneratorParam<std::string> gc_description{"gc_description", "Correct lens distortion."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<float> cx{"cx"};
    GeneratorInput<float> cy{"cy"};
    GeneratorInput<Halide::Func> lut{"lut", Halide::Float(32), 1};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), D};

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, width}, {0, height}});

        Halide::Expr max_x, max_y, dx, dy, r2, r_coef;
        Halide::Expr map_x, map_y, x0, y0, x1, y1, x_coef, y_coef;

        max_x = Halide::max(cx, Halide::cast<float>(width) - cx);
        max_y = Halide::max(cy, Halide::cast<float>(height) - cy);
        dx = x - cx;
        dy = y - cy;
        r2 = (dx * dx + dy * dy) / (max_x * max_x + max_y * max_y);

        r_coef = lut_interpolation_float(lut, r2, 256);

        map_x = cx + dx * r_coef;
        map_y = cy + dy * r_coef;

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

class LensDistortionCorrectionLUT2D : public LensDistortionCorrectionLUT<LensDistortionCorrectionLUT2D, 2> {
    // GeneratorParam<std::string> gc_title{"gc_title", "LensDistortionCorrectionLUT2D"};
};

class LensDistortionCorrectionLUT3D : public LensDistortionCorrectionLUT<LensDistortionCorrectionLUT3D, 3> {
    // GeneratorParam<std::string> gc_title{"gc_title", "LensDistortionCorrectionLUT3D"};
};

template<typename X, int32_t D>
class LensDistortionCorrectionModel : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    // Output fx, fy is scaled by output_scale
public:
    GeneratorParam<std::string> gc_description{"gc_description", "Correct lens distortion."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<float> k1{"k1"};
    GeneratorInput<float> k2{"k2"};
    GeneratorInput<float> k3{"k3"};
    GeneratorInput<float> p1{"p1"};
    GeneratorInput<float> p2{"p2"};
    GeneratorInput<float> fx{"fx"};
    GeneratorInput<float> fy{"fy"};
    GeneratorInput<float> cx{"cx"};
    GeneratorInput<float> cy{"cy"};
    GeneratorInput<float> output_scale{"output_scale"};
    GeneratorInput<Halide::Func> input{"input", Halide::Float(32), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::Float(32), D};

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{0, width}, {0, height}});

        Halide::Expr sx, sy, r2;
        Halide::Expr map_x, map_y, x0, y0, x1, y1, x_coef, y_coef;

        sx = (x - cx) / (Halide::cast<float>(fx) * output_scale);
        sy = (y - cy) / (Halide::cast<float>(fy) * output_scale);
        r2 = sx * sx + sy * sy;

        map_x = cx + (sx * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + Halide::cast<float>(2) * p1 * sx * sy + p2 * (r2 + 2 * sx * sx)) * fx;
        map_y = cy + (sy * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2 * sy * sy) + Halide::cast<float>(2) * p2 * sx * sy) * fy;

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

class LensDistortionCorrectionModel2D : public LensDistortionCorrectionModel<LensDistortionCorrectionModel2D, 2> {
    GeneratorParam<std::string> gc_title{"gc_title", "LensDistortionCorrectionModel2D"};
};

class LensDistortionCorrectionModel3D : public LensDistortionCorrectionModel<LensDistortionCorrectionModel3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "LensDistortionCorrectionModel3D"};
};

template<typename X, int D>
class ResizeNearest : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Resize image by nearest algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map((x, i) => i < 2 ? Math.floor(x * parseFloat(v.scale)) : x) }}))"};
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
        output(x, y, Halide::_) = input_wrapper(Halide::cast<int32_t>(Halide::floor((x + 0.5f) / scale)), Halide::cast<int32_t>(Halide::floor((y + 0.5f) / scale)), Halide::_);
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

class ResizeNearest2D : public ResizeNearest<ResizeNearest2D, 2> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeNearest2D"};
};

class ResizeNearest3D : public ResizeNearest<ResizeNearest3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeNearest3D"};
};

template<typename X, int32_t D>
class ResizeBilinear : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Resize image by bilinear algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map((x, i) => i < 2 ? Math.floor(x * parseFloat(v.scale)) : x) }}))"};
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

class ResizeBilinear2D : public ResizeBilinear<ResizeBilinear2D, 2> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeBilinear2D"};
};

class ResizeBilinear3D : public ResizeBilinear<ResizeBilinear3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeBilinear3D"};
};

template<typename X, int32_t D>
class ResizeAreaAverage : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Resize image by area average algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map((x, i) => i < 2 ? Math.floor(x * parseFloat(v.scale)) : x) }}))"};
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

        Halide::Expr pix_size, conv_size, start_x, start_y, end_x, end_y, base_x, base_y;
        pix_size = Halide::cast<float>(1.f) / scale;
        conv_size = Halide::cast<int32_t>(Halide::ceil(pix_size)) + 1;
        r = {0, conv_size, 0, conv_size};

        start_x = x / scale;
        start_y = y / scale;
        end_x = start_x + pix_size;
        end_y = start_y + pix_size;
        base_x = Halide::cast<int32_t>(Halide::floor(start_x));
        base_y = Halide::cast<int32_t>(Halide::floor(start_y));

        sum(x, y, Halide::_) += input_wrapper(base_x + r.x, base_y + r.y, Halide::_) *
                                (Halide::clamp(end_x - (base_x + r.x), 0.f, 1.f) - Halide::max(start_x - (base_x + r.x), 0.f)) *
                                (Halide::clamp(end_y - (base_y + r.y), 0.f, 1.f) - Halide::max(start_y - (base_y + r.y), 0.f));

        output(x, y, Halide::_) = sum(x, y, Halide::_) / (pix_size * pix_size);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (D == 3) {
            Halide::Var c = output.args()[2];
            sum.reorder(c, x, y).bound(c, 0, 3).unroll(c);
            sum.update().reorder(c, r.x, r.y, x, y).unroll(c);
            output.reorder(c, x, y).bound(c, 0, 3).unroll(c);
        }

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);

            sum.compute_at(output, xi);
        } else {
            output.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }

private:
    Halide::RDom r;
    Halide::Func sum{"sum"};
};

class ResizeAreaAverage2D : public ResizeAreaAverage<ResizeAreaAverage2D, 2> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeAreaAverage2D"};
};

class ResizeAreaAverage3D : public ResizeAreaAverage<ResizeAreaAverage3D, 3> {
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeAreaAverage3D"};
};

template<typename X, typename T>
class BayerDownscale : public BuildingBlock<X> {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

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
class FitImageToCenter : public BuildingBlock<X> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

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

class FitImageToCenter2DUInt8 : public FitImageToCenter<FitImageToCenter2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FitImageToCenter2DUInt8"};
};

class FitImageToCenter3DUInt8 : public FitImageToCenter<FitImageToCenter3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FitImageToCenter3DUInt8"};
};

class FitImageToCenter2DFloat : public FitImageToCenter<FitImageToCenter2DFloat, float, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FitImageToCenter2DFloat"};
};

class FitImageToCenter3DFloat : public FitImageToCenter<FitImageToCenter3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FitImageToCenter3DFloat"};
};

template<typename X, typename T, int32_t D>
class ReorderColorChannel : public BuildingBlock<X> {
    static_assert(D > 0, "D must be greater than 0.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Reorder color channel (RGB <-> BGR)."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> color_dim{"color_dim", D - 1, 0, D - 1};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        std::vector<Halide::Var> vars(D);
        std::vector<Halide::Expr> base_args(vars.begin(), vars.end());
        std::vector<Halide::Expr> mux_input;
        for (int i = 0; i < 3; i++) {
            std::vector<Halide::Expr> args = base_args;
            args[color_dim] = 2 - i;
            mux_input.push_back(input(args));
        }

        output(vars) = Halide::mux(vars[color_dim], mux_input);
    }

    void schedule() {
    }
};

class ReorderColorChannel3DUInt8 : public ReorderColorChannel<ReorderColorChannel3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderColorChannel3DUInt8"};
};

class ReorderColorChannel3DFloat : public ReorderColorChannel<ReorderColorChannel3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ReorderColorChannel3DFloat"};
};

template<typename X, typename T, int32_t D>
class OverlayImage : public BuildingBlock<X> {
    static_assert(D > 1, "D must be greater than 1.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Overlay image to another image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => i === parseInt(v.x_dim) ? Math.max(parseInt(v.input1_left) + v.input1[i], x) : i === parseInt(v.y_dim) ? Math.max(parseInt(v.input1_top) + v.input1[i], x) : Math.min(x, v.input1[i])) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> x_dim{"x_dim", 0, 0, D - 1};
    GeneratorParam<int32_t> y_dim{"y_dim", 1, 0, D - 1};
    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_left{"input1_left", 0};
    GeneratorParam<int32_t> input1_top{"input1_top", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<T>(), D};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Func input0_wrapper;
        Halide::Func input1_wrapper;

        Halide::Region region(D, {Halide::Expr(), Halide::Expr()});

        region[x_dim] = {0, input0_width};
        region[y_dim] = {0, input0_height};
        input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, region);
        region[x_dim] = {0, input1_width};
        region[y_dim] = {0, input1_height};
        input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, region);

        std::vector<Halide::Var> vars(D);
        Halide::Var x = vars[x_dim];
        Halide::Var y = vars[y_dim];

        std::vector<Halide::Expr> args(vars.begin(), vars.end());
        args[x_dim] -= input1_left;
        args[y_dim] -= input1_top;

        output(vars) = Halide::select(
            x >= input1_left && x < input1_left + input1_width && y >= input1_top && y < input1_top + input1_height,
            input1_wrapper(args),
            input0_wrapper(vars));
    }

    void schedule() {
    }
};

class OverlayImage2DUInt8 : public OverlayImage<OverlayImage2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "OverlayImage2DUInt8"};
};

class OverlayImage3DUInt8 : public OverlayImage<OverlayImage3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "OverlayImage3DUInt8"};
};

class OverlayImage2DFloat : public OverlayImage<OverlayImage2DFloat, float, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "OverlayImage2DFloat"};
};

class OverlayImage3DFloat : public OverlayImage<OverlayImage3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "OverlayImage3DFloat"};
};

template<typename X, typename T, int32_t D>
class TileImageHorizontal : public BuildingBlock<X> {
    static_assert(D > 1, "D must be greater than 1.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Tile two images horizontally."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => i === parseInt(v.x_dim) ? x + v.input1[i] : i === parseInt(v.y_dim) ? Math.max(x, v.input1[i]) : Math.min(x, v.input1[i])) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> x_dim{"x_dim", 0, 0, D - 1};
    GeneratorParam<int32_t> y_dim{"y_dim", 1, 0, D - 1};
    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<T>(), D};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Func input0_wrapper;
        Halide::Func input1_wrapper;

        Halide::Region region(D, {Halide::Expr(), Halide::Expr()});

        region[x_dim] = {0, input0_width};
        region[y_dim] = {0, input0_height};
        input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, region);
        region[x_dim] = {0, input1_width};
        region[y_dim] = {0, input1_height};
        input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, region);

        std::vector<Halide::Var> vars(D);
        Halide::Var x = vars[x_dim];

        std::vector<Halide::Expr> args(vars.begin(), vars.end());
        args[x_dim] -= input0_width;

        output(vars) = Halide::select(
            x >= input0_width,
            input1_wrapper(args),
            input0_wrapper(vars));
    }

    void schedule() {
    }
};

class TileImageHorizontal2DUInt8 : public TileImageHorizontal<TileImageHorizontal2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageHorizontal2DUInt8"};
};

class TileImageHorizontal3DUInt8 : public TileImageHorizontal<TileImageHorizontal3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageHorizontal3DUInt8"};
};

class TileImageHorizontal2DFloat : public TileImageHorizontal<TileImageHorizontal2DFloat, float, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageHorizontal2DFloat"};
};

class TileImageHorizontal3DFloat : public TileImageHorizontal<TileImageHorizontal3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageHorizontal3DFloat"};
};

template<typename X, typename T, int32_t D>
class TileImageVertical : public BuildingBlock<X> {
    static_assert(D > 1, "D must be greater than 1.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Tile two images vertically."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input0.map((x, i) => i === parseInt(v.x_dim) ? Math.max(x, v.input1[i]) : i === parseInt(v.y_dim) ? x + v.input1[i] : Math.min(x, v.input1[i])) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input0_width,input0_height,input1_width,input1_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> x_dim{"x_dim", 0, 0, D - 1};
    GeneratorParam<int32_t> y_dim{"y_dim", 1, 0, D - 1};
    GeneratorParam<int32_t> input0_width{"input0_width", 0};
    GeneratorParam<int32_t> input0_height{"input0_height", 0};
    GeneratorParam<int32_t> input1_width{"input1_width", 0};
    GeneratorParam<int32_t> input1_height{"input1_height", 0};
    GeneratorInput<Halide::Func> input0{"input0", Halide::type_of<T>(), D};
    GeneratorInput<Halide::Func> input1{"input1", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Func input0_wrapper;
        Halide::Func input1_wrapper;

        Halide::Region region(D, {Halide::Expr(), Halide::Expr()});

        region[x_dim] = {0, input0_width};
        region[y_dim] = {0, input0_height};
        input0_wrapper = Halide::BoundaryConditions::constant_exterior(input0, 0, region);
        region[x_dim] = {0, input1_width};
        region[y_dim] = {0, input1_height};
        input1_wrapper = Halide::BoundaryConditions::constant_exterior(input1, 0, region);

        std::vector<Halide::Var> vars(D);
        Halide::Var y = vars[y_dim];

        std::vector<Halide::Expr> args(vars.begin(), vars.end());
        args[y_dim] -= input0_height;

        output(vars) = Halide::select(
            y >= input0_height,
            input1_wrapper(args),
            input0_wrapper(vars));
    }

    void schedule() {
    }
};

class TileImageVertical2DUInt8 : public TileImageVertical<TileImageVertical2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageVertical2DUInt8"};
};

class TileImageVertical3DUInt8 : public TileImageVertical<TileImageVertical3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageVertical3DUInt8"};
};

class TileImageVertical2DFloat : public TileImageVertical<TileImageVertical2DFloat, float, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageVertical2DFloat"};
};

class TileImageVertical3DFloat : public TileImageVertical<TileImageVertical3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "TileImageVertical3DFloat"};
};

template<typename X, typename T, int32_t D>
class CropImage : public BuildingBlock<X> {
    static_assert(D > 1, "D must be greater than 1.");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Crop image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map((x, i) => i === parseInt(v.x_dim) ? parseInt(v.output_width) : i === parseInt(v.y_dim) ? parseInt(v.output_height) : x) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height,output_width,output_height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inline"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> x_dim{"x_dim", 0, 0, D - 1};
    GeneratorParam<int32_t> y_dim{"y_dim", 1, 0, D - 1};
    GeneratorParam<int32_t> input_width{"input_width", 0};
    GeneratorParam<int32_t> input_height{"input_height", 0};
    GeneratorParam<int32_t> left{"left", 0};
    GeneratorParam<int32_t> top{"top", 0};
    GeneratorParam<int32_t> output_width{"output_width", 0};
    GeneratorParam<int32_t> output_height{"output_height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};

    void generate() {
        Halide::Func input_wrapper;

        Halide::Region region(D, {Halide::Expr(), Halide::Expr()});

        region[x_dim] = {0, input_width};
        region[y_dim] = {0, input_height};
        input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, region);

        std::vector<Halide::Var> vars(D);
        Halide::Var x = vars[x_dim];
        Halide::Var y = vars[y_dim];

        std::vector<Halide::Expr> args(vars.begin(), vars.end());
        args[x_dim] += left;
        args[y_dim] += top;

        output(vars) = Halide::select(
            x >= 0 && x < output_width && y >= 0 && y < output_height,
            input_wrapper(args),
            0);
    }

    void schedule() {
    }
};

class CropImage2DUInt8 : public CropImage<CropImage2DUInt8, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CropImage2DUInt8"};
};

class CropImage3DUInt8 : public CropImage<CropImage3DUInt8, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CropImage3DUInt8"};
};

class CropImage2DFloat : public CropImage<CropImage2DFloat, float, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CropImage2DFloat"};
};

class CropImage3DFloat : public CropImage<CropImage3DFloat, float, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CropImage3DFloat"};
};

class ColorSpaceConverterRGBToHSV : public ion::BuildingBlock<ColorSpaceConverterRGBToHSV> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ColorSpaceConverter RGB to HSV"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from RGB into HSV."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        Var x, y, c;
        Expr zero = cast<float>(0.0f);
        Expr one = cast<float>(1.0f);
        Expr two = cast<float>(2.0f);
        Expr four = cast<float>(4.0f);
        Expr six = cast<float>(6.0f);

        Expr r = input(x, y, 0);
        Expr g = input(x, y, 1);
        Expr b = input(x, y, 2);

        Expr minv = min(r, min(g, b));
        Expr maxv = max(r, max(g, b));
        Expr diff = select(maxv == minv, one, maxv - minv);

        Expr h = select(maxv == minv, zero,
                        maxv == r, (g - b) / diff,
                        maxv == g, (b - r) / diff + two,
                        (r - g) / diff + four);

        h = select(h < zero, h + six, h) / six;

        Expr dmaxv = select(maxv == zero, one, maxv);
        Expr s = select(maxv == zero, zero, (maxv - minv) / dmaxv);
        Expr v = maxv;

        output(x, y, c) = select(c == 0, h, c == 1, s, v);
    }
};

class ColorSpaceConverterHSVToRGB : public ion::BuildingBlock<ColorSpaceConverterHSVToRGB> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ColorSpaceConverter HSV to RGB"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from HSV into RGB."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        Var x, y, c;

        Expr zero = cast<float>(0.0f);
        Expr one = cast<float>(1.0f);
        Expr six = cast<float>(6.0f);

        Expr h = input(x, y, 0);
        Expr s = input(x, y, 1);
        Expr v = input(x, y, 2);

        Expr i = cast<int32_t>(floor(six * h));

        Expr c0 = i == 0 || i == 6;
        Expr c1 = i == 1;
        Expr c2 = i == 2;
        Expr c3 = i == 3;
        Expr c4 = i == 4;
        Expr c5 = i == 5;

        Expr f = six * h - floor(six * h);

        Expr r = select(s > zero,
                        select(c0, v,
                               c1, v * (one - s * f),
                               c2, v * (one - s),
                               c3, v * (one - s),
                               c4, v * (one - s * (one - f)),
                               v),
                        v);
        Expr g = select(s > zero,
                        select(c0, v * (one - s * (one - f)),
                               c1, v,
                               c2, v,
                               c3, v * (one - s * f),
                               c4, v * (one - s),
                               v * (one - s)),
                        v);

        Expr b = select(s > zero,
                        select(c0, v * (one - s),
                               c1, v * (one - s),
                               c2, v * (one - s * (one - f)),
                               c3, v,
                               c4, v,
                               v * (one - s * f)),
                        v);

        output(x, y, c) = select(c == 0, r, c == 1, g, b);
    }
};

class ColorAdjustment : public ion::BuildingBlock<ColorAdjustment> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Color Adjustment"};
    GeneratorParam<std::string> gc_description{"gc_description", "This applies color adjustment."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "target_color"};

    GeneratorParam<float> adjustment_value{"adjustment_value", 1.0f};
    GeneratorParam<int32_t> target_color{"target_channel", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;
        Var x, y, c;

        Expr zero = cast<float>(0.0f);
        Expr one = cast<float>(1.0f);

        Expr v = input(x, y, c);

        output(x, y, c) = select(c == target_color, clamp(fast_pow(v, cast<float>(adjustment_value)), zero, one), v);
    }
};

}  // namespace image_processing
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BayerOffset, image_processing_bayer_offset);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BayerWhiteBalance, image_processing_bayer_white_balance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BayerDemosaicSimple, image_processing_bayer_demosaic_simple);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BayerDemosaicLinear, image_processing_bayer_demosaic_linear);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BayerDemosaicFilter, image_processing_bayer_demosaic_filter);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::GammaCorrection2D, image_processing_gamma_correction_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::GammaCorrection3D, image_processing_gamma_correction_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::LensShadingCorrectionLinear, image_processing_lens_shading_correction_linear);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::LensShadingCorrectionLUT, image_processing_lens_shading_correction_lut);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ColorMatrix, image_processing_color_matrix);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::CalcLuminance, image_processing_calc_luminance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BilateralFilter2D, image_processing_bilateral_filter_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BilateralFilter3D, image_processing_bilateral_filter_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::Convolution2D, image_processing_convolution_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::Convolution3D, image_processing_convolution_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::LensDistortionCorrectionLUT2D, image_processing_lens_distortion_correction_lut_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::LensDistortionCorrectionLUT3D, image_processing_lens_distortion_correction_lut_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::LensDistortionCorrectionModel2D, image_processing_lens_distortion_correction_model_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::LensDistortionCorrectionModel3D, image_processing_lens_distortion_correction_model_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ResizeNearest2D, image_processing_resize_nearest_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ResizeNearest3D, image_processing_resize_nearest_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ResizeBilinear2D, image_processing_resize_bilinear_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ResizeBilinear3D, image_processing_resize_bilinear_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ResizeAreaAverage2D, image_processing_resize_area_average_2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ResizeAreaAverage3D, image_processing_resize_area_average_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::BayerDownscaleUInt16, image_processing_bayer_downscale_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::NormalizeRawImage, image_processing_normalize_raw_image);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::FitImageToCenter2DUInt8, image_processing_fit_image_to_center_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::FitImageToCenter3DUInt8, image_processing_fit_image_to_center_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::FitImageToCenter2DFloat, image_processing_fit_image_to_center_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::FitImageToCenter3DFloat, image_processing_fit_image_to_center_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ReorderColorChannel3DUInt8, image_processing_reorder_color_channel_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ReorderColorChannel3DFloat, image_processing_reorder_color_channel_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::OverlayImage2DUInt8, image_processing_overlay_image_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::OverlayImage3DUInt8, image_processing_overlay_image_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::OverlayImage2DFloat, image_processing_overlay_image_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::OverlayImage3DFloat, image_processing_overlay_image_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageHorizontal2DUInt8, image_processing_tile_image_horizontal_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageHorizontal3DUInt8, image_processing_tile_image_horizontal_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageHorizontal2DFloat, image_processing_tile_image_horizontal_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageHorizontal3DFloat, image_processing_tile_image_horizontal_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageVertical2DUInt8, image_processing_tile_image_vertical_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageVertical3DUInt8, image_processing_tile_image_vertical_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageVertical2DFloat, image_processing_tile_image_vertical_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::TileImageVertical3DFloat, image_processing_tile_image_vertical_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::CropImage2DUInt8, image_processing_crop_image_2d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::CropImage3DUInt8, image_processing_crop_image_3d_uint8);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::CropImage2DFloat, image_processing_crop_image_2d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::CropImage3DFloat, image_processing_crop_image_3d_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ColorSpaceConverterRGBToHSV, image_processing_color_space_converter_rgb_to_hsv);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ColorSpaceConverterHSVToRGB, image_processing_color_space_converter_hsv_to_rgb);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_processing::ColorAdjustment, image_processing_color_adjustment);

#endif
