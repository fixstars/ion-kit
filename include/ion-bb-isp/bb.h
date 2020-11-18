#ifndef ION_BB_ISP_BB_H
#define ION_BB_ISP_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace isp {

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
                get_color(pat, 1)
            ),
            Halide::select(
                x % 2 == 0,
                get_color(pat, 2),
                get_color(pat, 3)
            )
        );
    }

private:
    static const int bayer_map[4][4];

    static int get_color(Pattern pat, int pos) {
        return bayer_map[static_cast<int>(pat)][pos];
    }
};

const std::map<std::string, BayerMap::Pattern> BayerMap::enum_map {
    { "RGGB", BayerMap::Pattern::RGGB },
    { "BGGR", BayerMap::Pattern::BGGR },
    { "GRBG", BayerMap::Pattern::GRBG },
    { "GBRG", BayerMap::Pattern::GBRG }
};

const int BayerMap::bayer_map[4][4] {
    { 0, 1, 1, 2}, // RGGB
    { 2, 1, 1, 0}, // BGGR
    { 1, 0, 2, 1}, // GRBG
    { 1, 2, 0, 1}  // GBRG
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
        case Method::Average:
        {
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

const std::map<std::string, ColorDifference::Method> ColorDifference::enum_map {
    { "PerChannel", ColorDifference::Method::PerChannel },
    { "Average", ColorDifference::Method::Average }
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
            return r * 0.2126f + g * 0.7152f + b * 0.0722f; // BT.709
        default:
            internal_error << "Unknown Luminance method";
        }

        return Halide::Expr();
    }
};

const std::map<std::string, Luminance::Method> Luminance::enum_map {
    { "Max", Luminance::Method::Max },
    { "Average", Luminance::Method::Average },
    { "SimpleY", Luminance::Method::SimpleY },
    { "Y", Luminance::Method::Y }
};

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

    static Halide::Func calc2D(Method method, Halide::Func f, Halide::Expr width, Halide::Expr height) {
        switch (method) {
        case Method::RepeatEdge:
            return Halide::BoundaryConditions::repeat_edge(f, {{ 0, width }, { 0, height }});
        case Method::RepeatImage:
            return Halide::BoundaryConditions::repeat_image(f, {{ 0, width }, { 0, height }});
        case Method::MirrorImage:
            return Halide::BoundaryConditions::mirror_image(f, {{ 0, width }, { 0, height }});
        case Method::MirrorInterior:
            return Halide::BoundaryConditions::mirror_interior(f, {{ 0, width }, { 0, height }});
        case Method::Zero:
            return Halide::BoundaryConditions::constant_exterior(f, 0, {{ 0, width }, { 0, height }});
        default:
            internal_error << "Unknown BoundaryCondition method";
        }

        return Halide::Func();
    }

    static Halide::Func calc3D(Method method, Halide::Func f, Halide::Expr width, Halide::Expr height) {
        switch (method) {
        case Method::RepeatEdge:
            return Halide::BoundaryConditions::repeat_edge(f, {{ 0, width }, { 0, height }, {0, 3}});
        case Method::RepeatImage:
            return Halide::BoundaryConditions::repeat_image(f, {{ 0, width }, { 0, height }, {0, 3}});
        case Method::MirrorImage:
            return Halide::BoundaryConditions::mirror_image(f, {{ 0, width }, { 0, height }, {0, 3}});
        case Method::MirrorInterior:
            return Halide::BoundaryConditions::mirror_interior(f, {{ 0, width }, { 0, height }, {0, 3}});
        case Method::Zero:
            return Halide::BoundaryConditions::constant_exterior(f, 0, {{ 0, width }, { 0, height }, {0, 3}});
        default:
            internal_error << "Unknown BoundaryCondition method";
        }

        return Halide::Func();
    }
};

const std::map<std::string, BoundaryConditions::Method> BoundaryConditions::enum_map {
    { "RepeatEdge", BoundaryConditions::Method::RepeatEdge },
    { "RepeatImage", BoundaryConditions::Method::RepeatImage },
    { "MirrorImage", BoundaryConditions::Method::MirrorImage },
    { "MirrorInterior", BoundaryConditions::Method::MirrorInterior },
    { "Zero", BoundaryConditions::Method::Zero }
};

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

Halide::Expr select_by_color(Halide::Expr color, Halide::Expr r_value, Halide::Expr g_value, Halide::Expr b_value) {
    return Halide::select(
        color == 0,
        r_value,
        Halide::select(
            color == 1,
            g_value,
            b_value
        )
    );
}

class Offset2D : public BuildingBlock<Offset2D> {
public:
    GeneratorInput<float> offset{ "offset" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
        output(x, y) = Halide::clamp(input(x, y) - offset, 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32)));
            output.parallel(y, 16);
        }
        output.compute_root();
#endif
    }
private:
    Halide::Var x, y;
};

class Offset3D : public BuildingBlock<Offset3D> {
public:
    GeneratorInput<float> offset_r{ "offset_r" };
    GeneratorInput<float> offset_g{ "offset_g" };
    GeneratorInput<float> offset_b{ "offset_b" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 3 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 3 };

    void generate() {
        output(x, y, c) = Halide::clamp(input(x, y, c) - select_by_color(c, offset_r, offset_g, offset_b), 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).bound(c, 0, 3).unroll(c);

        if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32)));
            output.parallel(y, 16);
        }
        output.compute_root();
#endif
    }
private:
    Halide::Var x, y, c;
};

class BayerOffset : public BuildingBlock<BayerOffset> {
public:
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<float> offset_r{ "offset_r" };
    GeneratorInput<float> offset_g{ "offset_g" };
    GeneratorInput<float> offset_b{ "offset_b" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2};

    void generate() {
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
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<float> gain_r{ "gain_r" };
    GeneratorInput<float> gain_g{ "gain_g" };
    GeneratorInput<float> gain_b{ "gain_b" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2};

    void generate() {
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

class BayerDemosaicLinear : public BuildingBlock<BayerDemosaicLinear> {
public:
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 3};

    void generate() {
        split(x, y, c) = Halide::select(c == BayerMap::get_color(bayer_pattern, x, y), input(x, y), 0);
        split_mirror = Halide::BoundaryConditions::mirror_interior(split, {{ 0, width }, { 0, height }});

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
        output.reorder(c, x, y).unroll(c);
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
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 3};

    void generate() {
        // Generate filters
        std::vector<float> lpf { 1 / 16.f, 2 / 16.f, 3 / 16.f, 4 / 16.f, 3 / 16.f, 2 / 16.f, 1 / 16.f};
        std::vector<float> hpf { -1 / 16.f, 2 / 16.f, -3 / 16.f, 4 / 16.f, -3 / 16.f, 2 / 16.f, -1 / 16.f};
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

        input_mirror = Halide::BoundaryConditions::mirror_interior(input, {{ 0, width }, { 0, height }});
        f_c1_shifted(x, y) += input_mirror(x + r.x, y + r.y) * fc1(r.x, r.y);
        f_c2v_shifted(x, y) += input_mirror(x + r.x, y + r.y) * fc2v(r.x, r.y);
        f_c2h_shifted(x, y) += input_mirror(x + r.x, y + r.y) * fc2h(r.x, r.y);
        f_l(x, y) += input_mirror(x + r.x, y + r.y) * fl(r.x, r.y);

        Halide::Expr c1_cond, c2v_cond, c2h_cond;

        if (bayer_pattern == BayerMap::Pattern::RGGB ||
            bayer_pattern == BayerMap::Pattern::BGGR) {
            c1_cond = x % 2 != y % 2;
        } else {
            c1_cond = x % 2 == y % 2;
        }
        if (bayer_pattern == BayerMap::Pattern::RGGB ||
            bayer_pattern == BayerMap::Pattern::GRBG) {
            c2v_cond = y % 2 == 1;
        } else {
            c2v_cond = y % 2 == 0;
        }
        if (bayer_pattern == BayerMap::Pattern::RGGB ||
            bayer_pattern == BayerMap::Pattern::GBRG) {
            c2h_cond = x % 2 == 1;
        } else {
            c2h_cond = x % 2 == 0;
        }

        f_c1(x, y) = f_c1_shifted(x, y) * Halide::select(c1_cond, -1 , 1);
        f_c2v(x, y) = f_c2v_shifted(x, y) * Halide::select(c2v_cond, -1 , 1);
        f_c2h(x, y) = f_c2h_shifted(x, y) * Halide::select(c2h_cond, -1 , 1);
        f_c2(x, y) = f_c2v(x, y) + f_c2h(x, y);

        output(x, y, c) = Halide::clamp(select_by_color(c,
            f_l(x, y) + f_c1(x, y) + f_c2(x, y),
            f_l(x, y) - f_c1(x, y),
            f_l(x, y) + f_c1(x, y) - f_c2(x, y)
        ), 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).unroll(c);
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

class BayerDemosaicSimple : public BuildingBlock<BayerDemosaicSimple> {
public:
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 3};

    void generate() {
        switch (bayer_pattern) {
        case BayerMap::Pattern::RGGB:
            output(x, y, c) = select_by_color(
                c,
                input(x * 2, y * 2),
                (input(x * 2 + 1, y * 2) + input(x * 2, y * 2 + 1)) / 2,
                input(x * 2 + 1, y * 2 + 1)
            );
            break;
        case BayerMap::Pattern::BGGR:
            output(x, y, c) = select_by_color(
                c,
                input(x * 2 + 1, y * 2 + 1),
                (input(x * 2 + 1, y * 2) + input(x * 2, y * 2 + 1)) / 2,
                input(x * 2, y * 2)
            );
            break;
        case BayerMap::Pattern::GRBG:
            output(x, y, c) = select_by_color(
                c,
                input(x * 2 + 1, y * 2),
                (input(x * 2, y * 2) + input(x * 2 + 1, y * 2 + 1)) / 2,
                input(x * 2 + 1, y * 2)
            );
            break;
        case BayerMap::Pattern::GBRG:
            output(x, y, c) = select_by_color(
                c,
                input(x * 2, y * 2 + 1),
                (input(x * 2, y * 2) + input(x * 2 + 1, y * 2 + 1)) / 2,
                input(x * 2 + 1, y * 2)
            );
            break;
        default:
            internal_error << "Unknown BayerMap pattern";
        }
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).unroll(c);

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

template<int D>
class GammaCorrection : public BuildingBlock<GammaCorrection<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
public:
    GeneratorInput<float> gamma{ "gamma" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

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

using GammaCorrection2D = GammaCorrection<2>;
using GammaCorrection3D = GammaCorrection<3>;

class ColorMatrix : public BuildingBlock<ColorMatrix> {
public:
    GeneratorInput<Halide::Func> matrix{ "matrix", Halide::Float(32), 2 };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 3 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 3 };

    void generate() {
        sum(x, y, c) += input(x, y, r) * matrix(r, c);
        output(x, y, c) = Halide::clamp(sum(x, y, c), 0.f, 1.f);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        output.reorder(c, x, y).unroll(c);
        sum.reorder(c, x, y).unroll(c);
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

class LensShadingCorrectionLUT : public BuildingBlock<LensShadingCorrectionLUT> {
public:
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<Halide::Func> lut_r{ "lut_r", Halide::Float(32), 1 };
    GeneratorInput<Halide::Func> lut_g{ "lut_g", Halide::Float(32), 1 };
    GeneratorInput<Halide::Func> lut_b{ "lut_b", Halide::Float(32), 1 };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
        Halide::Expr center_x, center_y, r2;

        center_x = width / Halide::cast<float>(2.f);
        center_y = height / Halide::cast<float>(2.f);
        r2 = ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) / (center_x * center_x + center_y * center_y);

        output(x, y) = input(x, y) * select_by_color(
            BayerMap::get_color(bayer_pattern, x, y),
            lut_interpolation_float(lut_r, input(x, y), 256),
            lut_interpolation_float(lut_g, input(x, y), 256),
            lut_interpolation_float(lut_b, input(x, y), 256)
        );
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

class LensShadingCorrectionLinear : public BuildingBlock<LensShadingCorrectionLinear> {
public:
    GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> slope_r{ "slope_r" };
    GeneratorInput<float> slope_g{ "slope_g" };
    GeneratorInput<float> slope_b{ "slope_b" };
    GeneratorInput<float> offset_r{ "offset_r" };
    GeneratorInput<float> offset_g{ "offset_g" };
    GeneratorInput<float> offset_b{ "offset_b" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
        Halide::Expr center_x, center_y, r2;

        center_x = width / Halide::cast<float>(2.f);
        center_y = height / Halide::cast<float>(2.f);
        r2 = ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) / (center_x * center_x + center_y * center_y);

        output(x, y) = input(x, y) * select_by_color(
            BayerMap::get_color(bayer_pattern, x, y),
            r2 * slope_r + offset_r,
            r2 * slope_g + offset_g,
            r2 * slope_b + offset_b
        );
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

class BilateralFilter2D : public BuildingBlock<BilateralFilter2D> {
public:
    GeneratorParam<int32_t> window_size{ "window_size", 2 }; // window_size=2 -> 5x5 window
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> coef_color{ "coef_color" };
    GeneratorInput<float> coef_space{ "coef_space" };
    GeneratorInput<Halide::Func> sigma{ "sigma", Halide::Float(32), 2 };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 2 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
        Halide::Func input_mirror = Halide::BoundaryConditions::mirror_interior(input, {{ 0, width }, { 0, height }, { 0, 3 }});
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
    Halide::Var x{"x"}, y{"y"}, rx{"rx"}, ry{"ry"};
    Halide::RDom r;
    Halide::Func sigma_inv{"sigma_inv"};
    Halide::Func weight_sum{"weight_sum"};
    Halide::Func image_sum{"image_sum"};
};

class BilateralFilter3D : public BuildingBlock<BilateralFilter3D> {
public:
    GeneratorParam<ColorDifference::Method> color_difference_method { "color_difference_method", ColorDifference::Method::Average, ColorDifference::enum_map };
    GeneratorParam<int32_t> window_size{ "window_size", 2 }; // window_size=2 -> 5x5 window
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> coef_color{ "coef_color" };
    GeneratorInput<float> coef_space{ "coef_space" };
    GeneratorInput<Halide::Func> sigma{ "sigma", Halide::Float(32), 2 };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 3 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 3 };

    void generate() {
        Halide::Func input_mirror = Halide::BoundaryConditions::mirror_interior(input, {{ 0, width }, { 0, height }, { 0, 3 }});
        Halide::Expr color_diff, weight;

        r = {-window_size, window_size * 2 + 1, -window_size, window_size * 2 + 1, "r"};

        color_diff = ColorDifference::calc(
            color_difference_method,
            input_mirror(x, y, 0),
            input_mirror(x, y, 1),
            input_mirror(x, y, 2),
            input_mirror(x + r.x, y + r.y, 0),
            input_mirror(x + r.x, y + r.y, 1),
            input_mirror(x + r.x, y + r.y, 2)
        );
        sigma_inv(x, y) = 1 / sigma(x, y);
        weight = Halide::exp(-(color_diff * coef_color + (r.x * r.x + r.y * r.y) * coef_space) * sigma_inv(x, y));
        weight_sum(x, y) += weight;
        image_sum(x, y, c) += input_mirror(x + r.x, y + r.y, c) * weight;

        output(x, y, c) = image_sum(x, y, c) / weight_sum(x, y);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        image_sum.reorder(c, x, y).unroll(c);
        image_sum.update().reorder(c, r.x, r.y, x, y).unroll(c);
        output.reorder(c, x, y).unroll(c);

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
    Halide::Var x{"x"}, y{"y"}, c{"c"}, rx{"rx"}, ry{"ry"};
    Halide::RDom r;
    Halide::Func sigma_inv{"sigma_inv"};
    Halide::Func weight_sum{"weight_sum"};
    Halide::Func image_sum{"image_sum"};
};

class CalcLuminance : public BuildingBlock<CalcLuminance> {
public:
    GeneratorParam<Luminance::Method> luminance_method { "luminance_method", Luminance::Method::Average, Luminance::enum_map };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), 3 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
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

template<int D>
class Filter : public BuildingBlock<Filter<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
public:
    GeneratorParam<BoundaryConditions::Method> boundary_conditions_method { "boundary_conditions_method", BoundaryConditions::Method::Zero, BoundaryConditions::enum_map };
    GeneratorParam<int32_t> window_size{ "window_size", 2 }; // window_size=2 -> 5x5 window
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<Halide::Func> filter{ "filter", Halide::Float(32), 2 };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = BoundaryConditions::calc2D(boundary_conditions_method, input, width, height);

        r = {-window_size, window_size * 2 + 1, -window_size, window_size * 2 + 1, "r"};
        sum(x, y, Halide::_) += input(x + r.x, y + r.y, Halide::_) * filter(r.x + window_size, r.y + window_size, Halide::_);
        output(x, y, Halide::_) = sum(x, y, Halide::_);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (D == 3) {
            Halide::Var c = output.args()[2];
            output.reorder(c, x, y).unroll(c);
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

using Filter2D = Filter<2>;
using Filter3D = Filter<3>;

template<int D>
class LensDistortionCorrectionLUT : public BuildingBlock<LensDistortionCorrectionLUT<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    // Input image is scaled so that r2 range is [0, 1]
    // p1, p2 is not supported
    // LUT(r2) = (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) / output_scale
    // fx = fy = sqrt(width^2 + height^2)
public:
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> cx{ "cx" };
    GeneratorInput<float> cy{ "cy" };
    GeneratorInput<Halide::Func> lut{ "lut", Halide::Float(32), 1 };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{ 0, width }, { 0, height }});

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

using LensDistortionCorrectionLUT2D = LensDistortionCorrectionLUT<2>;
using LensDistortionCorrectionLUT3D = LensDistortionCorrectionLUT<3>;

template<int D>
class LensDistortionCorrectionModel : public BuildingBlock<LensDistortionCorrectionModel<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
    // Output fx, fy is scaled by output_scale
public:
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> k1{ "k1" };
    GeneratorInput<float> k2{ "k2" };
    GeneratorInput<float> k3{ "k3" };
    GeneratorInput<float> p1{ "p1" };
    GeneratorInput<float> p2{ "p2" };
    GeneratorInput<float> fx{ "fx" };
    GeneratorInput<float> fy{ "fy" };
    GeneratorInput<float> cx{ "cx" };
    GeneratorInput<float> cy{ "cy" };
    GeneratorInput<float> output_scale{ "output_scale" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::constant_exterior(input, 0, {{ 0, width }, { 0, height }});

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

using LensDistortionCorrectionModel2D = LensDistortionCorrectionModel<2>;
using LensDistortionCorrectionModel3D = LensDistortionCorrectionModel<3>;

template<int D>
class ResizeNearest : public BuildingBlock<ResizeNearest<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
public:
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> scale{ "scale" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::repeat_edge(input, {{ 0, width }, { 0, height }});
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

using ResizeNearest2D = ResizeNearest<2>;
using ResizeNearest3D = ResizeNearest<3>;

template<int D>
class ResizeAreaAverage : public BuildingBlock<ResizeAreaAverage<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
public:
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> scale{ "scale" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::repeat_edge(input, {{ 0, width }, { 0, height }});

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

using ResizeAreaAverage2D = ResizeAreaAverage<2>;
using ResizeAreaAverage3D = ResizeAreaAverage<3>;

template<int D>
class ResizeBilinear : public BuildingBlock<ResizeBilinear<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
public:
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<float> scale{ "scale" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_wrapper = Halide::BoundaryConditions::repeat_edge(input, {{ 0, width }, { 0, height }});

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

using ResizeBilinear2D = ResizeBilinear<2>;
using ResizeBilinear3D = ResizeBilinear<3>;

template<typename T>
class BayerDownscale : public BuildingBlock<BayerDownscale<T>> {
    static_assert(std::is_arithmetic<T>::value, "T is not arithmetic");
public:
    GeneratorInput<int32_t> downscale_factor{ "downscale_factor" };
    GeneratorInput<Halide::Func> input{ "input", Halide::type_of<T>(), 2 };
    GeneratorOutput<Halide::Func> output{ "output", Halide::type_of<T>(), 2 };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        output(x, y) = input(x / 2 * 2 * downscale_factor + x % 2, y / 2 * 2 * downscale_factor + y % 2);
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

using BayerDownscaleInt16 = BayerDownscale<int16_t>;
using BayerDownscaleUint16 = BayerDownscale<uint16_t>;
using BayerDownscaleFloat = BayerDownscale<float>;

template<int D>
class Crop : public BuildingBlock<Crop<D>> {
    static_assert(D == 2 || D == 3, "D must be 2 or 3.");
public:
    GeneratorInput<int32_t> top{ "top" };
    GeneratorInput<int32_t> left{ "left" };
    GeneratorInput<int32_t> width{ "width" };
    GeneratorInput<int32_t> height{ "height" };
    GeneratorInput<Halide::Func> input{ "input", Halide::Float(32), D };
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), D };

    void generate() {
        Halide::Var x;
        Halide::Var y;

        Halide::Func input_shift;

        input_shift(x, y, Halide::_) = input(x + left, y + top, Halide::_);

        Halide::Func output_tmp = Halide::BoundaryConditions::constant_exterior(input_shift, 0, {{ 0, width }, { 0, height }});

        output(Halide::_) = output_tmp(Halide::_);
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

using Crop2D = Crop<2>;
using Crop3D = Crop<3>;

class MatrixDefinition : public BuildingBlock<MatrixDefinition> {
public:
    // Note: Now use GeneratorParam to avoid creating too many input ports
    // TODO: Use GeneratorInput to set on runtime
    GeneratorParam<float> matrix_value_00{ "matrix_value_00", 1.f};
    GeneratorParam<float> matrix_value_10{ "matrix_value_10", 0.f};
    GeneratorParam<float> matrix_value_20{ "matrix_value_20", 0.f};
    GeneratorParam<float> matrix_value_01{ "matrix_value_01", 0.f};
    GeneratorParam<float> matrix_value_11{ "matrix_value_11", 1.f};
    GeneratorParam<float> matrix_value_21{ "matrix_value_21", 0.f};
    GeneratorParam<float> matrix_value_02{ "matrix_value_02", 0.f};
    GeneratorParam<float> matrix_value_12{ "matrix_value_12", 0.f};
    GeneratorParam<float> matrix_value_22{ "matrix_value_22", 1.f};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
        Buffer<float> matrix(3, 3);

        matrix(0, 0) = matrix_value_00;
        matrix(1, 0) = matrix_value_10;
        matrix(2, 0) = matrix_value_20;
        matrix(0, 1) = matrix_value_01;
        matrix(1, 1) = matrix_value_11;
        matrix(2, 1) = matrix_value_21;
        matrix(0, 2) = matrix_value_02;
        matrix(1, 2) = matrix_value_12;
        matrix(2, 2) = matrix_value_22;

        output(Halide::_) = matrix(Halide::_);
    }

    void schedule() {
    }
};

class Table5x5Definition : public BuildingBlock<Table5x5Definition> {
public:
    // Note: Now use GeneratorParam to avoid creating too many input ports
    // TODO: Use GeneratorInput to set on runtime
    GeneratorParam<float> value_00{ "value_00", 0.f};
    GeneratorParam<float> value_10{ "value_10", 0.f};
    GeneratorParam<float> value_20{ "value_20", 0.f};
    GeneratorParam<float> value_30{ "value_30", 0.f};
    GeneratorParam<float> value_40{ "value_40", 0.f};
    GeneratorParam<float> value_01{ "value_01", 0.f};
    GeneratorParam<float> value_11{ "value_11", 0.f};
    GeneratorParam<float> value_21{ "value_21", 0.f};
    GeneratorParam<float> value_31{ "value_31", 0.f};
    GeneratorParam<float> value_41{ "value_41", 0.f};
    GeneratorParam<float> value_02{ "value_02", 0.f};
    GeneratorParam<float> value_12{ "value_12", 0.f};
    GeneratorParam<float> value_22{ "value_22", 0.f};
    GeneratorParam<float> value_32{ "value_32", 0.f};
    GeneratorParam<float> value_42{ "value_42", 0.f};
    GeneratorParam<float> value_03{ "value_03", 0.f};
    GeneratorParam<float> value_13{ "value_13", 0.f};
    GeneratorParam<float> value_23{ "value_23", 0.f};
    GeneratorParam<float> value_33{ "value_33", 0.f};
    GeneratorParam<float> value_43{ "value_43", 0.f};
    GeneratorParam<float> value_04{ "value_04", 0.f};
    GeneratorParam<float> value_14{ "value_14", 0.f};
    GeneratorParam<float> value_24{ "value_24", 0.f};
    GeneratorParam<float> value_34{ "value_34", 0.f};
    GeneratorParam<float> value_44{ "value_44", 0.f};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 2 };

    void generate() {
        Halide::Buffer<float> table(5, 5);

        table(0, 0) = value_00;
        table(1, 0) = value_10;
        table(2, 0) = value_20;
        table(3, 0) = value_30;
        table(4, 0) = value_40;
        table(0, 1) = value_01;
        table(1, 1) = value_11;
        table(2, 1) = value_21;
        table(3, 1) = value_31;
        table(4, 1) = value_41;
        table(0, 2) = value_02;
        table(1, 2) = value_12;
        table(2, 2) = value_22;
        table(3, 2) = value_32;
        table(4, 2) = value_42;
        table(0, 3) = value_03;
        table(1, 3) = value_13;
        table(2, 3) = value_23;
        table(3, 3) = value_33;
        table(4, 3) = value_43;
        table(0, 4) = value_04;
        table(1, 4) = value_14;
        table(2, 4) = value_24;
        table(3, 4) = value_34;
        table(4, 4) = value_44;

        output(Halide::_) = table(Halide::_);
    }

    void schedule() {
    }
};

class LUTDefinition : public BuildingBlock<LUTDefinition> {
public:
    // Note: Now use GeneratorParam to avoid creating too many input ports
    // TODO: Use GeneratorInput to set on runtime
    GeneratorParam<float> lut_value_0{ "lut_value_0", 0.f};
    GeneratorParam<float> lut_value_1{ "lut_value_1", 0.f};
    GeneratorParam<float> lut_value_2{ "lut_value_2", 0.f};
    GeneratorParam<float> lut_value_3{ "lut_value_3", 0.f};
    GeneratorParam<float> lut_value_4{ "lut_value_4", 0.f};
    GeneratorParam<float> lut_value_5{ "lut_value_5", 0.f};
    GeneratorParam<float> lut_value_6{ "lut_value_6", 0.f};
    GeneratorParam<float> lut_value_7{ "lut_value_7", 0.f};
    GeneratorParam<float> lut_value_8{ "lut_value_8", 0.f};
    GeneratorParam<float> lut_value_9{ "lut_value_9", 0.f};
    GeneratorParam<float> lut_value_10{ "lut_value_10", 0.f};
    GeneratorParam<float> lut_value_11{ "lut_value_11", 0.f};
    GeneratorParam<float> lut_value_12{ "lut_value_12", 0.f};
    GeneratorParam<float> lut_value_13{ "lut_value_13", 0.f};
    GeneratorParam<float> lut_value_14{ "lut_value_14", 0.f};
    GeneratorParam<float> lut_value_15{ "lut_value_15", 0.f};
    GeneratorParam<float> lut_value_16{ "lut_value_16", 0.f};
    GeneratorParam<float> lut_value_17{ "lut_value_17", 0.f};
    GeneratorParam<float> lut_value_18{ "lut_value_18", 0.f};
    GeneratorParam<float> lut_value_19{ "lut_value_19", 0.f};
    GeneratorParam<float> lut_value_20{ "lut_value_20", 0.f};
    GeneratorParam<float> lut_value_21{ "lut_value_21", 0.f};
    GeneratorParam<float> lut_value_22{ "lut_value_22", 0.f};
    GeneratorParam<float> lut_value_23{ "lut_value_23", 0.f};
    GeneratorParam<float> lut_value_24{ "lut_value_24", 0.f};
    GeneratorParam<float> lut_value_25{ "lut_value_25", 0.f};
    GeneratorParam<float> lut_value_26{ "lut_value_26", 0.f};
    GeneratorParam<float> lut_value_27{ "lut_value_27", 0.f};
    GeneratorParam<float> lut_value_28{ "lut_value_28", 0.f};
    GeneratorParam<float> lut_value_29{ "lut_value_29", 0.f};
    GeneratorParam<float> lut_value_30{ "lut_value_30", 0.f};
    GeneratorParam<float> lut_value_31{ "lut_value_31", 0.f};
    GeneratorParam<float> lut_value_32{ "lut_value_32", 0.f};
    GeneratorParam<float> lut_value_33{ "lut_value_33", 0.f};
    GeneratorParam<float> lut_value_34{ "lut_value_34", 0.f};
    GeneratorParam<float> lut_value_35{ "lut_value_35", 0.f};
    GeneratorParam<float> lut_value_36{ "lut_value_36", 0.f};
    GeneratorParam<float> lut_value_37{ "lut_value_37", 0.f};
    GeneratorParam<float> lut_value_38{ "lut_value_38", 0.f};
    GeneratorParam<float> lut_value_39{ "lut_value_39", 0.f};
    GeneratorParam<float> lut_value_40{ "lut_value_40", 0.f};
    GeneratorParam<float> lut_value_41{ "lut_value_41", 0.f};
    GeneratorParam<float> lut_value_42{ "lut_value_42", 0.f};
    GeneratorParam<float> lut_value_43{ "lut_value_43", 0.f};
    GeneratorParam<float> lut_value_44{ "lut_value_44", 0.f};
    GeneratorParam<float> lut_value_45{ "lut_value_45", 0.f};
    GeneratorParam<float> lut_value_46{ "lut_value_46", 0.f};
    GeneratorParam<float> lut_value_47{ "lut_value_47", 0.f};
    GeneratorParam<float> lut_value_48{ "lut_value_48", 0.f};
    GeneratorParam<float> lut_value_49{ "lut_value_49", 0.f};
    GeneratorParam<float> lut_value_50{ "lut_value_50", 0.f};
    GeneratorParam<float> lut_value_51{ "lut_value_51", 0.f};
    GeneratorParam<float> lut_value_52{ "lut_value_52", 0.f};
    GeneratorParam<float> lut_value_53{ "lut_value_53", 0.f};
    GeneratorParam<float> lut_value_54{ "lut_value_54", 0.f};
    GeneratorParam<float> lut_value_55{ "lut_value_55", 0.f};
    GeneratorParam<float> lut_value_56{ "lut_value_56", 0.f};
    GeneratorParam<float> lut_value_57{ "lut_value_57", 0.f};
    GeneratorParam<float> lut_value_58{ "lut_value_58", 0.f};
    GeneratorParam<float> lut_value_59{ "lut_value_59", 0.f};
    GeneratorParam<float> lut_value_60{ "lut_value_60", 0.f};
    GeneratorParam<float> lut_value_61{ "lut_value_61", 0.f};
    GeneratorParam<float> lut_value_62{ "lut_value_62", 0.f};
    GeneratorParam<float> lut_value_63{ "lut_value_63", 0.f};
    GeneratorParam<float> lut_value_64{ "lut_value_64", 0.f};
    GeneratorParam<float> lut_value_65{ "lut_value_65", 0.f};
    GeneratorParam<float> lut_value_66{ "lut_value_66", 0.f};
    GeneratorParam<float> lut_value_67{ "lut_value_67", 0.f};
    GeneratorParam<float> lut_value_68{ "lut_value_68", 0.f};
    GeneratorParam<float> lut_value_69{ "lut_value_69", 0.f};
    GeneratorParam<float> lut_value_70{ "lut_value_70", 0.f};
    GeneratorParam<float> lut_value_71{ "lut_value_71", 0.f};
    GeneratorParam<float> lut_value_72{ "lut_value_72", 0.f};
    GeneratorParam<float> lut_value_73{ "lut_value_73", 0.f};
    GeneratorParam<float> lut_value_74{ "lut_value_74", 0.f};
    GeneratorParam<float> lut_value_75{ "lut_value_75", 0.f};
    GeneratorParam<float> lut_value_76{ "lut_value_76", 0.f};
    GeneratorParam<float> lut_value_77{ "lut_value_77", 0.f};
    GeneratorParam<float> lut_value_78{ "lut_value_78", 0.f};
    GeneratorParam<float> lut_value_79{ "lut_value_79", 0.f};
    GeneratorParam<float> lut_value_80{ "lut_value_80", 0.f};
    GeneratorParam<float> lut_value_81{ "lut_value_81", 0.f};
    GeneratorParam<float> lut_value_82{ "lut_value_82", 0.f};
    GeneratorParam<float> lut_value_83{ "lut_value_83", 0.f};
    GeneratorParam<float> lut_value_84{ "lut_value_84", 0.f};
    GeneratorParam<float> lut_value_85{ "lut_value_85", 0.f};
    GeneratorParam<float> lut_value_86{ "lut_value_86", 0.f};
    GeneratorParam<float> lut_value_87{ "lut_value_87", 0.f};
    GeneratorParam<float> lut_value_88{ "lut_value_88", 0.f};
    GeneratorParam<float> lut_value_89{ "lut_value_89", 0.f};
    GeneratorParam<float> lut_value_90{ "lut_value_90", 0.f};
    GeneratorParam<float> lut_value_91{ "lut_value_91", 0.f};
    GeneratorParam<float> lut_value_92{ "lut_value_92", 0.f};
    GeneratorParam<float> lut_value_93{ "lut_value_93", 0.f};
    GeneratorParam<float> lut_value_94{ "lut_value_94", 0.f};
    GeneratorParam<float> lut_value_95{ "lut_value_95", 0.f};
    GeneratorParam<float> lut_value_96{ "lut_value_96", 0.f};
    GeneratorParam<float> lut_value_97{ "lut_value_97", 0.f};
    GeneratorParam<float> lut_value_98{ "lut_value_98", 0.f};
    GeneratorParam<float> lut_value_99{ "lut_value_99", 0.f};
    GeneratorParam<float> lut_value_100{ "lut_value_100", 0.f};
    GeneratorParam<float> lut_value_101{ "lut_value_101", 0.f};
    GeneratorParam<float> lut_value_102{ "lut_value_102", 0.f};
    GeneratorParam<float> lut_value_103{ "lut_value_103", 0.f};
    GeneratorParam<float> lut_value_104{ "lut_value_104", 0.f};
    GeneratorParam<float> lut_value_105{ "lut_value_105", 0.f};
    GeneratorParam<float> lut_value_106{ "lut_value_106", 0.f};
    GeneratorParam<float> lut_value_107{ "lut_value_107", 0.f};
    GeneratorParam<float> lut_value_108{ "lut_value_108", 0.f};
    GeneratorParam<float> lut_value_109{ "lut_value_109", 0.f};
    GeneratorParam<float> lut_value_110{ "lut_value_110", 0.f};
    GeneratorParam<float> lut_value_111{ "lut_value_111", 0.f};
    GeneratorParam<float> lut_value_112{ "lut_value_112", 0.f};
    GeneratorParam<float> lut_value_113{ "lut_value_113", 0.f};
    GeneratorParam<float> lut_value_114{ "lut_value_114", 0.f};
    GeneratorParam<float> lut_value_115{ "lut_value_115", 0.f};
    GeneratorParam<float> lut_value_116{ "lut_value_116", 0.f};
    GeneratorParam<float> lut_value_117{ "lut_value_117", 0.f};
    GeneratorParam<float> lut_value_118{ "lut_value_118", 0.f};
    GeneratorParam<float> lut_value_119{ "lut_value_119", 0.f};
    GeneratorParam<float> lut_value_120{ "lut_value_120", 0.f};
    GeneratorParam<float> lut_value_121{ "lut_value_121", 0.f};
    GeneratorParam<float> lut_value_122{ "lut_value_122", 0.f};
    GeneratorParam<float> lut_value_123{ "lut_value_123", 0.f};
    GeneratorParam<float> lut_value_124{ "lut_value_124", 0.f};
    GeneratorParam<float> lut_value_125{ "lut_value_125", 0.f};
    GeneratorParam<float> lut_value_126{ "lut_value_126", 0.f};
    GeneratorParam<float> lut_value_127{ "lut_value_127", 0.f};
    GeneratorParam<float> lut_value_128{ "lut_value_128", 0.f};
    GeneratorParam<float> lut_value_129{ "lut_value_129", 0.f};
    GeneratorParam<float> lut_value_130{ "lut_value_130", 0.f};
    GeneratorParam<float> lut_value_131{ "lut_value_131", 0.f};
    GeneratorParam<float> lut_value_132{ "lut_value_132", 0.f};
    GeneratorParam<float> lut_value_133{ "lut_value_133", 0.f};
    GeneratorParam<float> lut_value_134{ "lut_value_134", 0.f};
    GeneratorParam<float> lut_value_135{ "lut_value_135", 0.f};
    GeneratorParam<float> lut_value_136{ "lut_value_136", 0.f};
    GeneratorParam<float> lut_value_137{ "lut_value_137", 0.f};
    GeneratorParam<float> lut_value_138{ "lut_value_138", 0.f};
    GeneratorParam<float> lut_value_139{ "lut_value_139", 0.f};
    GeneratorParam<float> lut_value_140{ "lut_value_140", 0.f};
    GeneratorParam<float> lut_value_141{ "lut_value_141", 0.f};
    GeneratorParam<float> lut_value_142{ "lut_value_142", 0.f};
    GeneratorParam<float> lut_value_143{ "lut_value_143", 0.f};
    GeneratorParam<float> lut_value_144{ "lut_value_144", 0.f};
    GeneratorParam<float> lut_value_145{ "lut_value_145", 0.f};
    GeneratorParam<float> lut_value_146{ "lut_value_146", 0.f};
    GeneratorParam<float> lut_value_147{ "lut_value_147", 0.f};
    GeneratorParam<float> lut_value_148{ "lut_value_148", 0.f};
    GeneratorParam<float> lut_value_149{ "lut_value_149", 0.f};
    GeneratorParam<float> lut_value_150{ "lut_value_150", 0.f};
    GeneratorParam<float> lut_value_151{ "lut_value_151", 0.f};
    GeneratorParam<float> lut_value_152{ "lut_value_152", 0.f};
    GeneratorParam<float> lut_value_153{ "lut_value_153", 0.f};
    GeneratorParam<float> lut_value_154{ "lut_value_154", 0.f};
    GeneratorParam<float> lut_value_155{ "lut_value_155", 0.f};
    GeneratorParam<float> lut_value_156{ "lut_value_156", 0.f};
    GeneratorParam<float> lut_value_157{ "lut_value_157", 0.f};
    GeneratorParam<float> lut_value_158{ "lut_value_158", 0.f};
    GeneratorParam<float> lut_value_159{ "lut_value_159", 0.f};
    GeneratorParam<float> lut_value_160{ "lut_value_160", 0.f};
    GeneratorParam<float> lut_value_161{ "lut_value_161", 0.f};
    GeneratorParam<float> lut_value_162{ "lut_value_162", 0.f};
    GeneratorParam<float> lut_value_163{ "lut_value_163", 0.f};
    GeneratorParam<float> lut_value_164{ "lut_value_164", 0.f};
    GeneratorParam<float> lut_value_165{ "lut_value_165", 0.f};
    GeneratorParam<float> lut_value_166{ "lut_value_166", 0.f};
    GeneratorParam<float> lut_value_167{ "lut_value_167", 0.f};
    GeneratorParam<float> lut_value_168{ "lut_value_168", 0.f};
    GeneratorParam<float> lut_value_169{ "lut_value_169", 0.f};
    GeneratorParam<float> lut_value_170{ "lut_value_170", 0.f};
    GeneratorParam<float> lut_value_171{ "lut_value_171", 0.f};
    GeneratorParam<float> lut_value_172{ "lut_value_172", 0.f};
    GeneratorParam<float> lut_value_173{ "lut_value_173", 0.f};
    GeneratorParam<float> lut_value_174{ "lut_value_174", 0.f};
    GeneratorParam<float> lut_value_175{ "lut_value_175", 0.f};
    GeneratorParam<float> lut_value_176{ "lut_value_176", 0.f};
    GeneratorParam<float> lut_value_177{ "lut_value_177", 0.f};
    GeneratorParam<float> lut_value_178{ "lut_value_178", 0.f};
    GeneratorParam<float> lut_value_179{ "lut_value_179", 0.f};
    GeneratorParam<float> lut_value_180{ "lut_value_180", 0.f};
    GeneratorParam<float> lut_value_181{ "lut_value_181", 0.f};
    GeneratorParam<float> lut_value_182{ "lut_value_182", 0.f};
    GeneratorParam<float> lut_value_183{ "lut_value_183", 0.f};
    GeneratorParam<float> lut_value_184{ "lut_value_184", 0.f};
    GeneratorParam<float> lut_value_185{ "lut_value_185", 0.f};
    GeneratorParam<float> lut_value_186{ "lut_value_186", 0.f};
    GeneratorParam<float> lut_value_187{ "lut_value_187", 0.f};
    GeneratorParam<float> lut_value_188{ "lut_value_188", 0.f};
    GeneratorParam<float> lut_value_189{ "lut_value_189", 0.f};
    GeneratorParam<float> lut_value_190{ "lut_value_190", 0.f};
    GeneratorParam<float> lut_value_191{ "lut_value_191", 0.f};
    GeneratorParam<float> lut_value_192{ "lut_value_192", 0.f};
    GeneratorParam<float> lut_value_193{ "lut_value_193", 0.f};
    GeneratorParam<float> lut_value_194{ "lut_value_194", 0.f};
    GeneratorParam<float> lut_value_195{ "lut_value_195", 0.f};
    GeneratorParam<float> lut_value_196{ "lut_value_196", 0.f};
    GeneratorParam<float> lut_value_197{ "lut_value_197", 0.f};
    GeneratorParam<float> lut_value_198{ "lut_value_198", 0.f};
    GeneratorParam<float> lut_value_199{ "lut_value_199", 0.f};
    GeneratorParam<float> lut_value_200{ "lut_value_200", 0.f};
    GeneratorParam<float> lut_value_201{ "lut_value_201", 0.f};
    GeneratorParam<float> lut_value_202{ "lut_value_202", 0.f};
    GeneratorParam<float> lut_value_203{ "lut_value_203", 0.f};
    GeneratorParam<float> lut_value_204{ "lut_value_204", 0.f};
    GeneratorParam<float> lut_value_205{ "lut_value_205", 0.f};
    GeneratorParam<float> lut_value_206{ "lut_value_206", 0.f};
    GeneratorParam<float> lut_value_207{ "lut_value_207", 0.f};
    GeneratorParam<float> lut_value_208{ "lut_value_208", 0.f};
    GeneratorParam<float> lut_value_209{ "lut_value_209", 0.f};
    GeneratorParam<float> lut_value_210{ "lut_value_210", 0.f};
    GeneratorParam<float> lut_value_211{ "lut_value_211", 0.f};
    GeneratorParam<float> lut_value_212{ "lut_value_212", 0.f};
    GeneratorParam<float> lut_value_213{ "lut_value_213", 0.f};
    GeneratorParam<float> lut_value_214{ "lut_value_214", 0.f};
    GeneratorParam<float> lut_value_215{ "lut_value_215", 0.f};
    GeneratorParam<float> lut_value_216{ "lut_value_216", 0.f};
    GeneratorParam<float> lut_value_217{ "lut_value_217", 0.f};
    GeneratorParam<float> lut_value_218{ "lut_value_218", 0.f};
    GeneratorParam<float> lut_value_219{ "lut_value_219", 0.f};
    GeneratorParam<float> lut_value_220{ "lut_value_220", 0.f};
    GeneratorParam<float> lut_value_221{ "lut_value_221", 0.f};
    GeneratorParam<float> lut_value_222{ "lut_value_222", 0.f};
    GeneratorParam<float> lut_value_223{ "lut_value_223", 0.f};
    GeneratorParam<float> lut_value_224{ "lut_value_224", 0.f};
    GeneratorParam<float> lut_value_225{ "lut_value_225", 0.f};
    GeneratorParam<float> lut_value_226{ "lut_value_226", 0.f};
    GeneratorParam<float> lut_value_227{ "lut_value_227", 0.f};
    GeneratorParam<float> lut_value_228{ "lut_value_228", 0.f};
    GeneratorParam<float> lut_value_229{ "lut_value_229", 0.f};
    GeneratorParam<float> lut_value_230{ "lut_value_230", 0.f};
    GeneratorParam<float> lut_value_231{ "lut_value_231", 0.f};
    GeneratorParam<float> lut_value_232{ "lut_value_232", 0.f};
    GeneratorParam<float> lut_value_233{ "lut_value_233", 0.f};
    GeneratorParam<float> lut_value_234{ "lut_value_234", 0.f};
    GeneratorParam<float> lut_value_235{ "lut_value_235", 0.f};
    GeneratorParam<float> lut_value_236{ "lut_value_236", 0.f};
    GeneratorParam<float> lut_value_237{ "lut_value_237", 0.f};
    GeneratorParam<float> lut_value_238{ "lut_value_238", 0.f};
    GeneratorParam<float> lut_value_239{ "lut_value_239", 0.f};
    GeneratorParam<float> lut_value_240{ "lut_value_240", 0.f};
    GeneratorParam<float> lut_value_241{ "lut_value_241", 0.f};
    GeneratorParam<float> lut_value_242{ "lut_value_242", 0.f};
    GeneratorParam<float> lut_value_243{ "lut_value_243", 0.f};
    GeneratorParam<float> lut_value_244{ "lut_value_244", 0.f};
    GeneratorParam<float> lut_value_245{ "lut_value_245", 0.f};
    GeneratorParam<float> lut_value_246{ "lut_value_246", 0.f};
    GeneratorParam<float> lut_value_247{ "lut_value_247", 0.f};
    GeneratorParam<float> lut_value_248{ "lut_value_248", 0.f};
    GeneratorParam<float> lut_value_249{ "lut_value_249", 0.f};
    GeneratorParam<float> lut_value_250{ "lut_value_250", 0.f};
    GeneratorParam<float> lut_value_251{ "lut_value_251", 0.f};
    GeneratorParam<float> lut_value_252{ "lut_value_252", 0.f};
    GeneratorParam<float> lut_value_253{ "lut_value_253", 0.f};
    GeneratorParam<float> lut_value_254{ "lut_value_254", 0.f};
    GeneratorParam<float> lut_value_255{ "lut_value_255", 0.f};
    GeneratorParam<float> lut_value_256{ "lut_value_256", 0.f};
    GeneratorOutput<Halide::Func> output{ "output", Halide::Float(32), 1 };

    void generate() {
        Buffer<float> lut(257);

        lut(0) = lut_value_0;
        lut(1) = lut_value_1;
        lut(2) = lut_value_2;
        lut(3) = lut_value_3;
        lut(4) = lut_value_4;
        lut(5) = lut_value_5;
        lut(6) = lut_value_6;
        lut(7) = lut_value_7;
        lut(8) = lut_value_8;
        lut(9) = lut_value_9;
        lut(10) = lut_value_10;
        lut(11) = lut_value_11;
        lut(12) = lut_value_12;
        lut(13) = lut_value_13;
        lut(14) = lut_value_14;
        lut(15) = lut_value_15;
        lut(16) = lut_value_16;
        lut(17) = lut_value_17;
        lut(18) = lut_value_18;
        lut(19) = lut_value_19;
        lut(20) = lut_value_20;
        lut(21) = lut_value_21;
        lut(22) = lut_value_22;
        lut(23) = lut_value_23;
        lut(24) = lut_value_24;
        lut(25) = lut_value_25;
        lut(26) = lut_value_26;
        lut(27) = lut_value_27;
        lut(28) = lut_value_28;
        lut(29) = lut_value_29;
        lut(30) = lut_value_30;
        lut(31) = lut_value_31;
        lut(32) = lut_value_32;
        lut(33) = lut_value_33;
        lut(34) = lut_value_34;
        lut(35) = lut_value_35;
        lut(36) = lut_value_36;
        lut(37) = lut_value_37;
        lut(38) = lut_value_38;
        lut(39) = lut_value_39;
        lut(40) = lut_value_40;
        lut(41) = lut_value_41;
        lut(42) = lut_value_42;
        lut(43) = lut_value_43;
        lut(44) = lut_value_44;
        lut(45) = lut_value_45;
        lut(46) = lut_value_46;
        lut(47) = lut_value_47;
        lut(48) = lut_value_48;
        lut(49) = lut_value_49;
        lut(50) = lut_value_50;
        lut(51) = lut_value_51;
        lut(52) = lut_value_52;
        lut(53) = lut_value_53;
        lut(54) = lut_value_54;
        lut(55) = lut_value_55;
        lut(56) = lut_value_56;
        lut(57) = lut_value_57;
        lut(58) = lut_value_58;
        lut(59) = lut_value_59;
        lut(60) = lut_value_60;
        lut(61) = lut_value_61;
        lut(62) = lut_value_62;
        lut(63) = lut_value_63;
        lut(64) = lut_value_64;
        lut(65) = lut_value_65;
        lut(66) = lut_value_66;
        lut(67) = lut_value_67;
        lut(68) = lut_value_68;
        lut(69) = lut_value_69;
        lut(70) = lut_value_70;
        lut(71) = lut_value_71;
        lut(72) = lut_value_72;
        lut(73) = lut_value_73;
        lut(74) = lut_value_74;
        lut(75) = lut_value_75;
        lut(76) = lut_value_76;
        lut(77) = lut_value_77;
        lut(78) = lut_value_78;
        lut(79) = lut_value_79;
        lut(80) = lut_value_80;
        lut(81) = lut_value_81;
        lut(82) = lut_value_82;
        lut(83) = lut_value_83;
        lut(84) = lut_value_84;
        lut(85) = lut_value_85;
        lut(86) = lut_value_86;
        lut(87) = lut_value_87;
        lut(88) = lut_value_88;
        lut(89) = lut_value_89;
        lut(90) = lut_value_90;
        lut(91) = lut_value_91;
        lut(92) = lut_value_92;
        lut(93) = lut_value_93;
        lut(94) = lut_value_94;
        lut(95) = lut_value_95;
        lut(96) = lut_value_96;
        lut(97) = lut_value_97;
        lut(98) = lut_value_98;
        lut(99) = lut_value_99;
        lut(100) = lut_value_100;
        lut(101) = lut_value_101;
        lut(102) = lut_value_102;
        lut(103) = lut_value_103;
        lut(104) = lut_value_104;
        lut(105) = lut_value_105;
        lut(106) = lut_value_106;
        lut(107) = lut_value_107;
        lut(108) = lut_value_108;
        lut(109) = lut_value_109;
        lut(110) = lut_value_110;
        lut(111) = lut_value_111;
        lut(112) = lut_value_112;
        lut(113) = lut_value_113;
        lut(114) = lut_value_114;
        lut(115) = lut_value_115;
        lut(116) = lut_value_116;
        lut(117) = lut_value_117;
        lut(118) = lut_value_118;
        lut(119) = lut_value_119;
        lut(120) = lut_value_120;
        lut(121) = lut_value_121;
        lut(122) = lut_value_122;
        lut(123) = lut_value_123;
        lut(124) = lut_value_124;
        lut(125) = lut_value_125;
        lut(126) = lut_value_126;
        lut(127) = lut_value_127;
        lut(128) = lut_value_128;
        lut(129) = lut_value_129;
        lut(130) = lut_value_130;
        lut(131) = lut_value_131;
        lut(132) = lut_value_132;
        lut(133) = lut_value_133;
        lut(134) = lut_value_134;
        lut(135) = lut_value_135;
        lut(136) = lut_value_136;
        lut(137) = lut_value_137;
        lut(138) = lut_value_138;
        lut(139) = lut_value_139;
        lut(140) = lut_value_140;
        lut(141) = lut_value_141;
        lut(142) = lut_value_142;
        lut(143) = lut_value_143;
        lut(144) = lut_value_144;
        lut(145) = lut_value_145;
        lut(146) = lut_value_146;
        lut(147) = lut_value_147;
        lut(148) = lut_value_148;
        lut(149) = lut_value_149;
        lut(150) = lut_value_150;
        lut(151) = lut_value_151;
        lut(152) = lut_value_152;
        lut(153) = lut_value_153;
        lut(154) = lut_value_154;
        lut(155) = lut_value_155;
        lut(156) = lut_value_156;
        lut(157) = lut_value_157;
        lut(158) = lut_value_158;
        lut(159) = lut_value_159;
        lut(160) = lut_value_160;
        lut(161) = lut_value_161;
        lut(162) = lut_value_162;
        lut(163) = lut_value_163;
        lut(164) = lut_value_164;
        lut(165) = lut_value_165;
        lut(166) = lut_value_166;
        lut(167) = lut_value_167;
        lut(168) = lut_value_168;
        lut(169) = lut_value_169;
        lut(170) = lut_value_170;
        lut(171) = lut_value_171;
        lut(172) = lut_value_172;
        lut(173) = lut_value_173;
        lut(174) = lut_value_174;
        lut(175) = lut_value_175;
        lut(176) = lut_value_176;
        lut(177) = lut_value_177;
        lut(178) = lut_value_178;
        lut(179) = lut_value_179;
        lut(180) = lut_value_180;
        lut(181) = lut_value_181;
        lut(182) = lut_value_182;
        lut(183) = lut_value_183;
        lut(184) = lut_value_184;
        lut(185) = lut_value_185;
        lut(186) = lut_value_186;
        lut(187) = lut_value_187;
        lut(188) = lut_value_188;
        lut(189) = lut_value_189;
        lut(190) = lut_value_190;
        lut(191) = lut_value_191;
        lut(192) = lut_value_192;
        lut(193) = lut_value_193;
        lut(194) = lut_value_194;
        lut(195) = lut_value_195;
        lut(196) = lut_value_196;
        lut(197) = lut_value_197;
        lut(198) = lut_value_198;
        lut(199) = lut_value_199;
        lut(200) = lut_value_200;
        lut(201) = lut_value_201;
        lut(202) = lut_value_202;
        lut(203) = lut_value_203;
        lut(204) = lut_value_204;
        lut(205) = lut_value_205;
        lut(206) = lut_value_206;
        lut(207) = lut_value_207;
        lut(208) = lut_value_208;
        lut(209) = lut_value_209;
        lut(210) = lut_value_210;
        lut(211) = lut_value_211;
        lut(212) = lut_value_212;
        lut(213) = lut_value_213;
        lut(214) = lut_value_214;
        lut(215) = lut_value_215;
        lut(216) = lut_value_216;
        lut(217) = lut_value_217;
        lut(218) = lut_value_218;
        lut(219) = lut_value_219;
        lut(220) = lut_value_220;
        lut(221) = lut_value_221;
        lut(222) = lut_value_222;
        lut(223) = lut_value_223;
        lut(224) = lut_value_224;
        lut(225) = lut_value_225;
        lut(226) = lut_value_226;
        lut(227) = lut_value_227;
        lut(228) = lut_value_228;
        lut(229) = lut_value_229;
        lut(230) = lut_value_230;
        lut(231) = lut_value_231;
        lut(232) = lut_value_232;
        lut(233) = lut_value_233;
        lut(234) = lut_value_234;
        lut(235) = lut_value_235;
        lut(236) = lut_value_236;
        lut(237) = lut_value_237;
        lut(238) = lut_value_238;
        lut(239) = lut_value_239;
        lut(240) = lut_value_240;
        lut(241) = lut_value_241;
        lut(242) = lut_value_242;
        lut(243) = lut_value_243;
        lut(244) = lut_value_244;
        lut(245) = lut_value_245;
        lut(246) = lut_value_246;
        lut(247) = lut_value_247;
        lut(248) = lut_value_248;
        lut(249) = lut_value_249;
        lut(250) = lut_value_250;
        lut(251) = lut_value_251;
        lut(252) = lut_value_252;
        lut(253) = lut_value_253;
        lut(254) = lut_value_254;
        lut(255) = lut_value_255;
        lut(256) = lut_value_256;

        output(Halide::_) = lut(Halide::_);
    }

    void schedule() {
    }
};

} // isp
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Offset2D, isp_offset2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Offset3D, isp_offset3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerOffset, isp_bayer_offset);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerWhiteBalance, isp_bayer_white_balance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerDemosaicLinear, isp_bayer_demosaic_linear);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerDemosaicFilter, isp_bayer_demosaic_filter);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerDemosaicSimple, isp_bayer_demosaic_simple);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::GammaCorrection2D, isp_gamma_correction2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::GammaCorrection3D, isp_gamma_correction3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ColorMatrix, isp_color_matrix);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LensShadingCorrectionLUT, isp_lens_shading_correction_lut);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LensShadingCorrectionLinear, isp_lens_shading_correction_linear);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BilateralFilter2D, isp_bilateral_filter2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BilateralFilter3D, isp_bilateral_filter3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::CalcLuminance, isp_calc_luminance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Filter2D, isp_filter2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Filter3D, isp_filter3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LensDistortionCorrectionLUT2D, isp_lens_distortion_correction_lut2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LensDistortionCorrectionLUT3D, isp_lens_distortion_correction_lut3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LensDistortionCorrectionModel2D, isp_lens_distortion_correction_model2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LensDistortionCorrectionModel3D, isp_lens_distortion_correction_model3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ResizeNearest2D, isp_resize_nearest2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ResizeNearest3D, isp_resize_nearest3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ResizeAreaAverage2D, isp_resize_area_average2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ResizeAreaAverage3D, isp_resize_area_average3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ResizeBilinear2D, isp_resize_bilinear2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::ResizeBilinear3D, isp_resize_bilinear3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerDownscaleInt16, isp_bayer_downscale_int16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerDownscaleUint16, isp_bayer_downscale_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::BayerDownscaleFloat, isp_bayer_downscale_float);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Crop2D, isp_crop2d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Crop3D, isp_crop3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::MatrixDefinition, isp_matrix_definition);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::Table5x5Definition, isp_table5x5_definition);
ION_REGISTER_BUILDING_BLOCK(ion::bb::isp::LUTDefinition, isp_lut_definition);

#endif
