#ifndef ION_BB_FPGA_BB_H
#define ION_BB_FPGA_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace fpga {

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
        SimpleY
    };

    static const std::map<std::string, Method> enum_map;

    // Int version
    static Halide::Expr calc(Method method, Halide::Expr r, Halide::Expr g, Halide::Expr b) {
        internal_assert(r.type().is_int_or_uint());
        Halide::Type t;
        switch (method) {
        case Method::Max:
            return Halide::max(r, g, b);
        case Method::Average:
            t = r.type().with_bits(r.type().bits() + 2);
            return Halide::cast(r.type(), (Halide::cast(t, r) + g + b) / 3);
        case Method::SimpleY:
            t = r.type().with_bits(r.type().bits() + 4);
            return Halide::cast(r.type(), (Halide::cast(t, r) * 3 + Halide::cast(t, g) * 12 + b) / 16);
        default:
            internal_error << "Unknown Luminance method";
        }

        // Unreachable
        return Halide::Expr();
    }
};

const std::map<std::string, Luminance::Method> Luminance::enum_map{
    {"Max", Luminance::Method::Max},
    {"Average", Luminance::Method::Average},
    {"SimpleY", Luminance::Method::SimpleY}};

int32_t bit_width(uint64_t n) {
    int32_t bits;
    for (bits = 0; n >> bits; bits++)
        ;
    return bits;
}

Halide::Func bayer_offset(Halide::Func input, BayerMap::Pattern bayer_pattern, Halide::Expr offset_r, Halide::Expr offset_g, Halide::Expr offset_b, std::string name = "bayer_offset") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"};

    Halide::Expr offset = Halide::mux(BayerMap::get_color(bayer_pattern, x, y), {offset_r, offset_g, offset_b});
    output(x, y) = Halide::select(input(x, y) >= offset, input(x, y) - offset, 0);

    return output;
}

class BayerOffset : public BuildingBlock<BayerOffset> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "BayerOffset(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Offset values of bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<uint16_t> offset_r{"offset_r"};
    GeneratorInput<uint16_t> offset_g{"offset_g"};
    GeneratorInput<uint16_t> offset_b{"offset_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 2};

    void generate() {
        output = bayer_offset(input, static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), offset_r, offset_g, offset_b);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        output.bound(x, 0, width).bound(y, 0, height);
        if (get_target().has_fpga_feature()) {
            output.accelerate({input}, {}, Var::outermost());
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi, xii, yii;
            output.gpu_tile(x, y, xi, yi, 32, 32).tile(xi, yi, xii, yii, 2, 2).unroll(xii).unroll(yii);
        } else {
            Halide::Var xi, yi;
            output.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi).vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func bayer_white_balance(Halide::Func input, BayerMap::Pattern bayer_pattern, int32_t input_bits, Halide::Expr gain_r, Halide::Expr gain_g, Halide::Expr gain_b, std::string name = "bayer_white_balance") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"};

    Halide::Expr gain = Halide::mux(BayerMap::get_color(bayer_pattern, x, y), {gain_r, gain_g, gain_b});
    Halide::Expr mul = Halide::cast(Halide::UInt(32), input(x, y)) * gain;
    Halide::Expr out = (mul >> 12) + ((mul >> 11) & 1);  // round
    uint16_t max_value = (1 << input_bits) - 1;
    output(x, y) = Halide::select(out > max_value, max_value, Halide::cast(Halide::UInt(16), out));

    return output;
}

class BayerWhiteBalance : public BuildingBlock<BayerWhiteBalance> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "BayerWhiteBalance(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Gain values of bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> input_bits{"input_bits", 16, 1, 16};
    // 12bit fractional
    GeneratorInput<uint16_t> gain_r{"gain_r"};
    GeneratorInput<uint16_t> gain_g{"gain_g"};
    GeneratorInput<uint16_t> gain_b{"gain_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 2};

    void generate() {
        output = bayer_white_balance(input, static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), input_bits, gain_r, gain_g, gain_b);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        output.bound(x, 0, width).bound(y, 0, height);
        if (get_target().has_fpga_feature()) {
            output.accelerate({input}, {}, Var::outermost());
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi, xii, yii;
            output.gpu_tile(x, y, xi, yi, 32, 32).tile(xi, yi, xii, yii, 2, 2).unroll(xii).unroll(yii);
        } else {
            Halide::Var xi, yi;
            output.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi).vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func bayer_demosaic_simple(Halide::Func input, BayerMap::Pattern bayer_pattern, std::string name = "bayer_demosaic_simple") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    switch (bayer_pattern) {
    case BayerMap::Pattern::RGGB:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2, y * 2),
                Halide::cast(Halide::UInt(16), (Halide::cast(Halide::UInt(17), input(x * 2 + 1, y * 2)) + input(x * 2, y * 2 + 1)) >> 1),
                input(x * 2 + 1, y * 2 + 1)});
        break;
    case BayerMap::Pattern::BGGR:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2 + 1, y * 2 + 1),
                Halide::cast(Halide::UInt(16), (Halide::cast(Halide::UInt(17), input(x * 2 + 1, y * 2)) + input(x * 2, y * 2 + 1)) >> 1),
                input(x * 2, y * 2)});
        break;
    case BayerMap::Pattern::GRBG:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2 + 1, y * 2),
                Halide::cast(Halide::UInt(16), (Halide::cast(Halide::UInt(17), input(x * 2, y * 2)) + input(x * 2 + 1, y * 2 + 1)) >> 1),
                input(x * 2 + 1, y * 2)});
        break;
    case BayerMap::Pattern::GBRG:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2, y * 2 + 1),
                Halide::cast(Halide::UInt(16), (Halide::cast(Halide::UInt(17), input(x * 2, y * 2)) + input(x * 2 + 1, y * 2 + 1)) >> 1),
                input(x * 2 + 1, y * 2)});
        break;
    default:
        internal_error << "Unknown BayerMap pattern";
    }

    return output;
}

class BayerDemosaicSimple : public BuildingBlock<BayerDemosaicSimple> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "BayerDemosaicSimple(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Demosaic bayer image by simple algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map(x => x / 2).concat([3]) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 3};

    void generate() {
        output = bayer_demosaic_simple(input, static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var c = output.args()[0];
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        output.bound(x, 0, width / 2).bound(y, 0, height / 2).bound(c, 0, 3);
        if (get_target().has_fpga_feature()) {
            std::vector<Func> ip_in, ip_out;
            std::tie(ip_in, ip_out) = output.accelerate({input}, {}, Var::outermost());
            ip_out[0].bound(ip_out[0].args()[0], 0, 3).unroll(ip_out[0].args()[0]).hls_burst(3);
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::UInt(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func gamma_correction_3d(Halide::Func input, int32_t input_bits, int32_t output_bits, int32_t lut_bits, int32_t lut_index_bits, double gamma, int32_t width, int32_t unroll_num, std::string name = "gamma_correction_3d") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    Halide::Expr lut_sel = (c + x * 3 + y * width * 3) % unroll_num;

    int32_t lut_size = 1 << lut_index_bits;
    int32_t max_value = (1 << output_bits) - 1;
    if (input_bits == lut_index_bits) {
        std::vector<Halide::Expr> lut_expr_list;
        for (int32_t i = 0; i < unroll_num; i++) {
            Halide::Buffer<uint16_t> lut(lut_size);
            for (int32_t j = 0; j < lut_size; j++) {
                lut(j) = static_cast<uint16_t>(pow(j / static_cast<double>(lut_size - 1), gamma) * max_value + 0.5);
            }
            lut_expr_list.push_back(lut(input(c, x, y)));
        }
        output(c, x, y) = unroll_num == 1 ? lut_expr_list[0] : Halide::mux(lut_sel, lut_expr_list);
    } else {
        Halide::Expr is_odd_index = ((input(c, x, y) >> (input_bits - lut_index_bits)) & 1) == 1;
        Halide::Expr lut1_index = (input(c, x, y) >> (input_bits - lut_index_bits + 1)) & (lut_size / 2 - 1);
        Halide::Expr lut0_index = Halide::select(is_odd_index, lut1_index + 1, lut1_index);
        Halide::Expr lut0_index_limited = Halide::min(lut0_index, lut_size / 2 - 1);

        std::vector<Halide::Expr> lut0_expr_list;
        std::vector<Halide::Expr> lut1_expr_list;
        for (int32_t i = 0; i < unroll_num; i++) {
            Halide::Buffer<uint16_t> lut0(lut_size / 2);
            Halide::Buffer<uint16_t> lut1(lut_size / 2);
            for (int32_t j = 0; j < lut_size / 2; j++) {
                lut0(j) = static_cast<uint16_t>(pow((j * 2) / static_cast<double>(lut_size), gamma) * max_value + 0.5);
                lut1(j) = static_cast<uint16_t>(pow((j * 2 + 1) / static_cast<double>(lut_size), gamma) * max_value + 0.5);
            }
            lut0_expr_list.push_back(lut0(lut0_index_limited));
            lut1_expr_list.push_back(lut1(lut1_index));
        }

        Halide::Expr lut0_value = Halide::cast(Halide::UInt(lut_bits),
                                               Halide::select(lut0_index == lut_size / 2, max_value,
                                                              unroll_num == 1 ? lut0_expr_list[0] : Halide::mux(lut_sel, lut0_expr_list)));
        Halide::Expr lut1_value = Halide::cast(Halide::UInt(lut_bits), unroll_num == 1 ? lut1_expr_list[0] : Halide::mux(lut_sel, lut1_expr_list));
        Halide::Expr base_value = Halide::select(is_odd_index, lut1_value, lut0_value);
        Halide::Expr next_value = Halide::select(is_odd_index, lut0_value, lut1_value);
        Halide::Expr diff = next_value - base_value;
        Halide::Expr coef = input(c, x, y) & ((1 << (input_bits - lut_index_bits)) - 1);
        Halide::Expr mul = (Halide::cast(Halide::UInt(lut_bits + input_bits - lut_index_bits), diff) * coef) >> (lut_bits + input_bits - lut_index_bits - output_bits);
        output(c, x, y) = Halide::cast(Halide::UInt(16), (Halide::cast(Halide::UInt(output_bits), base_value) << (output_bits - lut_bits)) + mul);
    }

    return output;
}

class GammaCorrection3D : public BuildingBlock<GammaCorrection3D> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "GammaCorrection3D(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Gamma correction."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> input_bits{"input_bits", 16, 1, 16};
    GeneratorParam<int32_t> output_bits{"output_bits", 8, 1, 16};
    GeneratorParam<int32_t> lut_bits{"lut_bits", 8, 1, 16};
    GeneratorParam<int32_t> lut_index_bits{"lut_index_bits", 8, 1, 16};
    GeneratorParam<double> gamma{"gamma", 1 / 2.2};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 3};

    void generate() {
        int32_t gamma_unroll = get_target().has_fpga_feature() ? 3 : 1;
        output = gamma_correction_3d(input, input_bits, output_bits, lut_bits, lut_index_bits, gamma, width, gamma_unroll);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var c = output.args()[0];
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        output.bound(x, 0, width).bound(y, 0, height).bound(c, 0, 3);
        if (get_target().has_fpga_feature()) {
            std::vector<Func> ip_in, ip_out;
            std::tie(ip_in, ip_out) = output.accelerate({input}, {}, Var::outermost());
            ip_in[0].hls_burst(3);
            ip_out[0].unroll(ip_out[0].args()[0]).hls_burst(3);
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            output.gpu_tile(x, y, xi, yi, 32, 8);
        } else {
            output.vectorize(x, natural_vector_size(Halide::UInt(32)));
            output.parallel(y, 16);
        }
        output.compute_root();
#endif
    }
};

Halide::Func lens_shading_correction_linear(Halide::Func input, BayerMap::Pattern bayer_pattern, int32_t width, int32_t height, int32_t input_bits,
                                            Halide::Expr slope_r, Halide::Expr slope_g, Halide::Expr slope_b,
                                            Halide::Expr offset_r, Halide::Expr offset_g, Halide::Expr offset_b, std::string name = "lens_shading_correction_linear") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"};

    int32_t center_x = width / 2;                                 // max 15bit
    int32_t center_y = height / 2;                                // max 15bit
    uint32_t r2_max = center_x * center_x + center_y * center_y;  // max 31bit

    int32_t dividend_bits = std::max(bit_width(center_x) * 2, bit_width(center_y) * 2) + 1 + 16;  // max 47bit

    Halide::Expr r2 = Halide::cast(Halide::UInt(16), (Halide::cast(Halide::UInt(dividend_bits), (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) << 16) / Halide::Expr(r2_max));  // 16bit

    Halide::Expr color = BayerMap::get_color(bayer_pattern, x, y);
    Halide::Expr coef_mul = (Halide::cast(Halide::UInt(32), r2) * Halide::mux(color, {slope_r, slope_g, slope_b})) >> 16;  // 16bit
    Halide::Expr coef = Halide::cast(Halide::UInt(17), coef_mul) + Halide::mux(color, {offset_r, offset_g, offset_b});     // 17bit
    Halide::Expr coef_clamp = Halide::select(coef >= 65536, 65535, Halide::cast(Halide::UInt(16), coef));                  // 16bit

    Halide::Expr mul = Halide::cast(Halide::UInt(32), input(x, y)) * coef_clamp;
    Halide::Expr out = (mul >> 12) + ((mul >> 11) & 1);  // round
    uint16_t max_value = (1 << input_bits) - 1;
    output(x, y) = Halide::select(out > max_value, max_value, Halide::cast(Halide::UInt(16), out));

    return output;
}

class LensShadingCorrectionLinear : public BuildingBlock<LensShadingCorrectionLinear> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "LensShadingCorrectionLinear(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Correct lens shading."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    // Max 16bit
    GeneratorParam<int32_t> width{"width", 0, 0, 65535};
    GeneratorParam<int32_t> height{"height", 0, 0, 65535};
    GeneratorParam<int32_t> input_bits{"input_bits", 16, 1, 16};
    // 12bit fractional
    GeneratorInput<uint16_t> slope_r{"slope_r"};
    GeneratorInput<uint16_t> slope_g{"slope_g"};
    GeneratorInput<uint16_t> slope_b{"slope_b"};
    GeneratorInput<uint16_t> offset_r{"offset_r"};
    GeneratorInput<uint16_t> offset_g{"offset_g"};
    GeneratorInput<uint16_t> offset_b{"offset_b"};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 2};

    void generate() {
        output = lens_shading_correction_linear(input, static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern)), width, height, input_bits,
                                                slope_r, slope_g, slope_b, offset_r, offset_g, offset_b);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (get_target().has_fpga_feature()) {
            output.bound(x, 0, width).bound(y, 0, height);
            output.accelerate({input}, {}, Var::outermost());
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xi, yi, xii, yii;
            output.align_bounds(x, 2).align_bounds(y, 2);
            output.gpu_tile(x, y, xi, yi, 32, 32).tile(xi, yi, xii, yii, 2, 2).unroll(xii).unroll(yii);
        } else {
            Halide::Var xi, yi;
            output.align_bounds(x, 2).align_bounds(y, 2);
            output.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi).vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }
        output.compute_root();
#endif
    }
};

Halide::Func calc_luminance(Halide::Func input, Luminance::Method luminance_method, std::string name = "calc_luminance") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"};

    output(x, y) = Luminance::calc(luminance_method, input(0, x, y), input(1, x, y), input(2, x, y));

    return output;
}

class CalcLuminance : public BuildingBlock<CalcLuminance> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "CalcLuminance(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Calc luminance of image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.slice(0, -1) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    //GeneratorParam<Luminance::Method> luminance_method { "luminance_method", Luminance::Method::SimpleY, Luminance::enum_map };
    GeneratorParam<int32_t> luminance_method{"luminance_method", 2, 0, 2};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 2};

    void generate() {
        output = calc_luminance(input, static_cast<Luminance::Method>(static_cast<int32_t>(luminance_method)));
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (get_target().has_fpga_feature()) {
            output.bound(x, 0, width).bound(y, 0, height);
            std::vector<Func> ip_in, ip_out;
            std::tie(ip_in, ip_out) = output.accelerate({input}, {}, Var::outermost());
            ip_in[0].hls_burst(3);
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func resize_bilinear_3d(Halide::Func input, int32_t width, int32_t height, float scale, std::string name = "resize_bilinear_3d") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    // 12bit, fractional 8bit
    int16_t scale12 = static_cast<int16_t>(256 * static_cast<float>(scale) + 0.5f);

    Halide::Func input_wrapper = Halide::BoundaryConditions::repeat_edge(input, {{}, {0, width}, {0, height}});

    Halide::Expr map_x = ((Halide::cast(Halide::Int(33), x) << 16) + (1 << 15)) / Halide::Expr(scale12) - (1 << 8);
    Halide::Expr map_y = ((Halide::cast(Halide::Int(33), y) << 16) + (1 << 15)) / Halide::Expr(scale12) - (1 << 8);

    Halide::Expr x0 = Halide::cast(Halide::Int(18), map_x >> 8);
    Halide::Expr y0 = Halide::cast(Halide::Int(18), map_y >> 8);
    Halide::Expr x1 = x0 + 1;
    Halide::Expr y1 = y0 + 1;
    Halide::Expr x_coef = Halide::cast(Halide::UInt(8), Halide::select(map_x < 0, 255 - (map_x & 255), map_x & 255));
    Halide::Expr y_coef = Halide::cast(Halide::UInt(8), Halide::select(map_y < 0, 255 - (map_y & 255), map_y & 255));

    Halide::Expr out0 = (Halide::cast(Halide::UInt(24), input_wrapper(c, x0, y0)) * (Halide::cast(Halide::UInt(9), 256) - x_coef) + Halide::cast(Halide::UInt(24), input_wrapper(c, x1, y0)) * x_coef) >> 8;
    Halide::Expr out1 = (Halide::cast(Halide::UInt(24), input_wrapper(c, x0, y1)) * (Halide::cast(Halide::UInt(9), 256) - x_coef) + Halide::cast(Halide::UInt(24), input_wrapper(c, x1, y1)) * x_coef) >> 8;
    output(c, x, y) = Halide::cast(Halide::UInt(16), (out0 * (Halide::cast(Halide::UInt(9), 256) - y_coef) + out1 * y_coef) >> 8);

    return output;
}

// NOTE: Generates a huge circuit depending on scale value.
class ResizeBilinear3D : public BuildingBlock<ResizeBilinear3D> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "ResizeBilinear3D(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Resize image by bilinear algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map((x, i) => i == 0 ? x : Math.floor(x * parseFloat(v.scale))) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    // Max 16bit
    GeneratorParam<int32_t> width{"width", 0, 0, 65535};
    GeneratorParam<int32_t> height{"height", 0, 0, 65535};
    // Output size max 16bit
    GeneratorParam<float> scale{"scale", 1.f, 0.5f, 15.f};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 3};

    void generate() {
        output = resize_bilinear_3d(input, width, height, scale);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var c = output.args()[0];
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        output.bound(x, 0, static_cast<int32_t>(static_cast<int32_t>(width) * scale)).bound(y, 0, static_cast<int32_t>(static_cast<int32_t>(height) * scale)).bound(c, 0, 3);
        if (get_target().has_fpga_feature()) {
            std::vector<Func> ip_in, ip_out;
            std::tie(ip_in, ip_out) = output.accelerate({input}, {}, Var::outermost());
            ip_in[0].hls_burst(3);
            ip_out[0].unroll(ip_out[0].args()[0]).hls_burst(3);
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func bayer_downscale(Halide::Func input, int32_t downscale_factor, std::string name = "bayer_downscale") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"};

    output(x, y) = input(x / 2 * 2 * downscale_factor + x % 2, y / 2 * 2 * downscale_factor + y % 2);

    return output;
}

class BayerDownscaleUInt16 : public BuildingBlock<BayerDownscaleUInt16> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "BayerDownscaleUInt16(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Downscale bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.map(x => Math.floor(x / parseInt(v.downscale_factor))) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> downscale_factor{"downscale_factor", 1};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 2};

    void generate() {
        output = bayer_downscale(input, downscale_factor);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (get_target().has_fpga_feature()) {
            output.bound(x, 0, width / downscale_factor).bound(y, 0, height / downscale_factor);
            output.accelerate({input}, {}, Var::outermost());
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func normalize_raw_image(Halide::Func input, int32_t input_bits, int32_t input_shift, int32_t output_bits, std::string name = "normalize_raw_image") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"};

    Halide::Expr in = (input(x, y) >> input_shift) & ((1 << input_bits) - 1);
    Halide::Expr out = 0;
    for (int32_t bits = 0; bits < output_bits; bits += input_bits) {
        int32_t shift = output_bits - bits - input_bits;
        if (shift < 0) {
            out += in >> -shift;
        } else {
            out += in << shift;
        }
    }
    output(x, y) = Halide::cast(Halide::UInt(16), out);

    return output;
}

class NormalizeRawImage : public BuildingBlock<NormalizeRawImage> {
public:
    // GeneratorParam<std::string> gc_title{"gc_title", "Normalize RAW(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Normalize raw image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "inlinable"};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> input_bits{"input_bits", 10, 1, 16};
    GeneratorParam<int32_t> input_shift{"input_shift", 6, 0, 15};
    GeneratorParam<int32_t> output_bits{"output_bits", 16, 1, 16};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(16), 2};

    void generate() {
        output = normalize_raw_image(input, input_bits, input_shift, output_bits);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::Var x = output.args()[0];
        Halide::Var y = output.args()[1];

        if (get_target().has_fpga_feature()) {
            output.bound(x, 0, width).bound(y, 0, height);
            output.accelerate({input}, {}, Var::outermost());
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
#endif
    }
};

Halide::Func convolution_3d(Halide::Func input, std::vector<int16_t> kernel, int32_t width, int32_t height, int32_t window_size, int input_bits, std::string name = "convolution_3d") {
    Halide::Func output{name};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    Halide::Func wrapper = Halide::BoundaryConditions::repeat_edge(input, {{}, {0, width}, {0, height}});

    int32_t n = window_size * 2 + 1;
    int32_t n2 = n * n;
    int32_t clogn = 0;
    for (uint32_t i = n2 - 1; i; i >>= 1)
        clogn++;
    int32_t sum_bit = 33 + clogn;  // 33 = 16(input bit) + 1(input sign bit) + 16(kernel bit)

    Halide::RDom r;
    Halide::Expr sum;
    sum = Halide::cast(Halide::Int(sum_bit), 0);
    for (int ry = 0; ry < n; ry++) {
        for (int rx = 0; rx < n; rx++) {
            sum += Halide::cast(Halide::Int(sum_bit), Halide::Expr(kernel[rx + ry * n])) * wrapper(c, x + (rx - window_size), y + (ry - window_size));
        }
    }
    Halide::Expr out = (sum >> 12) + ((sum >> 11) & 1);  // round
    uint16_t max_value = (1 << input_bits) - 1;
    output(c, x, y) = Halide::select(out > max_value, max_value,
                                     out < 0, 0,
                                     Halide::cast(Halide::UInt(16), out));

    return output;
}

class SimpleISP : public BuildingBlock<SimpleISP> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Simple ISP(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Make RGB image from RAW image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3].concat(v.input.map(x => Math.floor(x / 2))) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<std::string> gc_required_features{"gc_required_features", "vivado_hls"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    // Max 16bit
    GeneratorParam<int32_t> width{"width", 0, 0, 65535};
    GeneratorParam<int32_t> height{"height", 0, 0, 65535};
    GeneratorParam<int32_t> normalize_input_bits{"normalize_input_bits", 10, 1, 16};
    GeneratorParam<int32_t> normalize_input_shift{"normalize_input_shift", 6, 0, 15};
    GeneratorParam<uint16_t> offset_offset_r{"offset_offset_r", 0};
    GeneratorParam<uint16_t> offset_offset_g{"offset_offset_g", 0};
    GeneratorParam<uint16_t> offset_offset_b{"offset_offset_b", 0};
    // 12bit fractional
    GeneratorParam<uint16_t> white_balance_gain_r{"white_balance_gain_r", 4096};
    GeneratorParam<uint16_t> white_balance_gain_g{"white_balance_gain_g", 4096};
    GeneratorParam<uint16_t> white_balance_gain_b{"white_balance_gain_b", 4096};
    GeneratorParam<double> gamma_gamma{"gamma_gamma", 1 / 2.2};
    GeneratorParam<int32_t> unroll_level{"unroll_level", 3, 0, 3};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(8), 3};

    void generate() {
        int32_t internal_bits = normalize_input_bits;
        BayerMap::Pattern pattern = static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern));
        int32_t input_width = width;
        int32_t input_height = height;
        int32_t output_width = input_width / 2;
        int32_t output_height = input_height / 2;
        int32_t gamma_unroll = get_target().has_fpga_feature() && unroll_level != 0 ? (1 << (unroll_level - 1)) * 3 : 1;

        normalize = normalize_raw_image(input, normalize_input_bits, normalize_input_shift, internal_bits);
        offset = bayer_offset(normalize, pattern, offset_offset_r, offset_offset_g, offset_offset_b);
        white_balance = bayer_white_balance(offset, pattern, internal_bits, white_balance_gain_r, white_balance_gain_g, white_balance_gain_b);
        demosaic = bayer_demosaic_simple(white_balance, pattern);
        gamma = gamma_correction_3d(demosaic, internal_bits, 8, 8, 8, gamma_gamma, output_width, gamma_unroll);

        final_cast = Halide::Func{static_cast<std::string>(gc_prefix) + "simple_isp"};
        final_cast(Halide::_) = Halide::cast(UInt(8), gamma(Halide::_));

        output = final_cast;
    }

    void schedule() {
        Halide::Var c = output.args()[0];
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (get_target().has_fpga_feature()) {
            int32_t input_width = width;
            int32_t input_height = height;
            output.bound(c, 0, 3).bound(x, 0, input_width / 2).bound(y, 0, input_height / 2);

            std::vector<Func> ip_in, ip_out;
            std::tie(ip_in, ip_out) = output.accelerate({input}, {}, Var::outermost());

            normalize.compute_at(output, Halide::Var::outermost());
            offset.compute_at(output, Halide::Var::outermost());
            white_balance.compute_at(output, Halide::Var::outermost());
            demosaic.compute_at(output, Halide::Var::outermost());
            gamma.compute_at(output, Halide::Var::outermost());

            int32_t input_burst = 1 << unroll_level;
            int32_t dim0_unroll = unroll_level == 0 ? 1 : 3;
            int32_t dim1_unroll = unroll_level == 0 ? 1 : (1 << (unroll_level - 1));
            int32_t output_burst = dim0_unroll * dim1_unroll;

            ip_in[0].unroll(ip_in[0].args()[0], input_burst).hls_burst(input_burst);
            normalize.unroll(normalize.args()[0], input_burst).hls_burst(input_burst);
            offset.unroll(offset.args()[0], input_burst).hls_burst(input_burst);
            white_balance.unroll(white_balance.args()[0], input_burst).hls_burst(input_burst);
            demosaic.bound(demosaic.args()[0], 0, 3).unroll(demosaic.args()[0], dim0_unroll).unroll(demosaic.args()[1], dim1_unroll).hls_burst(output_burst);
            gamma.bound(gamma.args()[0], 0, 3).unroll(gamma.args()[0], dim0_unroll).unroll(gamma.args()[1], dim1_unroll).hls_burst(output_burst);
            ip_out[0].bound(ip_out[0].args()[0], 0, 3).unroll(ip_out[0].args()[0], dim0_unroll).unroll(ip_out[0].args()[1], dim1_unroll).hls_burst(output_burst);
        } else if (get_target().has_gpu_feature()) {
            Halide::Var xo, yo, xi, yi;
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
    }

private:
    Halide::Func normalize;
    Halide::Func offset;
    Halide::Func white_balance;
    Halide::Func demosaic;
    Halide::Func gamma;
    Halide::Func final_cast;
};

class SimpleISPWithUnsharpMask : public BuildingBlock<SimpleISPWithUnsharpMask> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Simple ISP with unsharp mask(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Make RGB image from RAW image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3].concat(v.input.map(x => Math.floor(x / 2))) }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<std::string> gc_required_features{"gc_required_features", "vivado_hls"};

    //GeneratorParam<BayerMap::Pattern> bayer_pattern { "bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map };
    GeneratorParam<int32_t> bayer_pattern{"bayer_pattern", 0, 0, 3};
    // Max 16bit
    GeneratorParam<int32_t> width{"width", 0, 0, 65535};
    GeneratorParam<int32_t> height{"height", 0, 0, 65535};
    GeneratorParam<int32_t> normalize_input_bits{"normalize_input_bits", 10, 1, 16};
    GeneratorParam<int32_t> normalize_input_shift{"normalize_input_shift", 6, 0, 15};
    GeneratorParam<uint16_t> offset_offset_r{"offset_offset_r", 0};
    GeneratorParam<uint16_t> offset_offset_g{"offset_offset_g", 0};
    GeneratorParam<uint16_t> offset_offset_b{"offset_offset_b", 0};
    // 12bit fractional
    GeneratorParam<uint16_t> white_balance_gain_r{"white_balance_gain_r", 4096};
    GeneratorParam<uint16_t> white_balance_gain_g{"white_balance_gain_g", 4096};
    GeneratorParam<uint16_t> white_balance_gain_b{"white_balance_gain_b", 4096};
    GeneratorParam<double> gamma_gamma{"gamma_gamma", 1 / 2.2};
    GeneratorParam<int32_t> unroll_level{"unroll_level", 3, 0, 3};
    GeneratorInput<Halide::Func> input{"input", Halide::UInt(16), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::UInt(8), 3};

    void generate() {
        int32_t internal_bits = normalize_input_bits;
        BayerMap::Pattern pattern = static_cast<BayerMap::Pattern>(static_cast<int32_t>(bayer_pattern));
        int32_t input_width = width;
        int32_t input_height = height;
        int32_t output_width = input_width / 2;
        int32_t output_height = input_height / 2;
        int32_t gamma_unroll = get_target().has_fpga_feature() && unroll_level != 0 ? (1 << (unroll_level - 1)) * 3 : 1;
        std::vector<int16_t> unsharp_mask_kernel = {-455, -455, -455,
                                                    -455, 7736, -455,
                                                    -455, -455, -455};

        normalize = normalize_raw_image(input, normalize_input_bits, normalize_input_shift, internal_bits);
        offset = bayer_offset(normalize, pattern, offset_offset_r, offset_offset_g, offset_offset_b);
        white_balance = bayer_white_balance(offset, pattern, internal_bits, white_balance_gain_r, white_balance_gain_g, white_balance_gain_b);
        demosaic = bayer_demosaic_simple(white_balance, pattern, static_cast<std::string>(gc_prefix) + "bayer_demosaic_simple");
        unsharp_mask = convolution_3d(demosaic, unsharp_mask_kernel, output_width, output_height, 1, internal_bits);
        gamma = gamma_correction_3d(unsharp_mask, internal_bits, 8, 8, 8, gamma_gamma, output_width, gamma_unroll);

        final_cast = Halide::Func{static_cast<std::string>(gc_prefix) + "simple_isp"};
        final_cast(Halide::_) = Halide::cast(UInt(8), gamma(Halide::_));

        output = final_cast;
    }

    void schedule() {
        Halide::Var c = output.args()[0];
        Halide::Var x = output.args()[1];
        Halide::Var y = output.args()[2];

        if (get_target().has_fpga_feature()) {
            int32_t input_width = width;
            int32_t input_height = height;
            output.bound(c, 0, 3).bound(x, 0, input_width / 2).bound(y, 0, input_height / 2);

            std::vector<Func> ip_in, ip_out;
            std::tie(ip_in, ip_out) = output.accelerate({input}, {}, Var::outermost());

            normalize.compute_at(output, Halide::Var::outermost());
            offset.compute_at(output, Halide::Var::outermost());
            white_balance.compute_at(output, Halide::Var::outermost());
            demosaic.compute_at(output, Halide::Var::outermost());
            unsharp_mask.compute_at(output, Halide::Var::outermost());
            gamma.compute_at(output, Halide::Var::outermost());

            int32_t input_burst = 1 << unroll_level;
            int32_t dim0_unroll = unroll_level == 0 ? 1 : 3;
            int32_t dim1_unroll = unroll_level == 0 ? 1 : (1 << (unroll_level - 1));
            int32_t output_burst = dim0_unroll * dim1_unroll;

            if (input_burst > 1) {
                ip_in[0].unroll(ip_in[0].args()[0], input_burst).hls_burst(input_burst);
                normalize.unroll(normalize.args()[0], input_burst).hls_burst(input_burst);
                offset.unroll(offset.args()[0], input_burst).hls_burst(input_burst);
                white_balance.unroll(white_balance.args()[0], input_burst).hls_burst(input_burst);
            }
            if (dim0_unroll > 1) {
                demosaic.bound(demosaic.args()[0], 0, 3).unroll(demosaic.args()[0], dim0_unroll);
                unsharp_mask.bound(unsharp_mask.args()[0], 0, 3).unroll(unsharp_mask.args()[0], dim0_unroll);
                gamma.bound(gamma.args()[0], 0, 3).unroll(gamma.args()[0], dim0_unroll);
                ip_out[0].bound(ip_out[0].args()[0], 0, 3).unroll(ip_out[0].args()[0], dim0_unroll);
            }
            if (dim1_unroll > 1) {
                demosaic.unroll(demosaic.args()[1], dim1_unroll);
                unsharp_mask.unroll(unsharp_mask.args()[1], dim1_unroll);
                gamma.unroll(gamma.args()[1], dim1_unroll);
                ip_out[0].unroll(ip_out[0].args()[1], dim1_unroll);
            }
            if (output_burst > 1) {
                demosaic.hls_burst(output_burst);
                unsharp_mask.hls_burst(output_burst);
                gamma.hls_burst(output_burst);
                ip_out[0].hls_burst(output_burst);
            }
        } else if (get_target().has_gpu_feature()) {
            Halide::Var d_xo, d_yo, d_xi, d_yi;
            Halide::Var xo, yo, xi, yi;
            demosaic.gpu_tile(demosaic.args()[1], demosaic.args()[2], d_xo, d_yo, d_xi, d_yi, 32, 16);
            demosaic.compute_root();
            output.gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        } else {
            demosaic.vectorize(demosaic.args()[1], natural_vector_size(Halide::Float(32))).parallel(demosaic.args()[2], 16);
            demosaic.compute_root();
            output.vectorize(x, natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }

        output.compute_root();
    }

private:
    Halide::Func normalize;
    Halide::Func offset;
    Halide::Func shading_correction;
    Halide::Func white_balance;
    Halide::Func demosaic;
    Halide::Func unsharp_mask;
    Halide::Func gamma;
    Halide::Func final_cast;
};

}  // namespace fpga
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::BayerOffset, fpga_bayer_offset);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::BayerWhiteBalance, fpga_bayer_white_balance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::BayerDemosaicSimple, fpga_bayer_demosaic_simple);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::GammaCorrection3D, fpga_gamma_correction_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::LensShadingCorrectionLinear, fpga_lens_shading_correction_linear);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::CalcLuminance, fpga_calc_luminance);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::ResizeBilinear3D, fpga_resize_bilinear_3d);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::BayerDownscaleUInt16, fpga_bayer_downscale_uint16);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::NormalizeRawImage, fpga_normalize_raw_image);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::SimpleISP, fpga_simple_isp);
ION_REGISTER_BUILDING_BLOCK(ion::bb::fpga::SimpleISPWithUnsharpMask, fpga_simple_isp_with_unsharp_mask);

#endif
