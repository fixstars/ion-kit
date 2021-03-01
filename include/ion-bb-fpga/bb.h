#ifndef ION_BB_FPGA_BB_H
#define ION_BB_FPGA_BB_H

#define HALIDE_FOR_FPGA
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

Halide::Func bayer_offset(Halide::Func input, BayerMap::Pattern bayer_pattern, Halide::Expr offset_r, Halide::Expr offset_g, Halide::Expr offset_b) {
    Halide::Func output{"bayer_offset"};
    Halide::Var x{"x"}, y{"y"};

    Halide::Expr offset = Halide::mux(BayerMap::get_color(bayer_pattern, x, y), {offset_r, offset_g, offset_b});
    output(x, y) = Halide::select(input(x, y) >= offset, input(x, y) - offset, 0);

    return output;
}

class BayerOffset : public BuildingBlock<BayerOffset> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerOffset(FPGA)"};
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
        Halide::var x = output.args()[0];
        Halide::var y = output.args()[1];

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

Halide::Func bayer_white_balance(Halide::Func input, BayerMap::Pattern bayer_pattern, int32_t input_bits, Halide::Expr gain_r, Halide::Expr gain_g, Halide::Expr gain_b) {
    Halide::Func output{"bayer_white_balance"};
    Halide::Var x{"x"}, y{"y"};

    Halide::Expr gain = Halide::mux(BayerMap::get_color(bayer_pattern, x, y), {gain_r, gain_g, gain_b});
    Halide::Expr mul = Halide::cast(UInt(32), input(x, y)) * gain;
    Halide::Expr out = (mul >> 12) + ((mul >> 11) & 1);  // round
    uint16_t max_value = (1 << input_bits) - 1;
    output(x, y) = Halide::select(out > max_value, max_value, Halide::cast(UInt(16), out));

    return output;
}

class BayerWhiteBalance : public BuildingBlock<BayerWhiteBalance> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerWhiteBalance(FPGA)"};
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
        Halide::var x = output.args()[0];
        Halide::var y = output.args()[1];

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

Halide::Func bayer_demosaic_simple(Halide::Func input, BayerMap::Pattern bayer_pattern) {
    Halide::Func output{"bayer_demosaic_simple"};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    switch (bayer_pattern) {
    case BayerMap::Pattern::RGGB:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2, y * 2),
                Halide::cast(UInt(16), (Halide::cast(UInt(17), input(x * 2 + 1, y * 2)) + input(x * 2, y * 2 + 1)) >> 1),
                input(x * 2 + 1, y * 2 + 1)});
        break;
    case BayerMap::Pattern::BGGR:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2 + 1, y * 2 + 1),
                Halide::cast(UInt(16), (Halide::cast(UInt(17), input(x * 2 + 1, y * 2)) + input(x * 2, y * 2 + 1)) >> 1),
                input(x * 2, y * 2)});
        break;
    case BayerMap::Pattern::GRBG:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2 + 1, y * 2),
                Halide::cast(UInt(16), (Halide::cast(UInt(17), input(x * 2, y * 2)) + input(x * 2 + 1, y * 2 + 1)) >> 1),
                input(x * 2 + 1, y * 2)});
        break;
    case BayerMap::Pattern::GBRG:
        output(c, x, y) = Halide::mux(
            c, {input(x * 2, y * 2 + 1),
                Halide::cast(UInt(16), (Halide::cast(UInt(17), input(x * 2, y * 2)) + input(x * 2 + 1, y * 2 + 1)) >> 1),
                input(x * 2 + 1, y * 2)});
        break;
    default:
        internal_error << "Unknown BayerMap pattern";
    }

    return output;
}

class BayerDemosaicSimple : public BuildingBlock<BayerDemosaicSimple> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerDemosaicSimple(FPGA)"};
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
        Halide::var c = output.args()[0];
        Halide::var x = output.args()[1];
        Halide::var y = output.args()[2];

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

Halide::Func gamma_correction_3d(Halide::Func input, int32_t input_bits, int32_t output_bits, int32_t lut_bits, int32_t lut_index_bits, double gamma) {
    Halide::Func output{"gamma_correction_3d"};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    int32_t lut_size = 1 << lut_index_bits;
    int32_t max_value = (1 << output_bits) - 1;
    if (input_bits == lut_index_bits) {
        std::vector<Halide::Buffer<uint16_t>> lut_list;
        for (int32_t i = 0; i < 3; i++) {
            Halide::Buffer<uint16_t> lut(lut_size);
            for (int32_t j = 0; j < lut_size; j++) {
                lut(j) = static_cast<uint16_t>(pow(j / static_cast<double>(lut_size - 1), gamma) * max_value + 0.5);
            }
            lut_list.push_back(lut);
        }
        output(c, x, y) = Halide::mux(c, {lut_list[0](input(c, x, y)),
                                          lut_list[1](input(c, x, y)),
                                          lut_list[2](input(c, x, y))});
    } else {
        std::vector<Halide::Buffer<uint16_t>> lut0_list;
        std::vector<Halide::Buffer<uint16_t>> lut1_list;
        for (int32_t i = 0; i < 3; i++) {
            Halide::Buffer<uint16_t> lut0(lut_size / 2);
            Halide::Buffer<uint16_t> lut1(lut_size / 2);
            for (int32_t j = 0; j < lut_size / 2; j++) {
                lut0(j) = static_cast<uint16_t>(pow((j * 2) / static_cast<double>(lut_size), gamma) * max_value + 0.5);
                lut1(j) = static_cast<uint16_t>(pow((j * 2 + 1) / static_cast<double>(lut_size), gamma) * max_value + 0.5);
            }
            lut0_list.push_back(lut0);
            lut1_list.push_back(lut1);
        }
        Halide::Expr is_odd_index = ((input(c, x, y) >> (input_bits - lut_index_bits)) & 1) == 1;
        Halide::Expr lut1_index = (input(c, x, y) >> (input_bits - lut_index_bits + 1)) & ((1 << (lut_index_bits - 1)) - 1);
        Halide::Expr lut0_index = Halide::select(is_odd_index, lut1_index + 1, lut1_index);
        Halide::Expr lut0_value = Halide::cast(UInt(lut_bits),
                                                Halide::select(lut0_index == lut_size / 2, max_value,
                                                                Halide::mux(c, {lut0_list[0](lut0_index),
                                                                                lut0_list[1](lut0_index),
                                                                                lut0_list[2](lut0_index)})));
        Halide::Expr lut1_value = Halide::cast(UInt(lut_bits), Halide::mux(c, {lut1_list[0](lut1_index),
                                                                               lut1_list[1](lut1_index),
                                                                               lut1_list[2](lut1_index)}));
        Halide::Expr base_value = Halide::select(is_odd_index, lut1_value, lut0_value);
        Halide::Expr next_value = Halide::select(is_odd_index, lut0_value, lut1_value);
        Halide::Expr diff = next_value - base_value;
        Halide::Expr coef = input(c, x, y) & ((1 << (input_bits - lut_index_bits)) - 1);
        Halide::Expr mul = (Halide::cast(UInt(lut_bits + input_bits - lut_index_bits), diff) * coef) >> (lut_bits + input_bits - lut_index_bits - output_bits);
        output(c, x, y) = Halide::cast(UInt(16), (Halide::cast(UInt(output_bits), base_value) << (output_bits - lut_bits)) + mul);
    }

    return output;
}

class GammaCorrection3D : public BuildingBlock<GammaCorrection3D> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "GammaCorrection3D(FPGA)"};
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
        output = gamma_correction_3d(input, input_bits, output_bits, lut_bits, lut_index_bits, gamma);
    }

    void schedule() {
#ifndef DISABLE_SCHEDULE
        Halide::var c = output.args()[0];
        Halide::var x = output.args()[1];
        Halide::var y = output.args()[2];

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
                                            Halide::Expr offset_r, Halide::Expr offset_g, Halide::Expr offset_b) {
    Halide::Func output{"lens_shading_correction_linear"};
    Halide::Var x{"x"}, y{"y"};

    int32_t center_x = width / 2;                                 // max 15bit
    int32_t center_y = height / 2;                                // max 15bit
    uint32_t r2_max = center_x * center_x + center_y * center_y;  // max 31bit

    int32_t dividend_bits = std::max(bit_width(center_x) * 2, bit_width(center_y) * 2) + 1 + 16;  // max 47bit

    Halide::Expr r2 = Halide::cast(UInt(16), (Halide::cast(UInt(dividend_bits), (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) << 16) / Halide::Expr(r2_max));  // 16bit

    Halide::Expr color = BayerMap::get_color(bayer_pattern, x, y);
    Halide::Expr coef_mul = (Halide::cast(UInt(32), r2) * Halide::mux(color, {slope_r, slope_g, slope_b})) >> 16;  // 16bit
    Halide::Expr coef = Halide::cast(UInt(17), coef_mul) + Halide::mux(color, {offset_r, offset_g, offset_b});     // 17bit
    Halide::Expr coef_clamp = Halide::select(coef >= 65536, 65535, Halide::cast(UInt(16), coef));                  // 16bit

    Halide::Expr mul = Halide::cast(UInt(32), input(x, y)) * coef_clamp;
    Halide::Expr out = (mul >> 12) + ((mul >> 11) & 1);  // round
    uint16_t max_value = (1 << input_bits) - 1;
    output(x, y) = Halide::select(out > max_value, max_value, Halide::cast(UInt(16), out));

    return output;
}

class LensShadingCorrectionLinear : public BuildingBlock<LensShadingCorrectionLinear> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "LensShadingCorrectionLinear(FPGA)"};
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
        Halide::var x = output.args()[0];
        Halide::var y = output.args()[1];

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

Halide::Func calc_luminance(Halide::Func input, Luminance::Method luminance_method) {
    Halide::Func output{"calc_luminance"};
    Halide::Var x{"x"}, y{"y"};

    output(x, y) = Luminance::calc(luminance_method, input(0, x, y), input(1, x, y), input(2, x, y));

    return output;
}

class CalcLuminance : public BuildingBlock<CalcLuminance> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CalcLuminance(FPGA)"};
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
        Halide::var x = output.args()[0];
        Halide::var y = output.args()[1];

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

Halide::Func resize_bilinear_3d(Halide::Func input, int32_t width, int32_t height, float scale) {
    Halide::Func output{"resize_bilinear_3d"};
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    // 12bit, fractional 8bit
    int16_t scale12 = static_cast<int16_t>(256 * static_cast<float>(scale) + 0.5f);

    Halide::Func input_wrapper = Halide::BoundaryConditions::repeat_edge(input, {{}, {0, width}, {0, height}});

    Halide::Expr map_x = ((Halide::cast(Int(33), x) << 16) + (1 << 15)) / Halide::Expr(scale12) - (1 << 8);
    Halide::Expr map_y = ((Halide::cast(Int(33), y) << 16) + (1 << 15)) / Halide::Expr(scale12) - (1 << 8);

    Halide::Expr x0 = Halide::cast(Int(18), map_x >> 8);
    Halide::Expr y0 = Halide::cast(Int(18), map_y >> 8);
    Halide::Expr x1 = x0 + 1;
    Halide::Expr y1 = y0 + 1;
    Halide::Expr x_coef = Halide::cast(UInt(8), Halide::select(map_x < 0, 255 - (map_x & 255), map_x & 255));
    Halide::Expr y_coef = Halide::cast(UInt(8), Halide::select(map_y < 0, 255 - (map_y & 255), map_y & 255));

    Halide::Expr out0 = (Halide::cast(UInt(24), input_wrapper(c, x0, y0)) * (Halide::cast(UInt(9), 256) - x_coef) + Halide::cast(UInt(24), input_wrapper(c, x1, y0)) * x_coef) >> 8;
    Halide::Expr out1 = (Halide::cast(UInt(24), input_wrapper(c, x0, y1)) * (Halide::cast(UInt(9), 256) - x_coef) + Halide::cast(UInt(24), input_wrapper(c, x1, y1)) * x_coef) >> 8;
    output(c, x, y) = Halide::cast(UInt(16), (out0 * (Halide::cast(UInt(9), 256) - y_coef) + out1 * y_coef) >> 8);

    return output;
}

// NOTE: Generates a huge circuit depending on scale value.
class ResizeBilinear3D : public BuildingBlock<ResizeBilinear3D> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "ResizeBilinear3D(FPGA)"};
    GeneratorParam<std::string> gc_description{"gc_description", "Resize image by bilinear algorithm."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input.slice(0, -1).map(x => Math.floor(x * parseFloat(v.scale))).concat(v.input.slice(-1)) }}))"};
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
        Halide::var c = output.args()[0];
        Halide::var x = output.args()[1];
        Halide::var y = output.args()[2];

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

Halide::Func bayer_downscale(Halide::Func input, int32_t downscale_factor) {
    Halide::Func output{"bayer_downscale"};
    Halide::Var x{"x"}, y{"y"};

    output(x, y) = input(x / 2 * 2 * downscale_factor + x % 2, y / 2 * 2 * downscale_factor + y % 2);

    return output;
}

class BayerDownscaleUInt16 : public BuildingBlock<BayerDownscaleUInt16> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "BayerDownscaleUInt16(FPGA)"};
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
        Halide::var x = output.args()[0];
        Halide::var y = output.args()[1];

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

Halide::Func normalize_raw_image(Halide::Func input, int32_t input_bits, int32_t input_shift, int32_t output_bits) {
    Halide::Func output{"normalize_raw_image"};
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
    output(x, y) = Halide::cast(UInt(16), out);

    return output;
}

class NormalizeRawImage : public BuildingBlock<NormalizeRawImage> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize RAW(FPGA)"};
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
        Halide::var x = output.args()[0];
        Halide::var y = output.args()[1];

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

#endif
