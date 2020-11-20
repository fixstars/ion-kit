#ifndef ION_BB_GENESIS_CLOUD_BB_H
#define ION_BB_GENESIS_CLOUD_BB_H

#include <numeric>

#include "ion/ion.h"

namespace ion {
namespace bb {
namespace genesis_cloud {

class Camera : public ion::BuildingBlock<Camera> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "USBCamera"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures USB camera image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [3, parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        using namespace Halide;
        std::vector<ExternFuncArgument> params = {cast<int32_t>(width), cast<int32_t>(height)};
        Func camera(static_cast<std::string>(gc_prefix) + "camera");
        camera.define_extern("ion_bb_genesis_cloud_camera", params, Halide::type_of<uint8_t>(), 2);
        camera.compute_root();

        Func camera_ = BoundaryConditions::repeat_edge(camera, {{0, 2*width}, {0, height}});

        Var c, x, y;

        Expr yv = cast<float>(camera_(2 * x, y));
        Expr uv = cast<float>(camera_(select((x & 1) == 0, 2 * x + 1, 2 * x - 1), y));
        Expr vv = cast<float>(camera_(select((x & 1) == 0, 2 * x + 3, 2 * x + 1), y));

        Expr f128 = cast<float>(128);

        Expr r = saturating_cast<uint8_t>(yv + cast<float>(1.403f) * (vv - f128));
        Expr g = saturating_cast<uint8_t>(yv - cast<float>(0.344f) * (uv - f128) - (cast<float>(0.714f) * (vv - f128)));
        Expr b = saturating_cast<uint8_t>(yv + cast<float>(1.773f) * (uv - f128));

        output(c, x, y) = select(c == 0, r, c == 1, g, b);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::Camera, genesis_cloud_camera);

namespace ion {
namespace bb {
namespace genesis_cloud {

class FBDisplay : public ion::BuildingBlock<FBDisplay> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FBDisplay"};
    GeneratorParam<std::string> gc_description{"gc_description", "This draws image into framebuffer display."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "output,display"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self,assume_compute_root"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<int32_t> output{"output"};

    void generate() {
        using namespace Halide;

        Func input_(static_cast<std::string>(gc_prefix)+"input");
        input_(_) = input(_);
        input_.compute_root();

        std::vector<ExternFuncArgument> params = {cast<int32_t>(width), cast<int32_t>(height), input_};

        Func display(static_cast<std::string>(gc_prefix)+"display");
        display.define_extern("ion_bb_genesis_cloud_display", params, Halide::type_of<int32_t>(), 0);
        display.compute_root();
        output() = display();
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::FBDisplay, genesis_cloud_display);

namespace ion {
namespace bb {
namespace genesis_cloud {


class ImageLoader : public ion::BuildingBlock<ImageLoader> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Image Loader"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads image from specified URL."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [3, parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};
    // TODO: Make it possible to use scalar output
    // GeneratorOutput<Halide::Func> output_extent{"output_extent", Halide::type_of<int32_t>(), 1};
    void generate() {
        using namespace Halide;
        std::string url_str(url);
        Halide::Buffer<uint8_t> url_buf(url_str.size()+1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());
        std::vector<ExternFuncArgument> params = {url_buf};
        Func image_loader(static_cast<std::string>(gc_prefix)+"image_loader");
        image_loader.define_extern("ion_bb_genesis_cloud_image_loader", params, Halide::type_of<uint8_t>(), 3);
        image_loader.compute_root();
        Var c, x, y;
        output(c, x, y) = select(c == 0, image_loader(2, x, y),
                                 c == 1, image_loader(1, x, y),
                                         image_loader(0, x, y));
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ImageLoader, genesis_cloud_image_loader);

namespace ion {
namespace bb {
namespace genesis_cloud {

class ImageSaver : public ion::BuildingBlock<ImageSaver> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Image Saver"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves image to specified path."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "output,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self,assume_compute_root"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<std::string> path{"path", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<int32_t> output{"output"};

    void generate() {
        using namespace Halide;
        std::string path_str(path);
        Halide::Buffer<uint8_t> path_buf(path_str.size()+1);
        path_buf.fill(0);
        std::memcpy(path_buf.data(), path_str.c_str(), path_str.size());

        Func input_(static_cast<std::string>(gc_prefix)+"input");
        input_(_) = input(_);
        input_.compute_root();

        std::vector<ExternFuncArgument> params = {input_, static_cast<int32_t>(width), static_cast<int32_t>(height), path_buf};
        Func image_saver(static_cast<std::string>(gc_prefix)+"image_saver");
        image_saver.define_extern("ion_bb_genesis_cloud_image_saver", params, Int(32), 0);
        image_saver.compute_root();
        output() = image_saver();
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ImageSaver, genesis_cloud_image_saver);

namespace ion {
namespace bb {
namespace genesis_cloud {

template<typename X, typename T, int32_t D>
class Denormalize : public ion::BuildingBlock<X> {
public:
    GeneratorParam<std::string> gc_description{"gc_description", "This denormalize [0..1.0] values into target type range."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};
    void generate() {
        using namespace Halide;
        output(_) = saturating_cast<T>(input(_) * cast<float>((std::numeric_limits<T>::max)()));
    }
};

class DenormalizeU8x2 : public Denormalize<DenormalizeU8x2, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize U8x2"};
};

class DenormalizeU8x3 : public Denormalize<DenormalizeU8x3, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Denormalize U8x3"};
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::DenormalizeU8x2, genesis_cloud_denormalize_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::DenormalizeU8x3, genesis_cloud_denormalize_u8x3);

namespace ion {
namespace bb {
namespace genesis_cloud {

template<typename X, typename T, int32_t D>
class Normalize : public ion::BuildingBlock<X> {
public:
    GeneratorParam<std::string> gc_description{"gc_description", "This normalize values into range [0..1.0]."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), D};
    void generate() {
        using namespace Halide;
        output(_) = cast<float>(input(_)) / (std::numeric_limits<T>::max)();
    }
};

class NormalizeU8x2 : public Normalize<NormalizeU8x2, uint8_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize U8x2"};
};

class NormalizeU8x3 : public Normalize<NormalizeU8x3, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize U8x3"};
};

class NormalizeU16x2 : public Normalize<NormalizeU16x2, uint16_t, 2> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize U16x2"};
};

class NormalizeU16x3 : public Normalize<NormalizeU16x3, uint16_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Normalize U16x3"};
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::NormalizeU8x2, genesis_cloud_normalize_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::NormalizeU8x3, genesis_cloud_normalize_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::NormalizeU16x2, genesis_cloud_normalize_u16x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::NormalizeU16x3, genesis_cloud_normalize_u16x3);

namespace ion {
namespace bb {
namespace genesis_cloud {

template<typename X, typename T, int32_t D>
class Copy : public ion::BuildingBlock<X> {
public:
    GeneratorParam<std::string> gc_description{"gc_description", "This just copies input to output."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,file"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};
    void generate() {
        using namespace Halide;
        output(_) = input(_);
    }
};

class CopyI32x3 : public Copy<CopyI32x3, int32_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Copy I32x3"};
};

class CopyU8x3 : public Copy<CopyU8x3, uint8_t, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Copy U8x3"};
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::CopyI32x3, genesis_cloud_copy_i32x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::CopyU8x3, genesis_cloud_copy_u8x3);

namespace ion {
namespace bb {
namespace genesis_cloud {

class OpticalBlackClamp : public ion::BuildingBlock<OpticalBlackClamp> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Optical Black Clamp"};
    GeneratorParam<std::string> gc_description{"gc_description", "This subtracts each pixel values by clamp_value parameter value. This expects 16-bit RAW input and emits 16-bit RAW output."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<uint16_t> clamp_value{"clamp_value", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint16_t>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};
    void generate() {
        using namespace Halide;
        Var x, y;
        output(x, y) = input(x, y) - min(input(x, y), cast<uint16_t>(clamp_value));
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::OpticalBlackClamp, genesis_cloud_optical_black_clamp);

namespace ion {
namespace bb {
namespace genesis_cloud {

class ColorInterpolationRawToRGB : public ion::BuildingBlock<ColorInterpolationRawToRGB> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Color Interpolation RAW to RGB"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from Bayered 16-bit RAW into RGB (normalized)."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [3, v.input[0], v.input[1]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<int32_t> available_bits{"available_bits", 12};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint16_t>(), 2};
    GeneratorInput<int32_t> input_extent_0{"input_extent_0"};
    GeneratorInput<int32_t> input_extent_1{"input_extent_1"};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        Var x, y, c;

        Func in = BoundaryConditions::constant_exterior(input, 0, { {0, input_extent_0}, {0, input_extent_1} });

        Expr is_b  = (x % 2 == 0) && (y % 2 == 0);
        Expr is_gr = (x % 2 == 1) && (y % 2 == 0);
        Expr is_r  = (x % 2 == 0) && (y % 2 == 1);
        Expr is_gb = (x % 2 == 1) && (y % 2 == 1);

        Expr self = in(x, y);
        Expr hori = (in(x - 1, y) + in(x + 1, y)) / 2;
        Expr vert = (in(x, y - 1) + in(x, y + 1)) / 2;
        Expr latt = (in(x - 1, y) + in(x + 1, y) + in(x, y - 1) + in(x, y + 1)) / 4;
        Expr diag = (in(x - 1, y - 1) + in(x + 1, y - 1) + in(x - 1, y + 1) + in(x + 1, y + 1)) / 4;

        Expr max_value = cast<float>((1 << available_bits) -1);

        Expr r = cast<float>(select(is_r, self, is_gr, hori, is_gb, vert, diag)) / max_value;
        Expr g = cast<float>(select(is_r, latt, is_gr, diag, is_gb, diag, latt)) / max_value;
        Expr b = cast<float>(select(is_r, diag, is_gr, vert, is_gb, hori, self)) / max_value;

        output(c, x, y) = select(c == 0, r, c == 1, g, b);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ColorInterpolationRawToRGB, genesis_cloud_color_interpolation_raw_to_rgb);

namespace ion {
namespace bb {
namespace genesis_cloud {

class ColorInterpolationRGBToHSV : public ion::BuildingBlock<ColorInterpolationRGBToHSV> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Color Interpolation RGB to HSV"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from RGB into HSV."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        Var x, y, c;
        Expr zero = cast<float>(0.0f);
        Expr one  = cast<float>(1.0f);
        Expr two  = cast<float>(2.0f);
        Expr four = cast<float>(4.0f);
        Expr six  = cast<float>(6.0f);

        Expr r = input(0, x, y);
        Expr g = input(1, x, y);
        Expr b = input(2, x, y);

        Expr minv = min(r, min(g, b));
        Expr maxv = max(r, max(g, b));
        Expr diff = select(maxv == minv, one, maxv - minv);

        Expr h = select(maxv == minv, zero,
                           maxv == r,    (g-b)/diff,
                           maxv == g,    (b-r)/diff+two,
                           (r-g)/diff+four);

        h = select(h < zero, h+six, h) / six;

        Expr dmaxv = select(maxv == zero, one, maxv);
        Expr s = select(maxv == zero, zero, (maxv-minv)/dmaxv);
        Expr v = maxv;

        output(c, x, y) = select(c == 0, h, c == 1, s, v);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ColorInterpolationRGBToHSV, genesis_cloud_color_interpolation_rgb_to_hsv);


namespace ion {
namespace bb {
namespace genesis_cloud {

class ColorInterpolationHSVToRGB : public ion::BuildingBlock<ColorInterpolationHSVToRGB> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Color Interpolation HSV to RGB"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from HSV into RGB."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        Var x, y, c;

        Expr zero = cast<float>(0.0f);
        Expr one  = cast<float>(1.0f);
        Expr six  = cast<float>(6.0f);

        Expr h = input(0, x, y);
        Expr s = input(1, x, y);
        Expr v = input(2, x, y);

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

        output(c, x, y) = select(c == 0, r, c == 1, g, b);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ColorInterpolationHSVToRGB, genesis_cloud_color_interpolation_hsv_to_rgb);

namespace ion {
namespace bb {
namespace genesis_cloud {

class SaturationAdjustment : public ion::BuildingBlock<SaturationAdjustment> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Saturation Adjustment"};
    GeneratorParam<std::string> gc_description{"gc_description", "This applies saturation adjustment."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<float> saturation_value{"saturation_value", 1.0f};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;
        Var x, y, c;

        Expr zero = cast<float>(0.0f);
        Expr one = cast<float>(1.0f);

        Expr v = input(c, x, y);

        output(c, x, y) = select(c == 1, clamp(fast_pow(v, cast<float>(saturation_value)), zero, one), v);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::SaturationAdjustment, genesis_cloud_saturation_adjustment);

namespace ion {
namespace bb {
namespace genesis_cloud {

class GammaCorrection : public ion::BuildingBlock<GammaCorrection> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Gamma Correction"};
    GeneratorParam<std::string> gc_description{"gc_description", "This applies gamma correction."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<float> gamma_value{"gamma_value", 1.0f};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;
        Expr zero = cast<float>(0.0f);
        Expr one = cast<float>(1.0f);
        output(_) = clamp(fast_pow(input(_), cast<float>(gamma_value)), zero, one);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::GammaCorrection, genesis_cloud_gamma_correction);

namespace ion {
namespace bb {
namespace genesis_cloud {

class ColorInterpolationRGBToRaw : public ion::BuildingBlock<ColorInterpolationRGBToRaw> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Color Interpolation RGB to RAW"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from RGB24 into Bayered 16-bit RAW."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [v.input[1], v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<int32_t> available_bits{"available_bits", 12};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;

        Var x, y, c;

        Expr is_b =  (x % 2 == 0) && (y % 2 == 0);
        Expr is_g = ((x % 2 == 1) && (y % 2 == 0)) || ((x % 2 == 1) && (y % 2 == 1));
        Expr is_r =  (x % 2 == 0) && (y % 2 == 1);

        Expr shift = cast<uint16_t>(available_bits) - cast<uint16_t>(8);

        Expr r = cast<uint16_t>(input(0, x, y)) << shift;
        Expr g = cast<uint16_t>(input(1, x, y)) << shift;
        Expr b = cast<uint16_t>(input(2, x, y)) << shift;

        output(x, y) = select(is_r, r,
                              is_g, g,
                                    b);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ColorInterpolationRGBToRaw, genesis_cloud_color_interpolation_rgb_to_raw);

namespace ion {
namespace bb {
namespace genesis_cloud {

class ColorInterpolationRGB8ToMono8 : public ion::BuildingBlock<ColorInterpolationRGB8ToMono8> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Color Interpolation RGB8 to Mono8"};
    GeneratorParam<std::string> gc_description{"gc_description", "This converts color space from RGB8 into Mono8."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [v.input[1], v.input[2]] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 2};

    void generate() {
        using namespace Halide;

        Var x, y, c;

        Expr r = cast<uint16_t>(input(0, x, y));
        Expr g = cast<uint16_t>(input(1, x, y));
        Expr b = cast<uint16_t>(input(2, x, y));

        output(x, y) = cast<uint8_t>((r + g + b) / 3);
    }
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ColorInterpolationRGB8ToMono8, genesis_cloud_color_interpolation_RGB8_to_Mono8);

namespace ion {
namespace bb {
namespace genesis_cloud {

template<typename X, typename T>
class Scalex2 : public ion::BuildingBlock<X> {
public:
    GeneratorParam<std::string> gc_description{"gc_description", "This changes size of the image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: v.input.map(function(e){ return Math.trunc(e * parseFloat(v.scale)); })}; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height,scale"};
    GeneratorParam<int> input_width{"input_width", 0};
    GeneratorParam<int> input_height{"input_height", 0};
    GeneratorParam<float> scale{"scale", 1};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), 2};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 2};

    void generate() {
        using namespace Halide;
        Var x, y;
        Expr ix = cast<int>(round(cast<float>(x) / cast<float>(scale)));
        Expr iy = cast<int>(round(cast<float>(y) / cast<float>(scale)));
        output(x, y) = BoundaryConditions::constant_exterior(input, cast<uint8_t>(0), {{0, input_width}, {0, input_height}})(ix, iy);
    }
};

class ScaleU8x2 : public Scalex2<ScaleU8x2, uint8_t> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Scale U8x2"};
};

template<typename X, typename T>
class Scalex3 : public ion::BuildingBlock<X> {
public:
    GeneratorParam<std::string> gc_description{"gc_description", "This changes size of the image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference",  R"((function(v){ return { output: [v.input[0], Math.trunc(v.input[1] * parseFloat(v.scale)), Math.trunc(v.input[2] * parseFloat(v.scale))] }; }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "input_width,input_height,scale"};
    GeneratorParam<int> input_width{"input_width", 0};
    GeneratorParam<int> input_height{"input_height", 0};
    GeneratorParam<float> scale{"scale", 1};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), 3};

    void generate() {
        using namespace Halide;
        Var c, x, y;
        Expr ix = cast<int>(round(cast<float>(x) / cast<float>(scale)));
        Expr iy = cast<int>(round(cast<float>(y) / cast<float>(scale)));
        output(c, x, y) = BoundaryConditions::constant_exterior(input, cast<uint8_t>(0), {{0, 3}, {0, input_width}, {0, input_height}})(c, ix, iy);
    }
};


class ScaleU8x3 : public Scalex3<ScaleU8x3, uint8_t> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Scale U8x3"};
};

} // genesis_cloud
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ScaleU8x2, genesis_cloud_scale_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::genesis_cloud::ScaleU8x3, genesis_cloud_scale_u8x3);

// #include "bb_sgm.h"

#endif
