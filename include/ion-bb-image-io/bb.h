#ifndef ION_BB_IMAGE_IO_BB_H
#define ION_BB_IMAGE_IO_BB_H

#include <ion/ion.h>
#ifndef _WIN32
#include <linux/videodev2.h>
#endif

#include "sole.hpp"

namespace ion {
namespace bb {
namespace image_io {

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

#ifndef _WIN32
uint32_t make_pixel_format(BayerMap::Pattern bayer_pattern, int32_t bit_width)
{
    uint32_t pix_format;
    switch (bit_width * 10 + static_cast<int32_t>(static_cast<BayerMap::Pattern>(bayer_pattern))) {
    case 80:  // RGGB 8bit
        pix_format = V4L2_PIX_FMT_SRGGB8;
        break;
    case 81:  // BGGR 8bit
        pix_format = V4L2_PIX_FMT_SBGGR8;
        break;
    case 82:  // GRBG 8bit
        pix_format = V4L2_PIX_FMT_SGRBG8;
        break;
    case 83:  // GBRG 8bit
        pix_format = V4L2_PIX_FMT_SGBRG8;
        break;
    case 100:  // RGGB 10bit
        pix_format = V4L2_PIX_FMT_SRGGB10;
        break;
    case 101:  // BGGR 10bit
        pix_format = V4L2_PIX_FMT_SBGGR10;
        break;
    case 102:  // GRBG 10bit
        pix_format = V4L2_PIX_FMT_SGRBG10;
        break;
    case 103:  // GBRG 10bit
        pix_format = V4L2_PIX_FMT_SGBRG10;
        break;
    case 120:  // RGGB 12bit
        pix_format = V4L2_PIX_FMT_SRGGB12;
        break;
    case 121:  // BGGR 12bit
        pix_format = V4L2_PIX_FMT_SBGGR12;
        break;
    case 122:  // GRBG 12bit
        pix_format = V4L2_PIX_FMT_SGRBG12;
        break;
    case 123:  // GBRG 12bit
        pix_format = V4L2_PIX_FMT_SGBRG12;
        break;
    default:
        throw std::runtime_error("Unsupported pixel_format combination");
    }

    return pix_format;
}

int instance_id = 0;

class IMX219 : public ion::BuildingBlock<IMX219> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "IMX219"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures IMX219 image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> fps{"fps", 24};
    GeneratorParam<int32_t> width{"width", 3264};
    GeneratorParam<int32_t> height{"height", 2464};
    GeneratorParam<int32_t> index{"index", 0};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorParam<bool> force_sim_mode{"force_sim_mode", false};

    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;
        std::string url_str = url;
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {
            instance_id++,
            cast<int32_t>(index),
            cast<int32_t>(fps),
            cast<int32_t>(width),
            cast<int32_t>(height),
            cast<uint32_t>(Expr(V4L2_PIX_FMT_SRGGB10)),
            cast<uint32_t>(force_sim_mode),
            url_buf,
            0.4f, 0.5f, 0.3125f,
            0.0625f,
            10, 6
        };
        Func v4l2_imx219(static_cast<std::string>(gc_prefix) + "output");
        v4l2_imx219.define_extern("ion_bb_image_io_v4l2", params, type_of<uint16_t>(), 2);
        v4l2_imx219.compute_root();

        output = v4l2_imx219;
    }
};

class D435 : public ion::BuildingBlock<D435> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "D435"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures D435 stereo image and depth."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output_l: [1280, 720], output_r: [1280, 720], output_d: [1280, 720] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorOutput<Halide::Func> output_l{"output_l", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output_r{"output_r", Halide::type_of<uint8_t>(), 2};
    GeneratorOutput<Halide::Func> output_d{"output_d", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;
        Func realsense_d435_frameset(static_cast<std::string>(gc_prefix) + "frameset");
        realsense_d435_frameset.define_extern("ion_bb_image_io_realsense_d435_frameset", {}, type_of<uint64_t>(), 0);
        realsense_d435_frameset.compute_root();

        // TODO: Seperate channel
        Func realsense_d435_infrared(static_cast<std::string>(gc_prefix) + "output_lr");
        realsense_d435_infrared.define_extern("ion_bb_image_io_realsense_d435_infrared", {realsense_d435_frameset}, {type_of<uint8_t>(), type_of<uint8_t>()}, 2);
        realsense_d435_infrared.compute_root();

        Func realsense_d435_depth(static_cast<std::string>(gc_prefix) + "output_d");
        realsense_d435_depth.define_extern("ion_bb_image_io_realsense_d435_depth", {realsense_d435_frameset}, type_of<uint16_t>(), 2);
        realsense_d435_depth.compute_root();

        output_l(_) = realsense_d435_infrared(_)[0];
        output_r(_) = realsense_d435_infrared(_)[1];
        output_d = realsense_d435_depth;
    }
};

class Camera : public ion::BuildingBlock<Camera> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "USBCamera"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures USB camera image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), 3] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> fps{"fps", 30};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> index{"index", 0};
    GeneratorParam<std::string> url{"url", ""};

    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        using namespace Halide;
        std::string url_str = url;
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {instance_id++, cast<int32_t>(index), cast<int32_t>(fps), cast<int32_t>(width), cast<int32_t>(height), url_buf};
        Func camera(static_cast<std::string>(gc_prefix) + "camera");
        camera.define_extern("ion_bb_image_io_camera", params, Halide::type_of<uint8_t>(), 2);
        camera.compute_root();

        Func camera_ = BoundaryConditions::repeat_edge(camera, {{0, 2 * width}, {0, height}});

        Var c, x, y;

        Expr yv = cast<float>(camera_(2 * x, y));
        Expr uv = cast<float>(camera_(select((x & 1) == 0, 2 * x + 1, 2 * x - 1), y));
        Expr vv = cast<float>(camera_(select((x & 1) == 0, 2 * x + 3, 2 * x + 1), y));

        Expr f128 = cast<float>(128);

        Expr r = saturating_cast<uint8_t>(yv + cast<float>(1.403f) * (vv - f128));
        Expr g = saturating_cast<uint8_t>(yv - cast<float>(0.344f) * (uv - f128) - (cast<float>(0.714f) * (vv - f128)));
        Expr b = saturating_cast<uint8_t>(yv + cast<float>(1.773f) * (uv - f128));

        Func f(static_cast<std::string>(gc_prefix) + "output");
        f(x, y, c) = mux(c, {r, g, b});

        output = f;
    }
};

class GenericV4L2Bayer : public ion::BuildingBlock<GenericV4L2Bayer> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "GenericV4L2Bayer"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures Bayer image from V4L2."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> index{"index", 0};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorParam<int32_t> fps{"fps", 20};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> bit_width{"bit_width", 10};
    GeneratorParam<BayerMap::Pattern> bayer_pattern{"bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;

        std::string url_str = url;
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {
            instance_id++,
            cast<int32_t>(index),
            cast<int32_t>(fps),
            cast<int32_t>(width),
            cast<int32_t>(height),
            Expr(make_pixel_format(bayer_pattern, bit_width)),
            cast<uint32_t>(0),
            url_buf,
            1.f, 1.f, 1.f,
            0.f,
            cast<int32_t>(bit_width), 16 - bit_width
        };
        Func v4l2(static_cast<std::string>(gc_prefix) + "output");
        v4l2.define_extern("ion_bb_image_io_v4l2", params, type_of<uint16_t>(), 2);
        v4l2.compute_root();

        output = v4l2;
    }
};

class CameraSimulation : public ion::BuildingBlock<CameraSimulation> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "CameraSimulation"};
    GeneratorParam<std::string> gc_description{"gc_description", "This simulates Bayer image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> fps{"fps", 30};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorParam<BayerMap::Pattern> bayer_pattern{"bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map};
    GeneratorParam<int32_t> bit_width{"bit_width", 10};
    GeneratorParam<int32_t> bit_shift{"bit_shift", 0};
    GeneratorParam<float> gain_r{"gain_r", 1.f};
    GeneratorParam<float> gain_g{"gain_g", 1.f};
    GeneratorParam<float> gain_b{"gain_b", 1.f};
    GeneratorParam<float> offset{"offset", 0.f};

    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;
        std::string url_str = url;
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {
            instance_id++,
            0,
            cast<int32_t>(fps),
            cast<int32_t>(width),
            cast<int32_t>(height),
            Expr(make_pixel_format(bayer_pattern, bit_width)),
            cast<uint32_t>(1),
            url_buf,
            cast<float>(gain_r), cast<float>(gain_g), cast<float>(gain_b),
            cast<float>(offset),
            cast<int32_t>(bit_width), cast<int32_t>(bit_shift)
        };
        Func camera(static_cast<std::string>(gc_prefix) + "output");
        camera.define_extern("ion_bb_image_io_v4l2", params, type_of<uint16_t>(), 2);
        camera.compute_root();

        output = camera;
    }
};
#endif

class GUIDisplay : public ion::BuildingBlock<GUIDisplay> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "GUI Display"};
    GeneratorParam<std::string> gc_description{"gc_description", "This renders RGB image on GUI window."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "output,display"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: []  }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> idx{"idx", 0};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Int(32), 0};

    void generate() {
        using namespace Halide;

        Func in(static_cast<std::string>(gc_prefix) + "input");
        Var x, y, c;
        in(c, x, y) = mux(c,
                          {input(x, y, 2),
                           input(x, y, 1),
                           input(x, y, 0)});
        in.compute_root();
        if (get_target().has_gpu_feature()) {
            Var xo, yo, xi, yi;
            in.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        } else {
            in.parallel(y);
        }

        std::vector<ExternFuncArgument> params = {in, static_cast<int>(width), static_cast<int>(height), static_cast<int>(idx)};
        Func display(static_cast<std::string>(gc_prefix) + "output");
        display.define_extern("ion_bb_image_io_gui_display", params, Int(32), 0);
        display.compute_root();

        output = display;
    }
};

#ifndef _WIN32
class FBDisplay : public ion::BuildingBlock<FBDisplay> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "FBDisplay"};
    GeneratorParam<std::string> gc_description{"gc_description", "This draws image into framebuffer display."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "output,display"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Int(32), 0};

    void generate() {
        using namespace Halide;

        Func in(static_cast<std::string>(gc_prefix) + "input");
        Var x, y, c;
        in(c, x, y) = mux(c,
                          {input(x, y, 2),
                           input(x, y, 1),
                           input(x, y, 0)});
        in.compute_root();
        if (get_target().has_gpu_feature()) {
            Var xo, yo, xi, yi;
            in.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        } else {
            in.parallel(y);
        }

        std::vector<ExternFuncArgument> params = {cast<int32_t>(width), cast<int32_t>(height), in};
        Func display(static_cast<std::string>(gc_prefix) + "output");
        display.define_extern("ion_bb_image_io_fb_display", params, Halide::type_of<int32_t>(), 0);
        display.compute_root();

        output = display;
    }
};

class GrayscaleDataLoader : public ion::BuildingBlock<GrayscaleDataLoader> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Data Loader / Grayscale"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 16-bit grayscale image from specified URL."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> dynamic_range{"dynamic_range", 65535};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

    void generate() {
        using namespace Halide;

        const std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string url_str(url);
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {session_id_buf, url_buf, static_cast<int32_t>(width), static_cast<int32_t>(height), static_cast<int32_t>(dynamic_range)};
        Func grayscale_data_loader(static_cast<std::string>(gc_prefix) + "output");
        grayscale_data_loader.define_extern("ion_bb_image_io_grayscale_data_loader", params, Halide::type_of<uint16_t>(), 2);
        grayscale_data_loader.compute_root();

        output = grayscale_data_loader;
    }
};

class ColorDataLoader : public ion::BuildingBlock<ColorDataLoader> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Data Loader / Color"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads 8-bit/RGB/CHW image from specified URL."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), 3] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<std::string> url{"url", ""};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        using namespace Halide;

        const std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string url_str(url);
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());

        std::vector<ExternFuncArgument> params = {session_id_buf, url_buf, static_cast<int32_t>(width), static_cast<int32_t>(height)};
        Func color_data_loader(static_cast<std::string>(gc_prefix) + "output");
        color_data_loader.define_extern("ion_bb_image_io_color_data_loader", params, Halide::type_of<uint8_t>(), 3);
        color_data_loader.compute_root();

        output = color_data_loader;
    }
};
#endif

class ImageSaver : public ion::BuildingBlock<ImageSaver> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Image Saver"};
    GeneratorParam<std::string> gc_description{"gc_description", "This saves image to specified path."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "output,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<std::string> path{"path", ""};
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    GeneratorOutput<Halide::Func> output{"output", Halide::Int(32), 0};

    void generate() {
        using namespace Halide;
        std::string path_str(path);
        Halide::Buffer<uint8_t> path_buf(path_str.size() + 1);
        path_buf.fill(0);
        std::memcpy(path_buf.data(), path_str.c_str(), path_str.size());

        Func in(static_cast<std::string>(gc_prefix) + "input");
        Var x, y, c;
        in(c, x, y) = mux(c,
                          {input(x, y, 2),
                           input(x, y, 1),
                           input(x, y, 0)});
        in.compute_root();
        if (get_target().has_gpu_feature()) {
            Var xo, yo, xi, yi;
            in.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        } else {
            in.parallel(y);
        }

        std::vector<ExternFuncArgument> params = {in, static_cast<int32_t>(width), static_cast<int32_t>(height), path_buf};
        Func image_saver(static_cast<std::string>(gc_prefix) + "output");
        image_saver.define_extern("ion_bb_image_io_image_saver", params, Int(32), 0);
        image_saver.compute_root();
        output = image_saver;
    }
};

template<typename T, int D>
class U3VCamera1 : public ion::BuildingBlock<U3VCamera1<T, D>> {
public:

    GeneratorParam<bool> frame_sync{"frame_sync", false};
    GeneratorParam<std::string> pixel_format_ptr{"pixel_format_ptr", "RGB8"};
    GeneratorParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    GeneratorParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};
    GeneratorParam<bool> realtime_diaplay_mode{"realtime_diaplay_mode", false};

    GeneratorInput<bool> dispose{ "dispose" };
    GeneratorInput<double> gain0{ "gain0" };
    GeneratorInput<double> exposure0{ "exposure0" };

    GeneratorOutput<Halide::Func> output0{ "output0", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };

    void generate() {
        using namespace Halide;

        const std::string pixel_format(pixel_format_ptr);
        Buffer<uint8_t> pixel_format_buf(static_cast<int>(pixel_format.size() + 1));
        pixel_format_buf.fill(0);
        std::memcpy(pixel_format_buf.data(), pixel_format.c_str(), pixel_format.size());

        const std::string gain_key(gain_key_ptr);
        Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
        gain_key_buf.fill(0);
        std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

        const std::string exposure_key(exposure_key_ptr);
        Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
        exposure_key_buf.fill(0);
        std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

        std::vector<ExternFuncArgument> params{
            static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode),
            gain0, exposure0, pixel_format_buf,
            gain_key_buf, exposure_key_buf
         };

        Func camera1("u3v_camera1");
        camera1.define_extern("ion_bb_image_io_u3v_camera1", params, Halide::type_of<T>(), D);
        camera1.compute_root();
        output0(_) = camera1(_);

        Buffer<uint8_t> pixel_format_buf_cpy(static_cast<int>(pixel_format.size() + 1));
        pixel_format_buf_cpy.fill(0);
        std::memcpy(pixel_format_buf_cpy.data(), pixel_format.c_str(), pixel_format.size());

        Func camera1_frame_count;
        camera1_frame_count.define_extern("ion_bb_image_io_u3v_camera1_frame_count", { camera1, dispose, 1, static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode), pixel_format_buf_cpy}, type_of<uint32_t>(), 1);
        camera1_frame_count.compute_root();
        frame_count(_) = camera1_frame_count(_);
    }
};

using U3VCamera1_U8x3 = U3VCamera1<uint8_t, 3>;
using U3VCamera1_U8x2 = U3VCamera1<uint8_t, 2>;
using U3VCamera1_U16x2 = U3VCamera1<uint16_t, 2>;

template<typename T, int D>
class U3VCamera2 : public ion::BuildingBlock<U3VCamera2<T, D>> {
public:

    GeneratorParam<bool> frame_sync{"frame_sync", false};
    GeneratorParam<std::string> pixel_format_ptr{"pixel_format_ptr", "RGB8"};
    GeneratorParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    GeneratorParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};
    GeneratorParam<bool> realtime_diaplay_mode{"realtime_diaplay_mode", false};

    GeneratorInput<bool> dispose{ "dispose" };
    GeneratorInput<double> gain0{ "gain0" };
    GeneratorInput<double> gain1{ "gain1" };
    GeneratorInput<double> exposure0{ "exposure0" };
    GeneratorInput<double> exposure1{ "exposure1" };

    GeneratorOutput<Halide::Func> output0{ "output0", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output1{ "output1", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };

    void generate() {
        using namespace Halide;

        const std::string pixel_format(pixel_format_ptr);
        Buffer<uint8_t> pixel_format_buf(static_cast<int>(pixel_format.size() + 1));
        pixel_format_buf.fill(0);
        std::memcpy(pixel_format_buf.data(), pixel_format.c_str(), pixel_format.size());

        const std::string gain_key(gain_key_ptr);
        Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
        gain_key_buf.fill(0);
        std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

        const std::string exposure_key(exposure_key_ptr);
        Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
        exposure_key_buf.fill(0);
        std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

        std::vector<ExternFuncArgument> params{
            static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode),
            gain0, gain1, exposure0, exposure1, pixel_format_buf,
            gain_key_buf, exposure_key_buf
         };

        Func camera2("u3v_camera2");
        camera2.define_extern("ion_bb_image_io_u3v_camera2", params, { Halide::type_of<T>(), Halide::type_of<T>() }, D);
        camera2.compute_root();
        output0(_) = camera2(_)[0];
        output1(_) = camera2(_)[1];

        Buffer<uint8_t> pixel_format_buf_cpy(static_cast<int>(pixel_format.size() + 1));
        pixel_format_buf_cpy.fill(0);
        std::memcpy(pixel_format_buf_cpy.data(), pixel_format.c_str(), pixel_format.size());

        Func camera2_frame_count;
        camera2_frame_count.define_extern("ion_bb_image_io_u3v_camera2_frame_count", { camera2, dispose, 2, static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode), pixel_format_buf_cpy}, type_of<uint32_t>(), 1);
        camera2_frame_count.compute_root();
        frame_count(_) = camera2_frame_count(_);
    }
};

using U3VCamera2_U8x3 = U3VCamera2<uint8_t, 3>;
using U3VCamera2_U8x2 = U3VCamera2<uint8_t, 2>;
using U3VCamera2_U16x2 = U3VCamera2<uint16_t, 2>;

template<typename T, int D>
class U3VCameraN : public ion::BuildingBlock<U3VCameraN<T, D>> {
public:
    GeneratorParam<int32_t> num_devices{"num_devices", 2};

    GeneratorParam<bool> frame_sync{"frame_sync", false};
    GeneratorParam<std::string> pixel_format_ptr{"pixel_format_ptr", "RGB8"};
    GeneratorParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    GeneratorParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};
    GeneratorParam<bool> realtime_diaplay_mode{"realtime_diaplay_mode", false};

    GeneratorInput<bool> dispose{ "dispose", false };
    GeneratorInput<Halide::Func> gain{ "gain", Halide::type_of<double>(), 1};
    GeneratorInput<Halide::Func> exposure{ "exposure", Halide::type_of<double>(), 1};

    GeneratorOutput<Halide::Func[]> output{ "output", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };

    void generate() {
        using namespace Halide;

        Func gain_func;
        gain_func(_) = gain(_);
        gain_func.compute_root();

        Func exposure_func;
        exposure_func(_) = exposure(_);
        exposure_func.compute_root();

        Func cameraN("u3v_cameraN");
        {
            const std::string pixel_format(pixel_format_ptr);
            Buffer<uint8_t> pixel_format_buf(static_cast<int>(pixel_format.size() + 1));
            pixel_format_buf.fill(0);
            std::memcpy(pixel_format_buf.data(), pixel_format.c_str(), pixel_format.size());

            const std::string gain_key(gain_key_ptr);
            Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
            gain_key_buf.fill(0);
            std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

            const std::string exposure_key(exposure_key_ptr);
            Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
            exposure_key_buf.fill(0);
            std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

            std::vector<ExternFuncArgument> params{
                dispose, static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode),
                    gain_func, exposure_func, pixel_format_buf,
                    gain_key_buf, exposure_key_buf
            };

            output.resize(num_devices);
            if (output.size() == 1){
                cameraN.define_extern("ion_bb_image_io_u3v_multiple_camera" + std::to_string(output.size()), params, Halide::type_of<T>(), D);
            }else{
                std::vector<Halide::Type> output_type;
                for (int i = 0; i < output.size(); i++) {
                    output_type.push_back(Halide::type_of<T>());
                }
                cameraN.define_extern("ion_bb_image_io_u3v_multiple_camera" + std::to_string(output.size()), params, output_type, D);

            }
            cameraN.compute_root();
            if (output.size() == 1){
                output[0](_) = cameraN(_);
            }else{
                for (int i = 0; i < output.size(); i++) {
                    output[i](_) = cameraN(_)[i];
                }
            }
        }

        Func cameraN_fc("u3v_cameraN_fc");
        {
            const std::string pixel_format(pixel_format_ptr);
            Buffer<uint8_t> pixel_format_buf(static_cast<int>(pixel_format.size() + 1));
            pixel_format_buf.fill(0);
            std::memcpy(pixel_format_buf.data(), pixel_format.c_str(), pixel_format.size());

            std::vector<ExternFuncArgument> params{
                cameraN, dispose, static_cast<int32_t>(output.size()), static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode), pixel_format_buf
            };
            cameraN_fc.define_extern("ion_bb_image_io_u3v_multiple_camera_frame_count" + std::to_string(output.size()), params, type_of<uint32_t>(), 1);
            cameraN_fc.compute_root();
            frame_count(_) = cameraN_fc(_);
        }
    }
};

using U3VCameraN_U8x3 = U3VCameraN<uint8_t, 3>;
using U3VCameraN_U8x2 = U3VCameraN<uint8_t, 2>;
using U3VCameraN_U16x2 = U3VCameraN<uint16_t, 2>;

class U3VGenDC : public ion::BuildingBlock<U3VGenDC> {
public:
    GeneratorParam<int32_t> num_devices{"num_devices", 2};

    GeneratorParam<bool> frame_sync{"frame_sync", false};
    GeneratorParam<std::string> pixel_format_ptr{"pixel_format_ptr", "RGB8"};
    GeneratorParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    GeneratorParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};
    GeneratorParam<bool> realtime_diaplay_mode{"realtime_diaplay_mode", false};

    GeneratorInput<bool> dispose{ "dispose" };
    GeneratorInput<Halide::Func> gain{ "gain", Halide::type_of<double>(), 1};
    GeneratorInput<Halide::Func> exposure{ "exposure", Halide::type_of<double>(), 1};

    GeneratorOutput<Halide::Func[]> gendc{ "gendc", Halide::type_of<uint8_t>(), 1};
    GeneratorOutput<Halide::Func[]> device_info{ "device_info", Halide::type_of<uint8_t>(), 1};

    void generate() {
        using namespace Halide;

        Func gain_func;
        gain_func(_) = gain(_);
        gain_func.compute_root();

        Func exposure_func;
        exposure_func(_) = exposure(_);
        exposure_func.compute_root();

        const std::string pixel_format(pixel_format_ptr);
        Buffer<uint8_t> pixel_format_buf(static_cast<int>(pixel_format.size() + 1));
        pixel_format_buf.fill(0);
        std::memcpy(pixel_format_buf.data(), pixel_format.c_str(), pixel_format.size());

        const std::string gain_key(gain_key_ptr);
        Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
        gain_key_buf.fill(0);
        std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

        const std::string exposure_key(exposure_key_ptr);
        Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
        exposure_key_buf.fill(0);
        std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

        std::vector<ExternFuncArgument> params{
            dispose, static_cast<bool>(frame_sync), static_cast<bool>(realtime_diaplay_mode),
            gain_func, exposure_func, pixel_format_buf,
            gain_key_buf, exposure_key_buf
         };

        Func u3v_gendc("u3v_gendc");
        gendc.resize(num_devices);
        device_info.resize(num_devices);
        std::vector<Halide::Type> output_type;
        for (int i = 0; i < gendc.size() * 2; i++) {
            output_type.push_back(Halide::type_of<uint8_t>());
        }
        u3v_gendc.define_extern("ion_bb_image_io_u3v_gendc_camera" + std::to_string(gendc.size()), params, output_type, 1);
        u3v_gendc.compute_root();
        for (int i = 0; i < gendc.size(); i++) {
            gendc[i](_) = u3v_gendc(_)[2*i];
            device_info[i](_) = u3v_gendc(_)[2*i+1];
        }
    }
};

class BinarySaver : public ion::BuildingBlock<BinarySaver> {
public:
    GeneratorParam<std::string> output_directory_ptr{ "output_directory", "." };
    GeneratorParam<float> fps{ "fps", 1.0 };
    GeneratorParam<float> r_gain0{ "r_gain0", 1.0 };
    GeneratorParam<float> g_gain0{ "g_gain0", 1.0 };
    GeneratorParam<float> b_gain0{ "b_gain0", 1.0 };

    GeneratorParam<float> r_gain1{ "r_gain1", 1.0 };
    GeneratorParam<float> g_gain1{ "g_gain1", 1.0 };
    GeneratorParam<float> b_gain1{ "b_gain1", 1.0 };

    GeneratorParam<int32_t> offset0_x{ "offset0_x", 0 };
    GeneratorParam<int32_t> offset0_y{ "offset0_y", 0 };
    GeneratorParam<int32_t> offset1_x{ "offset1_x", 0 };
    GeneratorParam<int32_t> offset1_y{ "offset1_y", 0 };

    GeneratorParam<int32_t> outputsize0_x{ "outputsize0_x", 1 };
    GeneratorParam<int32_t> outputsize0_y{ "outputsize0_y", 1 };
    GeneratorParam<int32_t> outputsize1_x{ "outputsize1_x", 1 };
    GeneratorParam<int32_t> outputsize1_y{ "outputsize1_y", 1 };

    Input<Halide::Func> input0{ "input0", UInt(16), 2 };
    Input<Halide::Func> input1{ "input1", UInt(16), 2 };
    Input<Halide::Func> frame_count{ "frame_count", UInt(32), 1 };
    Input<bool> dispose{ "dispose" };
    Input<int32_t> width{ "width", 0 };
    Input<int32_t> height{ "height", 0 };

    Output<int> output{ "output" };
    void generate() {
        using namespace Halide;
        Func in0;
        in0(_) = input0(_);
        in0.compute_root();

        Func in1;
        in1(_) = input1(_);
        in1.compute_root();

        const std::string output_directory(output_directory_ptr);
        Halide::Buffer<uint8_t> output_directory_buf(static_cast<int>(output_directory.size() + 1));
        output_directory_buf.fill(0);
        std::memcpy(output_directory_buf.data(), output_directory.c_str(), output_directory.size());

        Func fc;
        fc(_) = frame_count(_);
        fc.compute_root();
        std::vector<ExternFuncArgument> params = { in0, in1, fc, dispose, width, height, output_directory_buf,
            static_cast<float>(r_gain0), static_cast<float>(g_gain0), static_cast<float>(b_gain0),
            static_cast<float>(r_gain1), static_cast<float>(g_gain1), static_cast<float>(b_gain1),
            static_cast<int32_t>(offset0_x), static_cast<int32_t>(offset0_x),
            static_cast<int32_t>(offset0_x), static_cast<int32_t>(offset1_y),
            static_cast<int32_t>(outputsize0_x), static_cast<int32_t>(outputsize0_y),
            static_cast<int32_t>(outputsize1_x), static_cast<int32_t>(outputsize1_y),
            cast<float>(fps) };
        Func binarysaver;
        binarysaver.define_extern("binarysaver", params, Int(32), 0);
        binarysaver.compute_root();
        output() = binarysaver();
    }
};

class BinaryGenDCSaver : public ion::BuildingBlock<BinaryGenDCSaver> {
public:
    GeneratorParam<std::string> output_directory_ptr{ "output_directory", "." };

    GeneratorParam<int32_t> num_devices{"num_devices", 2};

    Input<Halide::Func[]> input_gendc{ "input_gendc", Halide::type_of<uint8_t>(), 1 };
    Input<Halide::Func[]> input_deviceinfo{ "input_deviceinfo", Halide::type_of<uint8_t>(), 1 };

    Input<bool> dispose{ "dispose" };
    Input<int32_t> payloadsize{ "payloadsize" };

    Output<int> output{ "output" };

    void generate() {
        using namespace Halide;
        int32_t num_gendc = static_cast<int32_t>(num_devices);

        const std::string output_directory(output_directory_ptr);
        Halide::Buffer<uint8_t> output_directory_buf(static_cast<int>(output_directory.size() + 1));
        output_directory_buf.fill(0);
        std::memcpy(output_directory_buf.data(), output_directory.c_str(), output_directory.size());

        if (num_gendc==1){
            Func gendc;
            gendc(_) = input_gendc(_);
            gendc.compute_root();

            Func deviceinfo;
            deviceinfo(_) = input_deviceinfo(_);
            deviceinfo.compute_root();

            std::vector<ExternFuncArgument> params = { gendc, deviceinfo, dispose, payloadsize, output_directory_buf };
            Func image_io_binary_gendc_saver;
            image_io_binary_gendc_saver.define_extern("ion_bb_image_io_binary_1gendc_saver", params, Int(32), 0);
            image_io_binary_gendc_saver.compute_root();
            output() = image_io_binary_gendc_saver();
        }else if (num_gendc ==2){
            Func gendc0, gendc1;
            Var x, y;
            gendc0(_) = input_gendc[0](_);
            gendc1(_) = input_gendc[1](_);
            gendc0.compute_root();
            gendc1.compute_root();

            Func deviceinfo0, deviceinfo1;
            deviceinfo0(_) = input_deviceinfo[0](_);
            deviceinfo1(_) = input_deviceinfo[1](_);
            deviceinfo0.compute_root();
            deviceinfo1.compute_root();

            std::vector<ExternFuncArgument> params = { gendc0, gendc1, deviceinfo0, deviceinfo1, dispose, payloadsize, output_directory_buf };
            Func image_io_binary_gendc_saver;
            image_io_binary_gendc_saver.define_extern("ion_bb_image_io_binary_2gendc_saver", params, Int(32), 0);
            image_io_binary_gendc_saver.compute_root();
            output() = image_io_binary_gendc_saver();
        }else{
            std::runtime_error("device number > 2 is not supported");
        }


    }
};

class BinaryLoader : public ion::BuildingBlock<BinaryLoader> {
public:
    GeneratorParam<std::string> output_directory_ptr{ "output_directory_ptr", "" };
    Input<int32_t> width{ "width", 0 };
    Input<int32_t> height{ "height", 0 };
    Output<Halide::Func> output0{ "output0", UInt(16), 2 };
    Output<Halide::Func> output1{ "output1", UInt(16), 2 };
    Output<Halide::Func> finished{ "finished", UInt(1), 1};
    Output<Halide::Func> bin_idx{ "bin_idx", UInt(32), 1 };

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(static_cast<int>(session_id.size() + 1));
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string output_directory(output_directory_ptr);
        Halide::Buffer<uint8_t> output_directory_buf(static_cast<int>(output_directory.size() + 1));
        output_directory_buf.fill(0);
        std::memcpy(output_directory_buf.data(), output_directory.c_str(), output_directory.size());

        std::vector<ExternFuncArgument> params = { session_id_buf, width, height, output_directory_buf };
        Func binaryloader;
        binaryloader.define_extern("binaryloader", params, { UInt(16), UInt(16) }, 2);
        binaryloader.compute_root();
        output0(_) = binaryloader(_)[0];
        output1(_) = binaryloader(_)[1];


        Func binaryloader_finished;
        binaryloader_finished.define_extern("binaryloader_finished",
            { binaryloader, session_id_buf, width, height, output_directory_buf },
            { type_of<bool>(), UInt(32)}, 1);
        binaryloader_finished.compute_root();
        finished(_) = binaryloader_finished(_)[0];
        bin_idx(_) = binaryloader_finished(_)[1];
    }
};

}  // namespace image_io
}  // namespace bb
}  // namespace ion

#ifndef _WIN32
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::IMX219, image_io_imx219);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::D435, image_io_d435);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::Camera, image_io_camera);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GenericV4L2Bayer, image_io_generic_v4l2_bayer);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::CameraSimulation, image_io_camera_simulation);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::FBDisplay, image_io_fb_display);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ColorDataLoader, image_io_color_data_loader);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GrayscaleDataLoader, image_io_grayscale_data_loader);
#endif

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GUIDisplay, image_io_gui_display);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ImageSaver, image_io_image_saver);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U8x3, image_io_u3v_camera1_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U16x2, image_io_u3v_camera1_u16x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U8x2, image_io_u3v_camera1_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U8x3, image_io_u3v_camera2_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U8x2, image_io_u3v_camera2_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U16x2, image_io_u3v_camera2_u16x2);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCameraN_U8x3, image_io_u3v_cameraN_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCameraN_U8x2, image_io_u3v_cameraN_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCameraN_U16x2, image_io_u3v_cameraN_u16x2);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VGenDC, image_io_u3v_gendc);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinarySaver, image_io_binarysaver);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinaryLoader, image_io_binaryloader);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinaryGenDCSaver, image_io_binary_gendc_saver);

//backward compatability
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U8x3, u3v_camera1_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U16x2, u3v_camera1_u16x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U8x3, u3v_camera2_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U16x2, u3v_camera2_u16x2);
#endif
