#ifndef ION_BB_IMAGE_IO_BB_H
#define ION_BB_IMAGE_IO_BB_H

#include <ion/ion.h>
#ifdef __linux__
#include <linux/videodev2.h>
#endif

#include "uuid/sole.hpp"

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

#ifdef __linux__
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


class Camera : public ion::BuildingBlock<Camera> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "USBCamera"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This captures USB camera image."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), 3] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> fps{"fps", 30};
    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<int32_t> index{"index", 0};
    BuildingBlockParam<std::string> url{"url", ""};

    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

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

class Camera2 : public ion::BuildingBlock<Camera2> {
public:
    BuildingBlockParam<int32_t> num_devices{"num_devices", 2};
    BuildingBlockParam<std::string> gc_title{"gc_title", "USBCamera"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This captures USB camera image."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), 3] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> fps{"fps", 30};
    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<int32_t> index{"index", 0};
    BuildingBlockParam<std::string> url0{"url0", ""};
    BuildingBlockParam<std::string> url1{"url1", ""};



    Output<Halide::Func> output0{"output0", Halide::type_of<uint8_t>(), 3};
    Output<Halide::Func> output1{"output1", Halide::type_of<uint8_t>(), 3};


    void generate() {
        using namespace Halide;


        for (int i =0; i < num_devices; i++){
            std::string url_str;
            if(i == 0){
                url_str = url0;
            }
            else{
                url_str = url1;
            }

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




            Func f(static_cast<std::string>(gc_prefix) + "output" + std::to_string(i));
            f(x, y, c) = mux(c, {r, g, b});


            if (i ==0)
                output0 = f;
            else
                output1 = f;
        }

    }
};


class CameraN : public ion::BuildingBlock<CameraN> {
public:
    BuildingBlockParam<int32_t> num_devices{"num_devices", 2};
    BuildingBlockParam<std::string> gc_title{"gc_title", "USBCamera"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This captures USB camera image."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), 3] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> fps{"fps", 30};
    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<int32_t> index{"index", 0};
    BuildingBlockParam<std::string> urls{"urls", ""};

    Output<Halide::Func[]> output{"output", Halide::type_of<uint8_t>(), 3};


    void generate() {

        std::stringstream urls_stream(urls);
        std::string url;
        std::vector<std::string> url_list;
        while(std::getline(urls_stream, url, ';'))
        {
            url_list.push_back(url);
        }


        using namespace Halide;

        output.resize(num_devices);

        for (int i =0; i < num_devices; i++){
            std::string url_str;
            if (url_list.size()!=0){
                url_str = url_list[i];
            }
            else{
                url_str = "";
            }



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


            Func f(static_cast<std::string>(gc_prefix) + "output" + std::to_string(i));
            f(x, y, c) = mux(c, {r, g, b});

            output[i](_) = f(_);
        }

    }
};

class IMX219 : public ion::BuildingBlock<IMX219> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "IMX219"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This captures IMX219 image."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> fps{"fps", 24};
    BuildingBlockParam<int32_t> width{"width", 3264};
    BuildingBlockParam<int32_t> height{"height", 2464};
    BuildingBlockParam<int32_t> index{"index", 0};
    BuildingBlockParam<std::string> url{"url", ""};
    BuildingBlockParam<bool> force_sim_mode{"force_sim_mode", false};

    Output<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

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
    BuildingBlockParam<std::string> gc_title{"gc_title", "D435"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This captures D435 stereo image and depth."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output_l: [1280, 720], output_r: [1280, 720], output_d: [1280, 720] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    Output<Halide::Func> output_l{"output_l", Halide::type_of<uint8_t>(), 2};
    Output<Halide::Func> output_r{"output_r", Halide::type_of<uint8_t>(), 2};
    Output<Halide::Func> output_d{"output_d", Halide::type_of<uint16_t>(), 2};

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


class GenericV4L2Bayer : public ion::BuildingBlock<GenericV4L2Bayer> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "GenericV4L2Bayer"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This captures Bayer image from V4L2."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> index{"index", 0};
    BuildingBlockParam<std::string> url{"url", ""};
    BuildingBlockParam<int32_t> fps{"fps", 20};
    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<int32_t> bit_width{"bit_width", 10};
    BuildingBlockParam<BayerMap::Pattern> bayer_pattern{"bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map};
    Output<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

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
    BuildingBlockParam<std::string> gc_title{"gc_title", "CameraSimulation"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This simulates Bayer image."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> fps{"fps", 30};
    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<std::string> url{"url", ""};
    BuildingBlockParam<BayerMap::Pattern> bayer_pattern{"bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map};
    BuildingBlockParam<int32_t> bit_width{"bit_width", 10};
    BuildingBlockParam<int32_t> bit_shift{"bit_shift", 0};
    BuildingBlockParam<float> gain_r{"gain_r", 1.f};
    BuildingBlockParam<float> gain_g{"gain_g", 1.f};
    BuildingBlockParam<float> gain_b{"gain_b", 1.f};
    BuildingBlockParam<float> offset{"offset", 0.f};

    Output<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

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
    BuildingBlockParam<std::string> gc_title{"gc_title", "GUI Display"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This renders RGB image on GUI window."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "output,display"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: []  }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> idx{"idx", 0};
    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    Output<Halide::Func> output{"output", Halide::Int(32), 0};

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

#ifdef __linux__
class FBDisplay : public ion::BuildingBlock<FBDisplay> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "FBDisplay"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This draws image into framebuffer display."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "output,display"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    Output<Halide::Func> output{"output", Halide::Int(32), 0};

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
#endif

class GrayscaleDataLoader : public ion::BuildingBlock<GrayscaleDataLoader> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "Data Loader / Grayscale"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This loads 16-bit grayscale image from specified URL."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,imgproc"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height)] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<int32_t> dynamic_range{"dynamic_range", 65535};
    BuildingBlockParam<std::string> url{"url", ""};
    Output<Halide::Func> output{"output", Halide::type_of<uint16_t>(), 2};

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
    BuildingBlockParam<std::string> gc_title{"gc_title", "Data Loader / Color"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This loads 8-bit/RGB/CHW image from specified URL."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "input,imgproc"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.width), parseInt(v.height), 3] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height,url"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<std::string> url{"url", ""};
    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

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


class ImageSaver : public ion::BuildingBlock<ImageSaver> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "Image Saver"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This saves image to specified path."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "output,imgproc"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<int32_t> width{"width", 0};
    BuildingBlockParam<int32_t> height{"height", 0};
    BuildingBlockParam<std::string> path{"path", ""};
    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    Output<Halide::Func> output{"output", Halide::Int(32), 0};

    void generate() {
        using namespace Halide;
        std::string path_str(path);
        Halide::Buffer<uint8_t> path_buf(path_str.size() + 1);
        path_buf.fill(0);
        std::memcpy(path_buf.data(), path_str.c_str(), path_str.size());

        Func in(static_cast<std::string>(gc_prefix) + "input");
        Var x, y, c;
        in(c, x, y) = mux(c,
                          {input(x, y, 0),
                           input(x, y, 1),
                           input(x, y, 2)});
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

    BuildingBlockParam<bool> frame_sync{"frame_sync", false};
    BuildingBlockParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    BuildingBlockParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};
    BuildingBlockParam<bool> realtime_display_mode{"realtime_display_mode", false};

    Input<double> gain0{ "gain0" };
    Input<double> exposure0{ "exposure0" };

    Output<Halide::Func> output0{ "output0", Halide::type_of<T>(), D};
    Output<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };

    void generate() {
        using namespace Halide;

        Func camera1("u3v_camera1");
        {
            Buffer<uint8_t> id_buf = this->get_id();

            const std::string gain_key(gain_key_ptr);
            Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
            gain_key_buf.fill(0);
            std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

            const std::string exposure_key(exposure_key_ptr);
            Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
            exposure_key_buf.fill(0);
            std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

            std::vector<ExternFuncArgument> params{
            static_cast<bool>(frame_sync), static_cast<bool>(realtime_display_mode),
            gain0, exposure0,
            id_buf, gain_key_buf, exposure_key_buf
         };
            camera1.define_extern("ion_bb_image_io_u3v_camera1", params, Halide::type_of<T>(), D);
            camera1.compute_root();
            output0(_) = camera1(_);
        }

        Func camera1_frame_count;
        {
            Buffer<uint8_t> id_buf = this->get_id();
            camera1_frame_count.define_extern("ion_bb_image_io_u3v_camera1_frame_count",{camera1, 1, static_cast<bool>(frame_sync), static_cast<bool>(realtime_display_mode), id_buf}, type_of<uint32_t>(), 1);
            camera1_frame_count.compute_root();
            frame_count(_) = camera1_frame_count(_);
        }

        this->register_disposer("u3v_dispose");

    }
};

using U3VCamera1_U8x3 = U3VCamera1<uint8_t, 3>;
using U3VCamera1_U8x2 = U3VCamera1<uint8_t, 2>;
using U3VCamera1_U16x2 = U3VCamera1<uint16_t, 2>;

template<typename T, int D>
class U3VCamera2 : public ion::BuildingBlock<U3VCamera2<T, D>> {
public:

    BuildingBlockParam<bool> frame_sync{"frame_sync", false};
    BuildingBlockParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    BuildingBlockParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};
    BuildingBlockParam<bool> realtime_display_mode{"realtime_display_mode", false};

    Input<double> gain0{ "gain0" };
    Input<double> gain1{ "gain1" };
    Input<double> exposure0{ "exposure0" };
    Input<double> exposure1{ "exposure1" };

    Output<Halide::Func> output0{ "output0", Halide::type_of<T>(), D};
    Output<Halide::Func> output1{ "output1", Halide::type_of<T>(), D};
    Output<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };

    void generate() {
        using namespace Halide;

        Func camera2("u3v_camera2");
        {
            Buffer<uint8_t> id_buf = this->get_id();

            const std::string gain_key(gain_key_ptr);
            Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
            gain_key_buf.fill(0);
            std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

            const std::string exposure_key(exposure_key_ptr);
            Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
            exposure_key_buf.fill(0);
            std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

            std::vector<ExternFuncArgument> params{
                static_cast<bool>(frame_sync), static_cast<bool>(realtime_display_mode),
                gain0, gain1, exposure0, exposure1,
                id_buf, gain_key_buf, exposure_key_buf
             };
            camera2.define_extern("ion_bb_image_io_u3v_camera2", params, { Halide::type_of<T>(), Halide::type_of<T>() }, D);
            camera2.compute_root();
            output0(_) = camera2(_)[0];
            output1(_) = camera2(_)[1];
        }

        Func camera2_frame_count;{
            Buffer<uint8_t> id_buf = this->get_id();
            camera2_frame_count.define_extern("ion_bb_image_io_u3v_camera2_frame_count", { camera2,  2, static_cast<bool>(frame_sync), static_cast<bool>(realtime_display_mode), id_buf}, type_of<uint32_t>(), 1);
            camera2_frame_count.compute_root();
            frame_count(_) = camera2_frame_count(_);
        }
        this->register_disposer("u3v_dispose");
    }
};

using U3VCamera2_U8x3 = U3VCamera2<uint8_t, 3>;
using U3VCamera2_U8x2 = U3VCamera2<uint8_t, 2>;
using U3VCamera2_U16x2 = U3VCamera2<uint16_t, 2>;

template<typename T, int D>
class U3VCameraN : public ion::BuildingBlock<U3VCameraN<T, D>> {
public:
    BuildingBlockParam<int32_t> num_devices{"num_devices", 2};
    BuildingBlockParam<bool> frame_sync{"frame_sync", false};
    BuildingBlockParam<bool> realtime_display_mode{"realtime_display_mode", false};

    BuildingBlockParam<bool> enable_control{"enable_control", false};
    BuildingBlockParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    BuildingBlockParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};

    Output<Halide::Func[]> output{ "output", Halide::type_of<T>(), D};
    Output<Halide::Func[]> device_info{ "device_info", Halide::type_of<uint8_t>(), 1};
    Output<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };

    std::vector<Input<double> *> gain;
    std::vector<Input<double> *> exposure;

    void configure() {
        if (enable_control) {
            for (auto i=0; i<num_devices; ++i) {
                gain.push_back(Halide::Internal::GeneratorBase::add_input<double>("gain_" + std::to_string(i)));
                exposure.push_back(Halide::Internal::GeneratorBase::add_input<double>("exposure_" + std::to_string(i)));
            }
        }
    }

    void generate() {
        using namespace Halide;

        Func cameraN("u3v_cameraN");
        {
            Buffer<uint8_t> id_buf = this->get_id();

            const std::string gain_key(gain_key_ptr);
            Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
            gain_key_buf.fill(0);
            std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

            const std::string exposure_key(exposure_key_ptr);
            Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
            exposure_key_buf.fill(0);
            std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

            std::vector<ExternFuncArgument> params{
                id_buf,
                static_cast<bool>(frame_sync),
                static_cast<bool>(realtime_display_mode),
                static_cast<bool>(enable_control),
                gain_key_buf, exposure_key_buf
            };

            for (int i = 0; i<num_devices; i++) {
                if (i < gain.size()) {
                    params.push_back(*gain[i]);
                } else {
                    params.push_back(Internal::make_const(type_of<double>(), 0.0));
                }
                if (i < exposure.size()) {
                    params.push_back(*exposure[i]);
                } else {
                    params.push_back(Internal::make_const(type_of<double>(), 0.0));
                }
            }

            output.resize(num_devices);
            cameraN.define_extern("ion_bb_image_io_u3v_multiple_camera" + std::to_string(num_devices), params, std::vector<Halide::Type>(num_devices, Halide::type_of<T>()), D);
            cameraN.compute_root();
            if (num_devices == 1){
                output[0](_) = cameraN(_);
            } else {
                for (int i = 0; i<num_devices; i++) {
                    output[i](_) = cameraN(_)[i];
                }
            }
        }

        Func u3v_device_info("u3v_device_info");
        {

            Buffer<uint8_t> id_buf = this->get_id();
            std::vector<ExternFuncArgument> params{
                cameraN, static_cast<int32_t>(num_devices), static_cast<bool>(frame_sync),
                static_cast<bool>(realtime_display_mode), id_buf
            };

            device_info.resize(num_devices);
            std::vector<Halide::Type> output_type;
            for (int i = 0; i < device_info.size(); i++) {
                output_type.push_back(Halide::type_of<uint8_t>());
            }
            u3v_device_info.define_extern("ion_bb_image_io_u3v_device_info" + std::to_string(device_info.size()), params, output_type, 1);
            u3v_device_info.compute_root();
            if (device_info.size() == 1){
                device_info[0](_) = u3v_device_info(_);
            }else{
                for (int i = 0; i < device_info.size(); i++) {
                    device_info[i](_) = u3v_device_info(_)[i];
                }
            }
        }

        Func cameraN_fc("u3v_cameraN_fc");
        {
            Buffer<uint8_t> id_buf = this->get_id();
            std::vector<ExternFuncArgument> params{
                cameraN, static_cast<int32_t>(output.size()), static_cast<bool>(frame_sync),
                static_cast<bool>(realtime_display_mode), id_buf
            };
            cameraN_fc.define_extern("ion_bb_image_io_u3v_multiple_camera_frame_count" + std::to_string(output.size()), params, type_of<uint32_t>(), 1);
            cameraN_fc.compute_root();
            frame_count(_) = cameraN_fc(_);
        }
        this->register_disposer("u3v_dispose");
    }

};

using U3VCameraN_U8x3 = U3VCameraN<uint8_t, 3>;
using U3VCameraN_U8x2 = U3VCameraN<uint8_t, 2>;
using U3VCameraN_U16x2 = U3VCameraN<uint16_t, 2>;

class U3VGenDC : public ion::BuildingBlock<U3VGenDC> {
public:
    BuildingBlockParam<int32_t> num_devices{"num_devices", 2};
    BuildingBlockParam<bool> frame_sync{"frame_sync", false};
    BuildingBlockParam<bool> realtime_display_mode{"realtime_display_mode", false};

    BuildingBlockParam<bool> enable_control{"enable_control", false};
    BuildingBlockParam<std::string> gain_key_ptr{"gain_key", "Gain"};
    BuildingBlockParam<std::string> exposure_key_ptr{"exposure_key", "Exposure"};

    Output<Halide::Func[]> gendc{ "gendc", Halide::type_of<uint8_t>(), 1};
    Output<Halide::Func[]> device_info{ "device_info", Halide::type_of<uint8_t>(), 1};

    std::vector<Input<double> *> gain;
    std::vector<Input<double> *> exposure;

    void configure() {
        if (enable_control) {
            for (auto i=0; i<num_devices; ++i) {
                gain.push_back(Halide::Internal::GeneratorBase::add_input<double>("gain_" + std::to_string(i)));
                exposure.push_back(Halide::Internal::GeneratorBase::add_input<double>("exposure_" + std::to_string(i)));
            }
        }
    }

    void generate() {
        using namespace Halide;

        Func u3v_gendc("u3v_gendc");
        {
            Buffer<uint8_t> id_buf =  this->get_id();

            const std::string gain_key(gain_key_ptr);
            Buffer<uint8_t> gain_key_buf(static_cast<int>(gain_key.size() + 1));
            gain_key_buf.fill(0);
            std::memcpy(gain_key_buf.data(), gain_key.c_str(), gain_key.size());

            const std::string exposure_key(exposure_key_ptr);
            Buffer<uint8_t> exposure_key_buf(static_cast<int>(exposure_key.size() + 1));
            exposure_key_buf.fill(0);
            std::memcpy(exposure_key_buf.data(), exposure_key.c_str(), exposure_key.size());

            std::vector<ExternFuncArgument> params{
                id_buf, 
                static_cast<bool>(frame_sync), 
                static_cast<bool>(realtime_display_mode),
                static_cast<bool>(enable_control),
                gain_key_buf, exposure_key_buf
            };

            for (int i = 0; i<num_devices; i++) {
                if (i < gain.size()) {
                    params.push_back(*gain[i]);
                } else {
                    params.push_back(Internal::make_const(type_of<double>(), 0.0));
                }
                if (i < exposure.size()) {
                    params.push_back(*exposure[i]);
                } else {
                    params.push_back(Internal::make_const(type_of<double>(), 0.0));
                }
            }

            gendc.resize(num_devices);
            std::vector<Halide::Type> output_type;
            for (int i = 0; i < gendc.size(); i++) {
                output_type.push_back(Halide::type_of<uint8_t>());
            }
            u3v_gendc.define_extern("ion_bb_image_io_u3v_gendc_camera" + std::to_string(gendc.size()), params, output_type, 1);
            u3v_gendc.compute_root();
            if (gendc.size() == 1){
                gendc[0](_) = u3v_gendc(_);
            }else{
                for (int i = 0; i < gendc.size(); i++) {
                    gendc[i](_) = u3v_gendc(_)[i];
                }
            }
        }

        Func u3v_device_info("u3v_device_info");
        {
            Buffer<uint8_t> id_buf =  this->get_id();
            std::vector<ExternFuncArgument> params{
                u3v_gendc, static_cast<int32_t>(num_devices), static_cast<bool>(frame_sync),
                static_cast<bool>(realtime_display_mode), id_buf
            };

            device_info.resize(num_devices);
            std::vector<Halide::Type> output_type;
            for (int i = 0; i < device_info.size(); i++) {
                output_type.push_back(Halide::type_of<uint8_t>());
            }
            u3v_device_info.define_extern("ion_bb_image_io_u3v_device_info" + std::to_string(device_info.size()), params, output_type, 1);
            u3v_device_info.compute_root();
            if (device_info.size() == 1){
                device_info[0](_) = u3v_device_info(_);
            }else{
                for (int i = 0; i < device_info.size(); i++) {
                    device_info[i](_) = u3v_device_info(_)[i];
                }
            }
        }

        this->register_disposer("u3v_dispose");
    }
};

template<typename T, int D>
class BinarySaver : public ion::BuildingBlock<BinarySaver<T, D>> {
public:
    BuildingBlockParam<std::string> output_directory_ptr{ "output_directory", "." };
    BuildingBlockParam<int32_t> num_devices{"num_devices", 2};

    Input<Halide::Func[]> input_images{ "input_images", Halide::type_of<T>(), D };

    Input<Halide::Func[]> input_deviceinfo{ "input_deviceinfo", Halide::type_of<uint8_t>(), 1 };
    Input<Halide::Func> frame_count{ "frame_count", Halide::type_of<uint32_t>(), 1 };


    Input<int32_t> width{ "width" };
    Input<int32_t> height{ "height" };

    Output<int32_t> output{"output"};

    void generate() {
        using namespace Halide;

        int32_t num_gendc = static_cast<int32_t>(num_devices);

        const std::string output_directory(output_directory_ptr);
        Halide::Buffer<uint8_t> output_directory_buf(static_cast<int>(output_directory.size() + 1));
        output_directory_buf.fill(0);
        std::memcpy(output_directory_buf.data(), output_directory.c_str(), output_directory.size());

        Func fc;
        fc(_) = frame_count(_);
        fc.compute_root();

        int32_t dim = D;
        int32_t byte_depth = sizeof(T);

         Buffer<uint8_t> id_buf = this->get_id();
        if (num_devices==1){
            Func image;
            image(_) = input_images(_);
            image.compute_root();

            Func deviceinfo;
            deviceinfo(_) = input_deviceinfo(_);
            deviceinfo.compute_root();

            std::vector<ExternFuncArgument> params = {id_buf, image, deviceinfo, fc, width, height, dim, byte_depth, output_directory_buf };
            Func ion_bb_image_io_binary_image_saver;
            ion_bb_image_io_binary_image_saver.define_extern("ion_bb_image_io_binary_1image_saver", params, Int(32), 0);
            ion_bb_image_io_binary_image_saver.compute_root();
            output() = ion_bb_image_io_binary_image_saver();
        }else if (num_devices==2){
            Func image0, image1;
            image0(_) = input_images[0](_);
            image1(_) = input_images[1](_);
            image0.compute_root();
            image1.compute_root();

            Func deviceinfo0, deviceinfo1;
            deviceinfo0(_) = input_deviceinfo[0](_);
            deviceinfo1(_) = input_deviceinfo[1](_);
            deviceinfo0.compute_root();
            deviceinfo1.compute_root();

            std::vector<ExternFuncArgument> params = {id_buf, image0, image1, deviceinfo0, deviceinfo1, fc, width, height, dim, byte_depth, output_directory_buf };
            Func ion_bb_image_io_binary_image_saver;
            ion_bb_image_io_binary_image_saver.define_extern("ion_bb_image_io_binary_2image_saver", params, Int(32), 0);
            ion_bb_image_io_binary_image_saver.compute_root();
            output() = ion_bb_image_io_binary_image_saver();
        }else{
            std::runtime_error("device number > 2 is not supported");
        }

        this->register_disposer("writer_dispose");
    }
};


using BinarySaver_U8x3 = BinarySaver<uint8_t, 3>;
using BinarySaver_U8x2 = BinarySaver<uint8_t, 2>;
using BinarySaver_U16x2 = BinarySaver<uint16_t, 2>;

class BinaryGenDCSaver : public ion::BuildingBlock<BinaryGenDCSaver> {
public:
    BuildingBlockParam<std::string> output_directory_ptr{ "output_directory", "." };

    BuildingBlockParam<int32_t> num_devices{"num_devices", 2};

    Input<Halide::Func[]> input_gendc{ "input_gendc", Halide::type_of<uint8_t>(), 1 };
    Input<Halide::Func[]> input_deviceinfo{ "input_deviceinfo", Halide::type_of<uint8_t>(), 1 };


    Input<int32_t> payloadsize{ "payloadsize" };

    Output<int> output{ "output" };

    void generate() {
        using namespace Halide;
        int32_t num_gendc = static_cast<int32_t>(num_devices);

        const std::string output_directory(output_directory_ptr);
        Halide::Buffer<uint8_t> output_directory_buf(static_cast<int>(output_directory.size() + 1));
        output_directory_buf.fill(0);
        std::memcpy(output_directory_buf.data(), output_directory.c_str(), output_directory.size());
        Buffer<uint8_t> id_buf = this->get_id();
        if (num_gendc==1){
            Func gendc;
            gendc(_) = input_gendc(_);
            gendc.compute_root();

            Func deviceinfo;
            deviceinfo(_) = input_deviceinfo(_);
            deviceinfo.compute_root();

            std::vector<ExternFuncArgument> params = { id_buf, gendc, deviceinfo, payloadsize, output_directory_buf };
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

            std::vector<ExternFuncArgument> params = { id_buf, gendc0, gendc1, deviceinfo0, deviceinfo1,  payloadsize, output_directory_buf };
            Func image_io_binary_gendc_saver;
            image_io_binary_gendc_saver.define_extern("ion_bb_image_io_binary_2gendc_saver", params, Int(32), 0);
            image_io_binary_gendc_saver.compute_root();
            output() = image_io_binary_gendc_saver();
        }else{
            std::runtime_error("device number > 2 is not supported");
        }
        this->register_disposer("writer_dispose");
    }
};

class BinaryLoader : public ion::BuildingBlock<BinaryLoader> {
public:
    BuildingBlockParam<std::string> output_directory_ptr{ "output_directory_ptr", "" };
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

#ifdef __linux__
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::IMX219, image_io_imx219);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::D435, image_io_d435);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GenericV4L2Bayer, image_io_generic_v4l2_bayer);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::CameraSimulation, image_io_camera_simulation);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::FBDisplay, image_io_fb_display);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::Camera, image_io_camera);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::Camera2, image_io_camera2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::CameraN, image_io_cameraN);
#endif


ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ColorDataLoader, image_io_color_data_loader);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GrayscaleDataLoader, image_io_grayscale_data_loader);

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

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinarySaver_U16x2, image_io_binarysaver);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinarySaver_U8x3, image_io_binarysaver_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinarySaver_U8x2, image_io_binarysaver_u8x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinarySaver_U16x2, image_io_binarysaver_u16x2);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinaryLoader, image_io_binaryloader);

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::BinaryGenDCSaver, image_io_binary_gendc_saver);

//backward compatability
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U8x3, u3v_camera1_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera1_U16x2, u3v_camera1_u16x2);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U8x3, u3v_camera2_u8x3);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::U3VCamera2_U16x2, u3v_camera2_u16x2);
#endif
