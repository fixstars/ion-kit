#ifndef ION_BB_IMAGE_IO_BB_H
#define ION_BB_IMAGE_IO_BB_H

#include <ion/ion.h>
#include <linux/videodev2.h>

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

int instance_id = 0;

class IMX219 : public ion::BuildingBlock<IMX219> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "IMX219"};
    GeneratorParam<std::string> gc_description{"gc_description", "This captures IMX219 image."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "input,sensor"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3264, 2464] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> index{"index", 0};
    GeneratorParam<bool> force_sim_mode{"force_sim_mode", false};
    GeneratorParam<int32_t> fps{"fps", 24};
    GeneratorParam<int32_t> width{"width", 3264};
    GeneratorParam<int32_t> height{"height", 2464};
    GeneratorParam<std::string> url{"url", ""};
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
            Expr(V4L2_PIX_FMT_SRGGB10),
            cast<bool>(force_sim_mode),
            url_buf,
            0.4f, 0.5f, 0.3125f,
            0.0625f,
            10, 6,
            0 /*RGGB*/
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

    GeneratorParam<int32_t> index{"index", 0};
    GeneratorParam<int32_t> fps{"fps", 30};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
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
            internal_error << "Unknown V4L2 format";
        }

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
            Expr(pix_format),
            Expr(false),
            url_buf,
            1.f, 1.f, 1.f,
            0.f,
            cast<int32_t>(bit_width), 16 - bit_width,
            static_cast<int32_t>(static_cast<BayerMap::Pattern>(bayer_pattern))};
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

    GeneratorParam<std::string> url{"url", ""};
    GeneratorParam<int32_t> fps{"fps", 30};
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorParam<int32_t> bit_width{"bit_width", 10};
    GeneratorParam<int32_t> bit_shift{"bit_shift", 0};
    GeneratorParam<BayerMap::Pattern> bayer_pattern{"bayer_pattern", BayerMap::Pattern::RGGB, BayerMap::enum_map};
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
            0,
            Expr(true),
            url_buf,
            cast<float>(gain_r), cast<float>(gain_g), cast<float>(gain_b),
            cast<float>(offset),
            cast<int32_t>(bit_width), cast<int32_t>(bit_shift),
            static_cast<int32_t>(static_cast<BayerMap::Pattern>(bayer_pattern))};
        Func camera(static_cast<std::string>(gc_prefix) + "output");
        camera.define_extern("ion_bb_image_io_v4l2", params, type_of<uint16_t>(), 2);
        camera.compute_root();

        output = camera;
    }
};

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
    GeneratorOutput<Halide::Func> output{"output"};

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
    GeneratorOutput<Halide::Func> output{"output"};

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

class ImageLoader : public ion::BuildingBlock<ImageLoader> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Image Loader"};
    GeneratorParam<std::string> gc_description{"gc_description", "This loads image from specified URL."};
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
        std::string url_str(url);
        Halide::Buffer<uint8_t> url_buf(url_str.size() + 1);
        url_buf.fill(0);
        std::memcpy(url_buf.data(), url_str.c_str(), url_str.size());
        std::vector<ExternFuncArgument> params = {url_buf};
        Func image_loader(static_cast<std::string>(gc_prefix) + "image_loader");
        image_loader.define_extern("ion_bb_image_io_image_loader", params, Halide::type_of<uint8_t>(), 3);
        image_loader.compute_root();
        Var c, x, y;

        Func f(static_cast<std::string>(gc_prefix) + "output");
        f(x, y, c) = mux(c,
                         {image_loader(2, x, y),
                         image_loader(1, x, y),
                         image_loader(0, x, y)});
        output = f;
    }
};

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
    GeneratorOutput<Halide::Func> output{"output"};

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

}  // namespace image_io
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::IMX219, image_io_imx219);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::D435, image_io_d435);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::Camera, image_io_camera);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GenericV4L2Bayer, image_io_generic_v4l2_bayer);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::CameraSimulation, image_io_camera_simulation);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GUIDisplay, image_io_gui_display);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::FBDisplay, image_io_fb_display);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ImageLoader, image_io_image_loader);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ImageSaver, image_io_image_saver);

#endif
