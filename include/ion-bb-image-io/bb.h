#ifndef ION_BB_IMAGE_IO_BB_H
#define ION_BB_IMAGE_IO_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace image_io {

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
        v4l2_imx219.define_extern("ion_bb_image_io_v4l2_imx219", params, type_of<uint16_t>(), 2);
        v4l2_imx219.compute_root();

        Var x, y;
        output(x, y) = v4l2_imx219(x, y);
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
        Func realsense_d435_frameset(static_cast<std::string>(gc_prefix) + "realsense_d435_frameset");
        realsense_d435_frameset.define_extern("ion_bb_image_io_realsense_d435_frameset", {}, type_of<uint64_t>(), 0);
        realsense_d435_frameset.compute_root();

        Func realsense_d435_infrared(static_cast<std::string>(gc_prefix) + "realsense_d435_infrared");
        realsense_d435_infrared.define_extern("ion_bb_image_io_realsense_d435_infrared", {realsense_d435_frameset}, {type_of<uint8_t>(), type_of<uint8_t>()}, 2);
        realsense_d435_infrared.compute_root();

        Func realsense_d435_depth(static_cast<std::string>(gc_prefix) + "realsense_d435_depth");
        realsense_d435_depth.define_extern("ion_bb_image_io_realsense_d435_depth", {realsense_d435_frameset}, type_of<uint16_t>(), 2);
        realsense_d435_depth.compute_root();

        output_l(_) = realsense_d435_infrared(_)[0];
        output_r(_) = realsense_d435_infrared(_)[1];
        output_d(_) = realsense_d435_depth(_);
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
    GeneratorParam<int32_t> width{"width", 0};
    GeneratorParam<int32_t> height{"height", 0};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        using namespace Halide;
        std::vector<ExternFuncArgument> params = {cast<int32_t>(index), cast<int32_t>(width), cast<int32_t>(height)};
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

        output(x, y, c) = select(c == 0, r, c == 1, g, b);
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
        display.define_extern("ion_bb_image_io_gui_display", params, Int(32), 0);
        display.compute_root();
        output() = display();
    }

    void schedule() {
    }

private:
    Halide::Var c, x, y;
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
    GeneratorOutput<int32_t> output{"output"};

    void generate() {
        using namespace Halide;

        Func input_(static_cast<std::string>(gc_prefix) + "input");
        input_(_) = input(_);
        input_.compute_root();

        std::vector<ExternFuncArgument> params = {cast<int32_t>(width), cast<int32_t>(height), input_};

        Func display(static_cast<std::string>(gc_prefix) + "display");
        display.define_extern("ion_bb_image_io_fb_display", params, Halide::type_of<int32_t>(), 0);
        display.compute_root();
        output() = display();
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
        output(x, y, c) = select(c == 0, image_loader(2, x, y),
                                 c == 1, image_loader(1, x, y),
                                 image_loader(0, x, y));
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
    GeneratorOutput<int32_t> output{"output"};

    void generate() {
        using namespace Halide;
        std::string path_str(path);
        Halide::Buffer<uint8_t> path_buf(path_str.size() + 1);
        path_buf.fill(0);
        std::memcpy(path_buf.data(), path_str.c_str(), path_str.size());

        Func input_(static_cast<std::string>(gc_prefix) + "input");
        Var c, x, y;
        input_(c, x, y) = select(c == 0, input(x, y, 2),
                                 c == 1, input(x, y, 1),
                                 input(x, y, 0));
        input_.compute_root();

        std::vector<ExternFuncArgument> params = {input_, static_cast<int32_t>(width), static_cast<int32_t>(height), path_buf};
        Func image_saver(static_cast<std::string>(gc_prefix) + "image_saver");
        image_saver.define_extern("ion_bb_image_io_image_saver", params, Int(32), 0);
        image_saver.compute_root();
        output() = image_saver();
    }
};

}  // namespace image_io
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::IMX219, image_io_imx219);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::D435, image_io_d435);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::Camera, image_io_camera);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::GUIDisplay, image_io_gui_display);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::FBDisplay, image_io_fb_display);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ImageLoader, image_io_image_loader);
ION_REGISTER_BUILDING_BLOCK(ion::bb::image_io::ImageSaver, image_io_image_saver);

#endif
