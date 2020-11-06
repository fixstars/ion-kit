#ifndef ION_BB_DNN_YOLOV4_BB_H
#define ION_BB_DNN_YOLOV4_BB_H

#include <fstream>

#include <ion/ion.h>
#include <uuid/uuid.h>

// For before uuid-v2014.01 undefined UUID_STR_LEN
#ifndef UUID_STR_LEN
#define UUID_STR_LEN 36
#endif

namespace {

using namespace ion;
using Halide::_;

template<typename T>
class SwitchColorMode : public BuildingBlock<SwitchColorMode<T>> {
public:
    constexpr static const int dim = 3;
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), dim};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), dim};

    void generate() {
        output(c, x, y) = select(c == 0, input(2, x, y),
                                 c == 1, input(1, x, y),
                                 input(0, x, y));
    }

    void schedule() {
        output.compute_root();
    }

private:
    Halide::Var x, y, c;
};
template<typename T>
using RGB2BGR = SwitchColorMode<T>;
template<typename T>
using BGR2RGB = SwitchColorMode<T>;

template<typename T>
class ReorderHWC2CHW : public BuildingBlock<ReorderHWC2CHW<T>> {
public:
    constexpr static const int dim = 3;
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), dim};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), dim};

    void generate() {
        output(x, y, c) = input(c, x, y);
    }

    void schedule() {
        output.compute_root().reorder_storage(x, y, c);
    }

private:
    Halide::Var x, y, c;
};

template<typename To, typename From, int D>
class Devide255 : public BuildingBlock<Devide255<To, From, D>> {
public:
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<From>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<To>(), D};

    void generate() {
        assert(input.args().size() == D);
        output(_) = Halide::cast<To>(input(_)) / static_cast<To>(255);
    }

    void schedule() {
        output.compute_root();
    }
};
using Devide255FromU8ToF32 = Devide255<float, uint8_t, 3>;

template<typename T, int D>
class SplitOutput : public BuildingBlock<SplitOutput<T, D>> {
public:
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), D};
    GeneratorOutput<Halide::Func> output2{"output2", Halide::type_of<T>(), D};

    void generate() {
        output(_) = input(_);
        output2(_) = input(_);
    }

    void
    schedule() {
        output.compute_root();
        output2.compute_root();
    }
};
using SplitU8 = SplitOutput<uint8_t, 3>;
using SplitF32 = SplitOutput<float, 3>;

template<typename X, int32_t D>
class YOLOv4ObjectDetectionBase : public BuildingBlock<X> {
    static_assert(D == 3 || D == 4, "D must be 3 or 4.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Detect objects by YOLOv4."};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<std::string> model_root_{"model_root", "resource/dnn/model"};
    GeneratorParam<std::string> model_name_{"model_name", "/yolov4-tiny_416_416.onnx"};
    GeneratorParam<std::string> cache_root_{"cache_root", "/tmp"};
    GeneratorParam<int32_t> width_{"width", 416};
    GeneratorParam<int32_t> height_{"height", 416};
    GeneratorInput<Halide::Func> input_{"input", Halide::type_of<float>(), D};
    GeneratorOutput<Halide::Func> boxes{"boxes", Halide::Float(32), D - 1};
    GeneratorOutput<Halide::Func> confs{"confs", Halide::Float(32), D - 1};

    void generate() {
        using namespace Halide;

        const std::string model_root(model_root_);
        const std::string model_name(model_name_);

        char *model_root_from_env = getenv("ION_BB_DNN_MODELS_ROOT");
        std::string model_path;
        if (model_root_from_env != nullptr) {
            model_path = model_root_from_env + model_name;
        } else {
            model_path = model_root_ + model_name;
        }

        std::ifstream ifs(model_path, std::ifstream::ate | std::ifstream::binary);
        if (!ifs) {
            std::cout << model_path << " not found..." << std::endl;
            return;
        }

        Halide::Buffer<uint8_t> model_buf;
        model_buf = Buffer<uint8_t>(static_cast<int>(ifs.tellg()));
        const int32_t model_size_in_bytes = model_buf.size_in_bytes();
        ifs.clear();
        ifs.seekg(ifs.beg);
        ifs.read(reinterpret_cast<char *>(model_buf.data()), model_size_in_bytes);

        input = Func(static_cast<std::string>(gc_prefix) + "yolov4_input");

        input(_) = input_(_);

        uuid_t session_id;
        uuid_generate(session_id);
        char session_id_chars[UUID_STR_LEN];
        uuid_unparse(session_id, session_id_chars);

        Buffer<uint8_t> session_id_buf(UUID_STR_LEN);
        std::memcpy(session_id_buf.data(), session_id_chars, UUID_STR_LEN);

        std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        const int32_t height = height_;
        const int32_t width = width_;
        const bool cuda_enable = this->get_target().has_feature(Target::Feature::CUDA);

        std::vector<ExternFuncArgument> params{input, session_id_buf, model_buf, cache_path_buf, height, width, cuda_enable};

        Func yolov4_object_detection(static_cast<std::string>(gc_prefix) + "yolov4_object_detection");
        yolov4_object_detection.define_extern("yolov4_object_detection", params, {Float(32), Float(32)}, D - 1);
        yolov4_object_detection.compute_root();

        boxes(_) = yolov4_object_detection(_)[0];
        confs(_) = yolov4_object_detection(_)[1];
    }

    void schedule() {
        Halide::Var x = input.args()[0];
        Halide::Var y = input.args()[1];
        Halide::Var c = input.args()[2];

        input.reorder(c, x, y).bound(c, 0, 3).unroll(c);

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            input.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            input.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }
        input.compute_root();
    }

private:
    Halide::Func input;
};

class YOLOv4ObjectDetection : public YOLOv4ObjectDetectionBase<YOLOv4ObjectDetection, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "YOLOv4 Object Detection"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { boxes: [4, 2535], confs: [80, 2535] }}))"};
};

class YOLOv4ObjectDetectionArray : public YOLOv4ObjectDetectionBase<YOLOv4ObjectDetectionArray, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "YOLOv4 Object Detection(Array)"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { boxes: [4, 2535, v.input[3]], confs: [80, 2535, v.input[3]] }}))"};
};

template<typename X, int32_t D>
class YOLOv4BoxRenderingBase : public BuildingBlock<X> {
    static_assert(D == 3 || D == 4, "D must be 3 or 4.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Render bounding boxes from YOLOv4ObjectDetection."};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.image }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "height,width"};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<int32_t> height_{"height", 512};
    GeneratorParam<int32_t> width_{"width", 512};
    GeneratorParam<int32_t> num_{"num", 2535};
    GeneratorParam<int32_t> num_classes_{"num_classes", 80};

    GeneratorInput<Halide::Func> image_{"image", Halide::type_of<uint8_t>(), D};
    GeneratorInput<Halide::Func> boxes_{"boxes", Halide::type_of<float>(), D - 1};
    GeneratorInput<Halide::Func> confs_{"confs", Halide::type_of<float>(), D - 1};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<uint8_t>(), D};

    void generate() {
        using namespace Halide;

        image = Func(static_cast<std::string>(gc_prefix) + "yolov4_image_in");
        boxes = Func(static_cast<std::string>(gc_prefix) + "yolov4_boxes_in");
        confs = Func(static_cast<std::string>(gc_prefix) + "yolov4_confs_in");

        image(_) = image_(_);
        boxes(_) = boxes_(_);
        confs(_) = confs_(_);

        const int32_t height = height_;
        const int32_t width = width_;
        const int32_t num = num_;
        const int32_t num_classes = num_classes_;

        std::vector<ExternFuncArgument> params{image, boxes, confs, height, width, num, num_classes};

        Func yolov4_box_rendering(static_cast<std::string>(gc_prefix) + "yolov4_box_rendering");
        yolov4_box_rendering.define_extern("yolov4_box_rendering", params, {Halide::type_of<uint8_t>()}, D);
        yolov4_box_rendering.compute_root();

        output(_) = yolov4_box_rendering(_);
    }

    void schedule() {
        Halide::Var x = image.args()[1];
        Halide::Var y = image.args()[2];
        Halide::Var c = image.args()[0];

        image.bound(c, 0, 3).unroll(c);

        if (this->get_target().has_gpu_feature()) {
            Halide::Var xi, yi;
            image.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            image.vectorize(x, this->natural_vector_size(Halide::Float(32))).parallel(y, 16);
        }
        image.compute_root();

        boxes.vectorize(boxes.args()[0], this->natural_vector_size(Halide::Float(32)));
        confs.vectorize(confs.args()[0], this->natural_vector_size(Halide::Float(32)));

        boxes.compute_root();
        confs.compute_root();
    }

private:
    Halide::Func image, boxes, confs;
};

class YOLOv4BoxRendering : public YOLOv4BoxRenderingBase<YOLOv4BoxRendering, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "YOLOv4 Box Rendering"};
};

class YOLOv4BoxRenderingArray : public YOLOv4BoxRenderingBase<YOLOv4BoxRenderingArray, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "YOLOv4 Box Rendering(Array)"};
};

}  // namespace

// ION_REGISTER_BUILDING_BLOCK(SplitU8, yolov4_split_u8);
// ION_REGISTER_BUILDING_BLOCK(SplitF32, yolov4_split_f32);
// ION_REGISTER_BUILDING_BLOCK(ReorderHWC2CHW<uint8_t>, yolov4_reorder_hwc2chw);
// ION_REGISTER_BUILDING_BLOCK(RGB2BGR<uint8_t>, yolov4_rgb2bgr);
// ION_REGISTER_BUILDING_BLOCK(BGR2RGB<uint8_t>, yolov4_bgr2rgb);
// ION_REGISTER_BUILDING_BLOCK(Devide255FromU8ToF32, yolov4_devide255);
ION_REGISTER_BUILDING_BLOCK(YOLOv4ObjectDetection, yolov4_object_detection);
ION_REGISTER_BUILDING_BLOCK(YOLOv4BoxRendering, yolov4_box_rendering);
ION_REGISTER_BUILDING_BLOCK(YOLOv4ObjectDetectionArray, yolov4_object_detection_array);
ION_REGISTER_BUILDING_BLOCK(YOLOv4BoxRenderingArray, yolov4_box_rendering_array);

#endif  // ION_BB_DNN_BB_H
