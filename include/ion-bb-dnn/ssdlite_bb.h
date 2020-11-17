#ifndef ION_BB_DNN_SSDLITE_BB_H
#define ION_BB_DNN_SSDLITE_BB_H

#include <fstream>

#include <ion/ion.h>
#include <uuid/uuid.h>

// For before uuid-v2014.01 undefined UUID_STR_LEN
#ifndef UUID_STR_LEN
#define UUID_STR_LEN 36
#endif

template<typename T>
static void generate_base(Halide::Func input,
                          T &model_ptr,
                          int32_t model_size,
                          std::string model_path_str,
                          int32_t batch_size,
                          int32_t height,
                          int32_t width,
                          int32_t channels,
                          bool can_cuda,
                          std::vector<ion::GeneratorOutput<Halide::Func> *> &outputs) {
    using namespace Halide;

    Func in;
    in(_) = input(_);
    in.compute_root();

    Func model;
    Halide::Expr model_size_expr;
    if (model_path_str.size() == 0) {
        model(_) = model_ptr(_);
        model_size_expr = model_size;
    } else {
        model(_) = 0;
        model_size_expr = cast<int32_t>(0);
    }
    model.compute_root();

    Buffer<uint8_t> model_path_buf(model_path_str.size() + 1);
    std::memcpy(model_path_buf.data(), model_path_str.c_str(), model_path_str.size() + 1);

    uuid_t session_id;
    uuid_generate(session_id);
    char session_id_chars[UUID_STR_LEN];
    uuid_unparse(session_id, session_id_chars);

    Buffer<uint8_t> session_id_buf(UUID_STR_LEN);
    std::memcpy(session_id_buf.data(), session_id_chars, UUID_STR_LEN);

    std::vector<ExternFuncArgument> params{in, session_id_buf, model, model_size_expr, model_path_buf, cast<int32_t>(channels), cast<int32_t>(width), cast<int32_t>(height), cast<int32_t>(batch_size), cast<bool>(can_cuda)};
    std::vector<Halide::Type> types(outputs.size(), Float(32));

    Func onnx_inference;
    onnx_inference.define_extern("onnx_inference_out" + std::to_string(outputs.size()), params, types, 1);
    onnx_inference.compute_root();

    if (outputs.size() == 1) {
        (*outputs[0])(_) = onnx_inference(_);
    } else {
        for (std::size_t i = 0; i < outputs.size(); i++) {
            (*outputs[i])(_) = onnx_inference(_)[i];
        }
    }
}

//
// NOTE: Defines model specialized BB and disables base BBs ATM
//

// template<typename T>
// class ONNXInferenceOut1 : public ion::BuildingBlock<ONNXInferenceOut1<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::GeneratorParam<std::string> model_path{"model_path", ""};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0};
//         std::string model_path_str{model_path};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, nullptr, nullptr, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceOut2 : public ion::BuildingBlock<ONNXInferenceOut2<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::GeneratorParam<std::string> model_path{"model_path", ""};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0, &output1};
//         std::string model_path_str{model_path};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, nullptr, nullptr, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceOut3 : public ion::BuildingBlock<ONNXInferenceOut3<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::GeneratorParam<std::string> model_path{"model_path", ""};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output2{"output2", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0, &output1, &output2};
//         std::string model_path_str{model_path};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, nullptr, nullptr, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceOut4 : public ion::BuildingBlock<ONNXInferenceOut4<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::GeneratorParam<std::string> model_path{"model_path", ""};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output2{"output2", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output3{"output3", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0, &output1, &output2, &output3};
//         std::string model_path_str{model_path};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, nullptr, nullptr, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceBufferOut1 : public ion::BuildingBlock<ONNXInferenceBufferOut1<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::Input<Halide::Func> model{"model", Halide::UInt(8), 1};
//     ion::Input<int32_t> model_size{"model_size", 0};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0};
//         std::string model_path_str{""};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, &model, &model_size, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceBufferOut2 : public ion::BuildingBlock<ONNXInferenceBufferOut2<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::Input<Halide::Func> model{"model", Halide::UInt(8), 1};
//     ion::Input<int32_t> model_size{"model_size", 0};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0, &output1};
//         std::string model_path_str{""};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, &model, &model_size, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceBufferOut3 : public ion::BuildingBlock<ONNXInferenceBufferOut3<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::Input<Halide::Func> model{"model", Halide::UInt(8), 1};
//     ion::Input<int32_t> model_size{"model_size", 0};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output2{"output2", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0, &output1, &output2};
//         std::string model_path_str{""};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, &model, &model_size, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };
//
// template<typename T>
// class ONNXInferenceBufferOut4 : public ion::BuildingBlock<ONNXInferenceBufferOut4<T>> {
// public:
//     ion::Input<Halide::Func> input{"input", Halide::type_of<T>(), 1};
//     ion::Input<Halide::Func> model{"model", Halide::UInt(8), 1};
//     ion::Input<int32_t> model_size{"model_size", 0};
//     ion::GeneratorParam<int32_t> dim0{"dim0", -1};
//     ion::GeneratorParam<int32_t> dim1{"dim1", -1};
//     ion::GeneratorParam<int32_t> dim2{"dim2", -1};
//     ion::GeneratorParam<int32_t> dim3{"dim3", -1};
//     ion::Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output2{"output2", Halide::Float(32), 1};
//     ion::Output<Halide::Func> output3{"output3", Halide::Float(32), 1};
//
//     void generate() {
//         std::vector<ion::Output<Halide::Func>*> outputs{&output0, &output1, &output2, &output3};
//         std::string model_path_str{""};
//         bool can_cuda = this->get_target().has_feature(Halide::Target::Feature::CUDA);
//         generate_base(input, &model, &model_size, model_path_str, dim0, dim1, dim2, dim3, can_cuda, outputs);
//     }
//
//     void schedule() {
//     }
//
// private:
// };

//
// using ONNXInferenceInUInt8Out1 = ONNXInferenceOut1<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInUInt8Out1, onnx_inference_by_filepath_in_uint8_out_1);
//
// using ONNXInferenceInFloatOut1 = ONNXInferenceOut1<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInFloatOut1, onnx_inference_by_filepath_in_float_out_1);
//
// using ONNXInferenceInUInt8Out2 = ONNXInferenceOut2<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInUInt8Out2, onnx_inference_by_filepath_in_uint8_out_2);
//
// using ONNXInferenceInFloatOut2 = ONNXInferenceOut2<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInFloatOut2, onnx_inference_by_filepath_in_float_out_2);
//
// using ONNXInferenceInUInt8Out3 = ONNXInferenceOut3<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInUInt8Out3, onnx_inference_by_filepath_in_uint8_out_3);
//
// using ONNXInferenceInFloatOut3 = ONNXInferenceOut3<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInFloatOut3, onnx_inference_by_filepath_in_float_out_3);
//
// using ONNXInferenceInUInt8Out4 = ONNXInferenceOut4<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInUInt8Out4, onnx_inference_by_filepath_in_uint8_out_4);
//
// using ONNXInferenceInFloatOut4 = ONNXInferenceOut4<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceInFloatOut4, onnx_inference_by_filepath_in_float_out_4);
//
// using ONNXInferenceBufferInUInt8Out1 = ONNXInferenceBufferOut1<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInUInt8Out1, onnx_inference_by_buffer_in_uint8_out_1);
//
// using ONNXInferenceBufferInFloatOut1 = ONNXInferenceBufferOut1<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInFloatOut1, onnx_inference_by_buffer_in_float_out_1);
//
// using ONNXInferenceBufferInUInt8Out2 = ONNXInferenceBufferOut2<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInUInt8Out2, onnx_inference_by_buffer_in_uint8_out_2);
//
// using ONNXInferenceBufferInFloatOut2 = ONNXInferenceBufferOut2<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInFloatOut2, onnx_inference_by_buffer_in_float_out_2);
//
// using ONNXInferenceBufferInUInt8Out3 = ONNXInferenceBufferOut3<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInUInt8Out3, onnx_inference_by_buffer_in_uint8_out_3);
//
// using ONNXInferenceBufferInFloatOut3 = ONNXInferenceBufferOut3<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInFloatOut3, onnx_inference_by_buffer_in_float_out_3);
//
// using ONNXInferenceBufferInUInt8Out4 = ONNXInferenceBufferOut4<uint8_t>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInUInt8Out4, onnx_inference_by_buffer_in_uint8_out_4);
//
// using ONNXInferenceBufferInFloatOut4 = ONNXInferenceBufferOut4<float>;
// ION_REGISTER_BUILDING_BLOCK(ONNXInferenceBufferInFloatOut4, onnx_inference_by_buffer_in_float_out_4);

namespace {

class SSDObjectDetection : public ion::BuildingBlock<SSDObjectDetection> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Object Detection"};
    GeneratorParam<std::string> gc_description{"gc_description", "Detect object."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,recognition"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output0: [400], output1:[100] , output2:[100] , output3:[100] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<int32_t> width_{"width", -1};
    GeneratorParam<int32_t> height_{"height", -1};

    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};

    Output<Halide::Func> output0{"output0", Halide::Float(32), 1};
    Output<Halide::Func> output1{"output1", Halide::Float(32), 1};
    Output<Halide::Func> output2{"output2", Halide::Float(32), 1};
    Output<Halide::Func> output3{"output3", Halide::Float(32), 1};

    void generate() {
        using namespace Halide;

        std::vector<ion::GeneratorOutput<Halide::Func> *> outputs{&output0, &output1, &output2, &output3};

        bool can_cuda = this->get_target().has_feature(Target::Feature::CUDA);

        const char *models_root = getenv("ION_BB_DNN_MODELS_ROOT");
        if (models_root == nullptr) {
            throw std::runtime_error("Set appropriate ION_BB_DNN_MODELS_ROOT");
        }
        Buffer<uint8_t> model;
        std::ifstream ifs(std::string(models_root) + "/ssdlite_mobilenet_v2_coco_2018_05_09.onnx", std::ifstream::ate | std::ifstream::binary);
        model = Buffer<uint8_t>(static_cast<int>(ifs.tellg()));
        ifs.clear();
        ifs.seekg(ifs.beg);
        ifs.read(reinterpret_cast<char *>(model.data()), model.size_in_bytes());

        // 1. change RGB => BGR
        // 2. add N (change HWC => NHWC)
        Func layouted_image;
        Var x, y, c, n;
        layouted_image(c, x, y, n) = select(c == 0, input(2, x, y),
                                            c == 1, input(1, x, y),
                                            input(0, x, y));
        int batch_size = 1;
        int height = static_cast<int>(height_);
        int width = static_cast<int>(width_);
        int channels = 3;

        generate_base(layouted_image, model, static_cast<int32_t>(model.size_in_bytes()), std::string(""), batch_size, height, width, channels, can_cuda, outputs);
    }
};

class SSDBoundingBoxRenderer : public ion::BuildingBlock<SSDBoundingBoxRenderer> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Bounding Box Renderer"};
    GeneratorParam<std::string> gc_description{"gc_description", "Render bounding box."};
    GeneratorParam<std::string> gc_tags{"gc_tags", "processing,imgproc"};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [3, parseInt(v.width), parseInt(v.height)] }}))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", "width,height"};
    GeneratorParam<int32_t> width_{"width", -1};
    GeneratorParam<int32_t> height_{"height", -1};

    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    Input<Halide::Func> boxes{"boxes", Halide::Float(32), 1};
    Input<Halide::Func> classes{"classes", Halide::Float(32), 1};
    Input<Halide::Func> scores{"scores", Halide::Float(32), 1};
    Input<Halide::Func> nums{"nums", Halide::Float(32), 1};

    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 3};

    void generate() {
        using namespace Halide;

        Func input_;
        input_(_) = input(_);
        input_.compute_root();

        Func boxes_;
        boxes_(_) = boxes(_);
        boxes_.compute_root();

        Func classes_;
        classes_(_) = classes(_);
        classes_.compute_root();

        Func scores_;
        scores_(_) = scores(_);
        scores_.compute_root();

        Func nums_;
        nums_(_) = nums(_);
        nums_.compute_root();

        std::vector<ExternFuncArgument> params = {input_, boxes_, classes_, scores_, nums_, static_cast<int32_t>(width_), static_cast<int32_t>(height_)};
        Func renderer("boundinx_box_renderer");
        renderer.define_extern("bounding_box_renderer", params, UInt(8), 3);
        renderer.compute_root();
        output(_) = renderer(_);
    }
};

}  // namespace

ION_REGISTER_BUILDING_BLOCK(SSDObjectDetection, dnn_ssd_object_detection);
ION_REGISTER_BUILDING_BLOCK(SSDBoundingBoxRenderer, dnn_ssd_bounding_box_renderer);

#endif  // ION_BB_DNN_BB_H
