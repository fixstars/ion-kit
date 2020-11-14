#ifndef ION_BB_DNN_BB_H
#define ION_BB_DNN_BB_H

#include <ion/ion.h>
#include <uuid/uuid.h>

namespace ion {
namespace bb {
namespace dnn {

template<typename T>
class ReorderHWC2CHW : public BuildingBlock<ReorderHWC2CHW<T>> {
public:
    constexpr static const int dim = 3;
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), dim};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), dim};

    void generate() {
        output(x, y, c) = input(c, x, y);
    }

private:
    Halide::Var x, y, c;
};

template<typename T>
class ReorderCHW2HWC : public BuildingBlock<ReorderCHW2HWC<T>> {
public:
    constexpr static const int dim = 3;
    GeneratorInput<Halide::Func> input{"input", Halide::type_of<T>(), dim};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<T>(), dim};

    void generate() {
        output(c, x, y) = input(x, y, c);
    }

private:
    Halide::Var c, x, y;
};

template<typename X, int32_t D>
class ObjectDetectionBase : public BuildingBlock<X> {
    static_assert(D == 3 || D == 4, "D must be 3 or 4.");

public:
    GeneratorParam<std::string> gc_description{"gc_description", "Detect objects by various DNN models."};
    GeneratorParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { outputs: inputs }))"};
    GeneratorParam<std::string> gc_mandatory{"gc_mandatory", ""};
    GeneratorParam<std::string> gc_strategy{"gc_strategy", "self"};
    GeneratorParam<std::string> gc_prefix{"gc_prefix", ""};

    GeneratorParam<std::string> model_root_url_{"model_base_url", "http://ion-archives.s3-us-west-2.amazonaws.com/models/"};
    GeneratorParam<std::string> cache_root_{"cache_root", "/var/cache/"};

    // TODO: Embed model at compilation time
    // GeneratorParam<bool> embed_model{"embed_model", false};

    GeneratorInput<Halide::Func> input_{"input", Halide::type_of<float>(), D};
    GeneratorOutput<Halide::Func> output{"output", Halide::type_of<float>(), D};

    void generate() {
        using namespace Halide;

        uuid_t session_id;
        uuid_generate(session_id);
        char session_id_chars[UUID_STR_LEN];
        uuid_unparse(session_id, session_id_chars);
        Buffer<uint8_t> session_id_buf(UUID_STR_LEN);
        std::memcpy(session_id_buf.data(), session_id_chars, UUID_STR_LEN);

        const std::string model_root_url(model_root_url_);
        Halide::Buffer<uint8_t> model_root_url_buf(model_root_url.size() + 1);
        model_root_url_buf.fill(0);
        std::memcpy(model_root_url_buf.data(), model_root_url.c_str(), model_root_url.size());

        const std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        const bool cuda_enable = this->get_target().has_feature(Target::Feature::CUDA);

        input = Func{static_cast<std::string>(gc_prefix) + "in"};
        input(_) = input_(_);

        std::vector<ExternFuncArgument> params{input, session_id_buf, model_root_url_buf, cache_path_buf, cuda_enable};
        Func object_detection(static_cast<std::string>(gc_prefix) + "object_detection");
        object_detection.define_extern("ion_bb_dnn_generic_object_detection", params, Float(32), D);
        object_detection.compute_root();

        output(_) = object_detection(_);
    }

    void schedule() {
        using namespace Halide;
        Var c = input.args()[0];
        Var x = input.args()[1];
        Var y = input.args()[2];

        input.bound(c, 0, 3).unroll(c);

        if (this->get_target().has_gpu_feature()) {
            Var xi, yi;
            input.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            input.vectorize(x, this->natural_vector_size(Float(32))).parallel(y, 16);
        }
        input.compute_root();
    }

private:
    Halide::Func input;
};

class ObjectDetection : public ObjectDetectionBase<ObjectDetection, 3> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Object Detection"};
};

class ObjectDetectionArray : public ObjectDetectionBase<ObjectDetectionArray, 4> {
public:
    GeneratorParam<std::string> gc_title{"gc_title", "Object Detection (Array)"};
};

} // dnn
} // bb
} // ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ReorderCHW2HWC<uint8_t>, dnn_reorder_chw2hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ReorderHWC2CHW<uint8_t>, dnn_reorder_hwc2chw);
ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ObjectDetection, dnn_object_detection);
ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ObjectDetectionArray, dnn_object_detection_array);

#endif  // ION_BB_DNN_BB_H
