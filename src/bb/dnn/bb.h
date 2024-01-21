#ifndef ION_BB_DNN_BB_H
#define ION_BB_DNN_BB_H

// For before uuid-v2014.01 undefined UUID_STR_LEN
#ifndef UUID_STR_LEN
#define UUID_STR_LEN 36
#endif

#include <ion/ion.h>

#include "uuid/sole.hpp"

namespace ion {
namespace bb {
namespace dnn {

template<typename T>
class ReorderHWC2CHW : public BuildingBlock<ReorderHWC2CHW<T>> {
public:
    constexpr static const int dim = 3;

    Input<Halide::Func> input{"input", Halide::type_of<T>(), dim};
    Output<Halide::Func> output{"output", Halide::type_of<T>(), dim};

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
    Input<Halide::Func> input{"input", Halide::type_of<T>(), dim};
    Output<Halide::Func> output{"output", Halide::type_of<T>(), dim};

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
    BuildingBlockParam<std::string> gc_description{"gc_description", "Detect objects by various DNN models."};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "processing,recognition"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<std::string> model_root_url_{"model_base_url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/"};
    BuildingBlockParam<std::string> cache_root_{"cache_root", "/tmp/"};

    // TODO: Embed model at compilation time
    // BuildingBlockParam<bool> embed_model{"embed_model", false};

    Input<Halide::Func> input_{"input", Halide::type_of<float>(), D};
    Output<Halide::Func> output{"output", Halide::type_of<float>(), D};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string model_root_url(model_root_url_);
        Halide::Buffer<uint8_t> model_root_url_buf(model_root_url.size() + 1);
        model_root_url_buf.fill(0);
        std::memcpy(model_root_url_buf.data(), model_root_url.c_str(), model_root_url.size());

        const std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        const bool cuda_enable = this->get_target().has_feature(Target::Feature::CUDA);
        bool dnndk_enable = true;
#ifdef HALIDE_FOR_FPGA
        dnndk_enable = (this->get_target().has_feature(Target::Feature::DPU));
#endif

        input = Func{static_cast<std::string>(gc_prefix) + "in"};
        input(_) = input_(_);

        std::vector<ExternFuncArgument> params{input, session_id_buf, model_root_url_buf, cache_path_buf, cuda_enable, dnndk_enable};
        Func object_detection(static_cast<std::string>(gc_prefix) + "output");
        object_detection.define_extern("ion_bb_dnn_generic_object_detection", params, Float(32), D);
        object_detection.compute_root();

        output = object_detection;
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
    BuildingBlockParam<std::string> gc_title{"gc_title", "Object Detection"};
};

class ObjectDetectionArray : public ObjectDetectionBase<ObjectDetectionArray, 4> {
public:
    // BuildingBlockParam<std::string> gc_title{"gc_title", "Object Detection (Array)"};
};

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ReorderCHW2HWC<uint8_t>, dnn_reorder_chw2hwc);
ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ReorderHWC2CHW<uint8_t>, dnn_reorder_hwc2chw);
ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ObjectDetection, dnn_object_detection);
ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ObjectDetectionArray, dnn_object_detection_array);

namespace ion {
namespace bb {
namespace dnn {

class TLTObjectDetectionSSD : public BuildingBlock<TLTObjectDetectionSSD> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "TLT Object Detection SSD"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "Detect objects by TLT Object Detection SSD models."};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "processing,recognition"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};
    BuildingBlockParam<std::string> gc_required_features{"gc_required_features", "cuda"};

    BuildingBlockParam<std::string> model_root_url_{"model_base_url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/tlt_object_detection_ssd_resnet18/"};
    BuildingBlockParam<std::string> cache_root_{"cache_root", "/tmp/"};

    Input<Halide::Func> input_{"input", Halide::type_of<float>(), 3};
    Output<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string model_root_url(model_root_url_);
        Halide::Buffer<uint8_t> model_root_url_buf(model_root_url.size() + 1);
        model_root_url_buf.fill(0);
        std::memcpy(model_root_url_buf.data(), model_root_url.c_str(), model_root_url.size());

        const std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        input = Func{static_cast<std::string>(gc_prefix) + "in"};
        input(_) = input_(_);

        std::vector<ExternFuncArgument> params{input, session_id_buf, model_root_url_buf, cache_path_buf};
        Func object_detection(static_cast<std::string>(gc_prefix) + "output");
        object_detection.define_extern("ion_bb_dnn_tlt_object_detection_ssd", params, Float(32), 3);
        object_detection.compute_root();

        output = object_detection;
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

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::TLTObjectDetectionSSD, dnn_tlt_object_detection_ssd);

namespace ion {
namespace bb {
namespace dnn {

class TLTPeopleNet : public BuildingBlock<TLTPeopleNet> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "TLT PeopleNet"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "Detect, People, Face and Bag by TLT PeopleNet models."};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: v.input }}))"};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "processing,recognition"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};
    BuildingBlockParam<std::string> gc_required_features{"gc_required_features", "cuda"};

    BuildingBlockParam<std::string> model_root_url_{"model_base_url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/tlt_peoplenet_detectnet_v2_resnet18/"};
    BuildingBlockParam<std::string> cache_root_{"cache_root", "/tmp/"};

    Input<Halide::Func> input_{"input", Halide::type_of<float>(), 3};
    Output<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string model_root_url(model_root_url_);
        Halide::Buffer<uint8_t> model_root_url_buf(model_root_url.size() + 1);
        model_root_url_buf.fill(0);
        std::memcpy(model_root_url_buf.data(), model_root_url.c_str(), model_root_url.size());

        const std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        input = Func{static_cast<std::string>(gc_prefix) + "in"};
        input(_) = input_(_);

        std::vector<ExternFuncArgument> params{input, session_id_buf, model_root_url_buf, cache_path_buf};
        Func inference(static_cast<std::string>(gc_prefix) + "output");
        inference.define_extern("ion_bb_dnn_tlt_peoplenet", params, Float(32), 3);
        inference.compute_root();

        output = inference;
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

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::TLTPeopleNet, dnn_tlt_peoplenet);

namespace ion {
namespace bb {
namespace dnn {

class TLTPeopleNetMD : public BuildingBlock<TLTPeopleNetMD> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "TLT PeopleNet metadata version"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "Detect, People, Face and Bag by TLT PeopleNet models and create detection metadata."};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.output_size)] }}))"};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "processing,recognition"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};
    BuildingBlockParam<std::string> gc_required_features{"gc_required_features", "cuda"};

    BuildingBlockParam<std::string> model_root_url_{"model_base_url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/tlt_peoplenet_detectnet_v2_resnet18/"};
    BuildingBlockParam<std::string> cache_root_{"cache_root", "/tmp/"};
    BuildingBlockParam<int> input_width{"width", 640};
    BuildingBlockParam<int> input_height{"height", 480};
    BuildingBlockParam<int> output_size{"output_size", 16 * 1024 * 1024};  // 16MiB

    Input<Halide::Func> input_{"input", Halide::type_of<float>(), 3};
    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 1};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string model_root_url(model_root_url_);
        Halide::Buffer<uint8_t> model_root_url_buf(model_root_url.size() + 1);
        model_root_url_buf.fill(0);
        std::memcpy(model_root_url_buf.data(), model_root_url.c_str(), model_root_url.size());

        const std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        input = Func{static_cast<std::string>(gc_prefix) + "in"};
        input(_) = input_(_);

        std::vector<ExternFuncArgument> params{input, static_cast<int>(input_width), static_cast<int>(input_height), static_cast<int>(output_size), session_id_buf, model_root_url_buf, cache_path_buf};
        Func inference(static_cast<std::string>(gc_prefix) + "output");
        inference.define_extern("ion_bb_dnn_tlt_peoplenet_md", params, UInt(8), 1);
        inference.compute_root();

        output = inference;
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

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::TLTPeopleNetMD, dnn_tlt_peoplenet_md);

namespace ion {
namespace bb {
namespace dnn {

class ClassifyGender : public BuildingBlock<ClassifyGender> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "ClassifyGender"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "Classify gender in image based on detection result."};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [parseInt(v.output_size)] }}))"};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "processing,recognition"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<std::string> model_root_url_{"model_base_url", "http://ion-kit.s3.us-west-2.amazonaws.com/models/classify_gender/"};
    BuildingBlockParam<std::string> cache_root_{"cache_root", "/tmp/"};
    BuildingBlockParam<int> input_img_width{"width", 0};
    BuildingBlockParam<int> input_img_height{"height", 0};
    BuildingBlockParam<int> input_md_size{"input_md_size", 16 * 1024 * 1024};  // 16MiB
    BuildingBlockParam<int> output_size{"output_size", 16 * 1024 * 1024};      // 16MiB

    Input<Halide::Func> input_img{"image", Halide::type_of<float>(), 3};
    Input<Halide::Func> input_md{"metadata", Halide::type_of<uint8_t>(), 1};
    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 1};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        const std::string model_root_url(model_root_url_);
        Halide::Buffer<uint8_t> model_root_url_buf(model_root_url.size() + 1);
        model_root_url_buf.fill(0);
        std::memcpy(model_root_url_buf.data(), model_root_url.c_str(), model_root_url.size());

        const std::string cache_root(cache_root_);
        Halide::Buffer<uint8_t> cache_path_buf(cache_root.size() + 1);
        cache_path_buf.fill(0);
        std::memcpy(cache_path_buf.data(), cache_root.c_str(), cache_root.size());

        input_img_ = Func{static_cast<std::string>(gc_prefix) + "input_img"};
        input_img_(_) = input_img(_);

        input_md_ = Func{static_cast<std::string>(gc_prefix) + "input_md"};
        input_md_(_) = input_md(_);

        std::vector<ExternFuncArgument> params{
            input_img_, cast<uint32_t>(input_img_width), cast<uint32_t>(input_img_height),
            input_md_, cast<uint32_t>(input_md_size),
            cast<uint32_t>(output_size), session_id_buf, model_root_url_buf, cache_path_buf};
        Func inference(static_cast<std::string>(gc_prefix) + "output");
        inference.define_extern("ion_bb_dnn_classify_gender", params, UInt(8), 1);
        inference.compute_root();

        output = inference;
    }

    void schedule() {
        using namespace Halide;
        Var c = input_img_.args()[0];
        Var x = input_img_.args()[1];
        Var y = input_img_.args()[2];

        input_img_.bound(c, 0, 3).unroll(c);

        if (this->get_target().has_gpu_feature()) {
            Var xi, yi;
            input_img_.gpu_tile(x, y, xi, yi, 32, 16);
        } else {
            input_img_.vectorize(x, this->natural_vector_size(Float(32))).parallel(y, 16);
        }
        input_img_.compute_root();

        input_md_.compute_root();
    }

private:
    Halide::Func input_img_;
    Halide::Func input_md_;
};

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::ClassifyGender, dnn_classify_gender);

namespace ion {
namespace bb {
namespace dnn {

class JSONDictAverageRegulator : public BuildingBlock<JSONDictAverageRegulator> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "JSONDictAverageRegulator"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "Takes JSON key/value dictionary, accumulate value and calculate average, and emit in paticular time period."};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [ parseInt(v.io_md_size) ] }}))"};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "processing,json"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", ""};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};

    BuildingBlockParam<uint32_t> io_md_size{"io_md_size", 16 * 1024 * 1024};  // 16MiB
    BuildingBlockParam<uint32_t> period_in_sec{"period_in_sec", 30};

    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 1};
    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 1};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        input_ = Func{static_cast<std::string>(gc_prefix) + "input"};
        input_(_) = input(_);

        std::vector<ExternFuncArgument> params{input_, cast<uint32_t>(io_md_size), session_id_buf, cast<uint32_t>(period_in_sec)};
        Func regurator(static_cast<std::string>(gc_prefix) + "output");
        regurator.define_extern("ion_bb_dnn_json_dict_average_regurator", params, UInt(8), 1);
        regurator.compute_root();

        output = regurator;
    }

    void schedule() {
        input_.compute_root();
    }

private:
    Halide::Func input_;
};

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::JSONDictAverageRegulator, dnn_json_dict_average_regulator);

namespace ion {
namespace bb {
namespace dnn {

class IFTTTWebHookUploader : public ion::BuildingBlock<IFTTTWebHookUploader> {
public:
    BuildingBlockParam<std::string> gc_title{"gc_title", "IFTTT WebHook Uploader"};
    BuildingBlockParam<std::string> gc_description{"gc_description", "This makes POST request against to webhook endpoint on ifttt.com."};
    BuildingBlockParam<std::string> gc_tags{"gc_tags", "output,network"};
    BuildingBlockParam<std::string> gc_inference{"gc_inference", R"((function(v){ return { output: [] }}))"};
    BuildingBlockParam<std::string> gc_mandatory{"gc_mandatory", "ifttt_webhook_url"};
    BuildingBlockParam<std::string> gc_strategy{"gc_strategy", "self"};
    BuildingBlockParam<std::string> gc_prefix{"gc_prefix", ""};
    BuildingBlockParam<uint32_t> input_md_size{"input_md_size", 16 * 1024 * 1024};  // 16MiB
    BuildingBlockParam<std::string> ifttt_webhook_url{"ifttt_webhook_url", ""};
    Input<Halide::Func> input_md{"input_md", Halide::type_of<uint8_t>(), 1};
    Output<int32_t> output{"output"};

    void generate() {
        using namespace Halide;

        std::string session_id = sole::uuid4().str();
        Buffer<uint8_t> session_id_buf(session_id.size() + 1);
        session_id_buf.fill(0);
        std::memcpy(session_id_buf.data(), session_id.c_str(), session_id.size());

        std::string ifttt_webhook_url_str(ifttt_webhook_url);
        Halide::Buffer<uint8_t> ifttt_webhook_url_buf(ifttt_webhook_url_str.size() + 1);
        ifttt_webhook_url_buf.fill(0);
        std::memcpy(ifttt_webhook_url_buf.data(), ifttt_webhook_url_str.c_str(), ifttt_webhook_url_str.size());

        input_md_ = Func{static_cast<std::string>(gc_prefix) + "input_md"};
        input_md_(_) = input_md(_);

        std::vector<ExternFuncArgument> params = {input_md_, static_cast<int>(input_md_size), session_id_buf, ifttt_webhook_url_buf};
        Func uploader(static_cast<std::string>(gc_prefix) + "ifttt_webhook_uploader");
        uploader.define_extern("ion_bb_dnn_ifttt_webhook_uploader", params, Int(32), 0);
        uploader.compute_root();
        output() = uploader();
    }

    void schedule() {
        input_md_.compute_root();
    }

    Halide::Func input_md_;
};

}  // namespace dnn
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::dnn::IFTTTWebHookUploader, dnn_ifttt_webhook_uploader);

#endif  // ION_BB_DNN_BB_H
