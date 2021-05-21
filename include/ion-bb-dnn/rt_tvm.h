#ifndef ION_BB_DNN_RT_TVM_H
#define ION_BB_DNN_RT_TVM_H

#include <string>
#include <sys/utsname.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#include <HalideBuffer.h>

namespace ion {
namespace bb {
namespace dnn {

namespace {

// Function ptrs for tvm c api functions
#define DECL_TVM_FUNC(NAME) decltype(&::NAME) NAME;
DECL_TVM_FUNC(TVMFuncGetGlobal);
DECL_TVM_FUNC(TVMFuncCall);
DECL_TVM_FUNC(TVMGetLastError);
DECL_TVM_FUNC(TVMModLoadFromFile);
DECL_TVM_FUNC(TVMModGetFunction);
DECL_TVM_FUNC(TVMArrayAlloc);
DECL_TVM_FUNC(TVMArrayCopyFromBytes);

// dlsym for tvm functions
#define RESOLVE_TVM_FUNC(NAME)                                    \
    ion::bb::dnn::NAME = dm.get_symbol<decltype(&::NAME)>(#NAME); \
    if (ion::bb::dnn::NAME == nullptr) {                          \
        throw std::runtime_error(                                 \
            #NAME " is unavailable on your tvm runtime");         \
    }

bool tvm_runtime_init() {
    // should locate libtvm_runtime.so under searchable directory like /usr/lib
    static ion::bb::dnn::DynamicModule dm("tvm_runtime");
    if (!dm.is_available()) {
        return false;
    }

    RESOLVE_TVM_FUNC(TVMFuncGetGlobal);
    RESOLVE_TVM_FUNC(TVMGetLastError);
    RESOLVE_TVM_FUNC(TVMFuncCall);
    RESOLVE_TVM_FUNC(TVMModLoadFromFile);
    RESOLVE_TVM_FUNC(TVMModGetFunction);
    RESOLVE_TVM_FUNC(TVMArrayAlloc);
    RESOLVE_TVM_FUNC(TVMArrayCopyFromBytes);
    return true;
}
#undef DECL_TVM_FUNC
#undef RESOLVE_TVM_FUNC

std::string load_file(const std::string &path, bool binary = false) {
    const auto mode = binary ? std::ios::binary : std::ios::in;
    std::ifstream ifs(path, mode);
    if (ifs.fail()) {
        throw std::runtime_error("could not open file");
    }

    std::string data((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());

    return data;
}

template<typename T>
T ndarray_at(TVMArrayHandle array,
             const std::vector<int64_t> &indices) {

    assert(array->ndim == indices.size());
    // TODO: Support strides.
    assert(array->strides == nullptr);
    // TODO: Support offset.
    assert(array->byte_offset == 0U);

    int64_t *shape = array->shape;
    int64_t index = 0;
    for (auto dim = decltype(indices.size())(0); dim < indices.size(); ++dim) {
        int64_t stride = 1;
        for (size_t i = dim + 1; i < array->ndim; ++i) {
            stride *= shape[i];
        }

        index += indices[dim] * stride;
    };

    return reinterpret_cast<T *>(array->data)[index];
}

std::vector<DetectionBox> postprocess(TVMArrayHandle nums,
                                      TVMArrayHandle boxes,
                                      TVMArrayHandle labels,
                                      TVMArrayHandle scores) {
    // Assume single batch.
    const int64_t batch = 0;
    int32_t num = -1;
    auto dtype_code = nums->dtype.code;
    if (dtype_code == kDLInt) {
        num = (int32_t)ndarray_at<int32_t>(nums, {0});
    } else if (dtype_code == kDLFloat) {
        num = (int32_t)ndarray_at<float>(nums, {0});
    }

    if (num < 0) {
        std::cerr << "Error: failed to get num" << std::endl;
    }

    std::vector<DetectionBox> valid_boxes;
    for (auto i = decltype(num)(0); i < num; ++i) {
        // Added offset of background.
        const auto class_id =
            static_cast<int32_t>(ndarray_at<float>(labels, {batch, i})) + 1;
        const auto confidence = ndarray_at<float>(scores, {batch, i});

        // Boxes is formatted as [y1, x1, y2, x2] nomalized [0, 1)
        const auto x1 = ndarray_at<float>(boxes, {batch, i, 1});
        const auto y1 = ndarray_at<float>(boxes, {batch, i, 0});
        const auto x2 = ndarray_at<float>(boxes, {batch, i, 3});
        const auto y2 = ndarray_at<float>(boxes, {batch, i, 2});

        valid_boxes.emplace_back(
            DetectionBox{class_id, confidence, x1, x2, y1, y2});
    }

    return valid_boxes;
}

}  // namespace

class TVMSessionManager {
private:
    bool init_configs_ = false;
    DLContext dlctx_;
    std::string model_base_url_ = "";
    std::string model_cache_dir_ = "";
    bool use_cuda_ = false;
    bool use_edgetpu_ = false;
    std::string dnn_model_name_;
    std::string target_arch_;

    std::unordered_map<std::string, TVMModuleHandle> modules_;
    TVMFunctionHandle runtime_creator_;

    bool is_available_tvm_;
    TVMSessionManager()
        : is_available_tvm_(false) {
        if (!tvm_runtime_init()) {
            std::cerr << "Log: could not init the tvm runtime" << std::endl;
            return;
        }
        is_available_tvm_ = true;
    }

    void init_runtime_creator() {

        if (use_edgetpu_) {
            TVMFuncGetGlobal("tvm.edgetpu_runtime.create", &runtime_creator_);
        } else {
            TVMFuncGetGlobal("tvm.graph_runtime.create", &runtime_creator_);
        }

        if (!runtime_creator_) {
            std::cerr << "Error: failed to get any runtime creator from tvm" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }

        return;
    }

    std::string resolve_model_filename() {
        std::string model_file = dnn_model_name_;

        if (use_edgetpu_) {
            model_file += "_int8_edgetpu.tflite";
        } else {
            model_file += "_" + target_arch_;
            if (use_cuda_) {
                model_file += "_cuda";
            }
            model_file += ".tar";
        }

        return model_file;
    }

    bool have_file_cache(const std::string &filename) {
        std::ifstream ifs(model_cache_dir_ + filename, std::ios::binary);
        return ifs.is_open();
    }

    bool download_model_file(const std::string &model_url, const std::string &model_file) {

        std::string host_name;
        std::string path_name;
        std::tie(host_name, path_name) = parse_url(model_url);
        if (host_name.empty() || path_name.empty()) {
            std::cerr << "Error: invalid model URL : " << model_url << std::endl;
            return false;
        }

        httplib::Client cli(host_name.c_str());
        cli.set_follow_location(true);
        auto res = cli.Get(path_name.c_str());
        if (!res || res->status != 200) {
            std::cerr << "Error: failed to download model (cli.Get): " << model_url << std::endl;
            return false;
        }

        std::shared_ptr<std::vector<uint8_t>> model_data;
        model_data = std::shared_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(res->body.size()));
        std::memcpy(model_data->data(), res->body.c_str(), res->body.size());
        std::ofstream ofs(model_cache_dir_ + model_file, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error: failed to open a cache entry : " << model_cache_dir_ + model_file << std::endl;
            return false;
        }
        ofs.write(reinterpret_cast<const char *>(model_data->data()), model_data->size());

        return true;
    }

    TVMModuleHandle create_runtime_module(const std::string &model_filename) {
        TVMModuleHandle M;

        if (use_edgetpu_) {
            const auto tflite_model_bytes = load_file(model_cache_dir_ + model_filename, true);
            TVMByteArray model_bytearray{tflite_model_bytes.c_str(), tflite_model_bytes.length()};
            TVMValue args[2];
            args[0].v_handle = &model_bytearray;
            args[1].v_ctx = dlctx_;

            int arg_types[2];
            arg_types[0] = kTVMBytes;
            arg_types[1] = kTVMContext;

            TVMValue ret;
            int ret_type;
            if (TVMFuncCall(runtime_creator_, &args[0], &arg_types[0], 2, &ret, &ret_type) < 0) {
                std::cerr << "Error: failed to execute runtime_creator_ for tflite(edgetpu)" << std::endl;
                std::cerr << TVMGetLastError() << std::endl;
                exit(1);
            }
            M = ret.v_handle;
        } else {
            // TODO a new graph API is comming in the next tvm release
            // Using mod.json, mod.so, and mod.params is depricated.
            // We can simply use one shared library for exporiting and loading a tvm::runtime::Module insted
            std::string extract_cmd = "tar xf " + model_cache_dir_ + model_filename + " -C " + model_cache_dir_;
            std::system(extract_cmd.c_str());

            const auto json_data = load_file(model_cache_dir_ + "mod.json");
            std::string model_file_path = model_cache_dir_ + "mod.so";
            TVMModuleHandle mod;
            if (TVMModLoadFromFile(model_file_path.c_str(), "", &mod) < 0) {
                std::cerr << "Error: failed to load model file" << std::endl;
                exit(1);
            }

            TVMValue args[4];
            args[0].v_str = json_data.c_str();
            args[1].v_handle = mod;
            args[2].v_int64 = static_cast<int>(dlctx_.device_type);
            args[3].v_int64 = dlctx_.device_id;

            int arg_types[4];
            arg_types[0] = kTVMStr;
            arg_types[1] = kTVMModuleHandle;
            arg_types[2] = kTVMArgInt;
            arg_types[3] = kTVMArgInt;

            TVMValue ret;
            int ret_type;
            if (TVMFuncCall(runtime_creator_, &args[0], &arg_types[0], 4, &ret, &ret_type) < 0) {
                std::cerr << "Error: failed to execute runtime_creator_" << std::endl;
                std::cerr << TVMGetLastError() << std::endl;
                exit(1);
            }
            M = ret.v_handle;

            const auto params_data = load_file(model_cache_dir_ + "mod.params", true);
            TVMByteArray params_bytearray{params_data.c_str(), params_data.length()};
            TVMFunctionHandle load_params;
            TVMModGetFunction(M, "load_params", false, &load_params);

            TVMValue arg2;
            arg2.v_handle = &params_bytearray;
            int type_code2 = kTVMBytes;
            if (TVMFuncCall(load_params, &arg2, &type_code2, 1, &ret, &ret_type) < 0) {
                std::cerr << "Error: failed to execute load_params" << std::endl;
                std::cerr << TVMGetLastError() << std::endl;
                exit(1);
            }
        }

        return M;
    }

public:
    static TVMSessionManager &get_instance() {
        static TVMSessionManager instance;
        return instance;
    }

    bool is_tvm_available() {
        return is_available_tvm_;
    }

    void setup_config(const DLContext &ctx,
                      const std::string &model_root_url,
                      const std::string &cache_root,
                      bool cuda_enable,
                      bool edgetpu_enable,
                      const std::string &dnn_model_name,
                      const std::string &target_arch) {
        dlctx_ = ctx;
        model_base_url_ = model_root_url;
        model_cache_dir_ = cache_root;
        use_cuda_ = cuda_enable;
        use_edgetpu_ = edgetpu_enable;
        dnn_model_name_ = dnn_model_name;
        target_arch_ = target_arch;
        init_configs_ = true;
    }

    TVMModuleHandle init_runtime_module() {

        if (!init_configs_) {
            std::cerr << "Error: setup_config() should be called before init_runtime_module()" << std::endl;
            exit(1);
        }

        init_runtime_creator();

        std::string model_filename = resolve_model_filename();
        if (model_filename == "") {
            std::cerr << "Error: failed to resolve the model filename" << std::endl;
            exit(1);
        }

        // Check the session manager has already cached the runtime module for this model.
        std::string model_url = model_base_url_ + "tvm/" + dnn_model_name_ + "/" + model_filename;
        if (modules_.count(model_url)) {
            return modules_[model_url];
        }

        if (!have_file_cache(model_filename)) {
            if (!download_model_file(model_url, model_filename)) {
                std::cerr << "Failed to download a model file from " << model_url << std::endl;
                exit(1);
            }
        }

        TVMModuleHandle M = create_runtime_module(model_filename);
        modules_[model_url] = M;
        return modules_[model_url];
    }

    void set_input(const TVMModuleHandle M, const TVMArrayHandle data) {
        TVMFunctionHandle set_input;
        TVMModGetFunction(M, "set_input", false, &set_input);

        if (!set_input) {
            std::cerr << "Error: failed to get a set_input function" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }

        TVMValue args[2];
        int arg_types[2];
        std::string input_name = "normalized_input_image_tensor";
        if (use_edgetpu_) {
            args[0].v_int64 = 0;
            arg_types[0] = kTVMArgInt;
        } else {
            args[0].v_str = input_name.c_str();
            arg_types[0] = kTVMStr;
        }
        args[1].v_handle = data;
        arg_types[1] = kTVMNDArrayHandle;
        TVMValue ret;
        int ret_code;
        if (TVMFuncCall(set_input, &args[0], &arg_types[0], 2, &ret, &ret_code) < 0) {
            std::cerr << "Error: failed to call set_input" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }
    }

    void run_inference(TVMModuleHandle M) {
        TVMFunctionHandle inference;

        if (use_edgetpu_) {
            TVMModGetFunction(M, "invoke", false, &inference);
        } else {
            TVMModGetFunction(M, "run", false, &inference);
        }

        if (!inference) {
            std::cerr << "Error: failed to get a inference function" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }

        TVMValue ret;
        int ret_code;
        if (TVMFuncCall(inference, nullptr, nullptr, 0, &ret, &ret_code) < 0) {
            std::cerr << "Error: failed to call inference" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }
    }

    TVMArrayHandle get_output(TVMModuleHandle M, int index) {
        TVMFunctionHandle get_output;

        TVMModGetFunction(M, "get_output", false, &get_output);

        TVMValue arg;
        arg.v_int64 = index;
        int arg_type = kTVMArgInt;

        TVMValue ret;
        int ret_code;
        if (TVMFuncCall(get_output, &arg, &arg_type, 1, &ret, &ret_code) < 0) {
            std::cerr << "Error: failed to call inference" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }

        return reinterpret_cast<TVMArrayHandle>(ret.v_handle);
    }
};

bool is_tvm_available() {
    return TVMSessionManager::get_instance().is_tvm_available();
}

int object_detection_tvm(halide_buffer_t *in,
                         const std::string &model_root_url,
                         const std::string &cache_root,
                         bool cuda_enable,
                         bool edgetpu_enable,
                         const std::string &dnn_model_name,
                         const std::string &target_arch,
                         halide_buffer_t *out) {
    const int channel = 3;
    const int width = in->dim[1].extent;
    const int height = in->dim[2].extent;

    size_t input_size = 3 * width * height * sizeof(float);
    int num_images = in->dimensions == 3 ? 1 : in->dim[3].extent;

    for (int i = 0; i < num_images; ++i) {
        // Open image
        int offset = input_size * i;
        cv::Mat in_(height, width, CV_32FC3, in->host + offset);

        // Init tvm::runtime::Module
        auto &mgr = TVMSessionManager::get_instance();
        const DLContext ctx{kDLCPU, 1};
        mgr.setup_config(ctx, model_root_url, cache_root, cuda_enable, edgetpu_enable, dnn_model_name, target_arch);
        TVMModuleHandle M = mgr.init_runtime_module();

        // Resize for tensor input data
        const auto internal_height = 300;
        const auto internal_width = 300;
        cv::Mat resized(internal_height, internal_width, CV_32FC3);
        cv::resize(in_, resized, resized.size());

        // Convert cv::Mat into tvm::runtime::NDArray
        TVMArrayHandle tensor_input_data;
        int64_t shape[4] = {1, internal_width, internal_height, 3};
        if (TVMArrayAlloc(shape, 4, kDLFloat, 32, 1, ctx.device_type, ctx.device_id, &tensor_input_data) < 0) {
            std::cerr << "Error: failed to allocate tvm array" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }

        if (TVMArrayCopyFromBytes(tensor_input_data, resized.ptr(), resized.total() * resized.elemSize()) < 0) {
            std::cerr << "Error: failed to copy from cv::Mat" << std::endl;
            std::cerr << TVMGetLastError() << std::endl;
            exit(1);
        }

        // SetInput
        mgr.set_input(M, tensor_input_data);

        // Run inference
        mgr.run_inference(M);

        // GetOutput
        TVMArrayHandle boxes = mgr.get_output(M, 0);
        TVMArrayHandle labels = mgr.get_output(M, 1);
        TVMArrayHandle scores = mgr.get_output(M, 2);
        TVMArrayHandle num = mgr.get_output(M, 3);
        const auto valid_boxes = postprocess(num, boxes, labels, scores);

        cv::Mat out_(height, width, CV_32FC3, out->host + offset);
        in_.copyTo(out_);
        coco_render_boxes(out_, valid_boxes, width, height);
    }

    return 0;
}

}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif  // ION_BB_DNN_RT_TVM_H
