#ifndef ION_BB_DNN_SSDLITE_RT_H
#define ION_BB_DNN_SSDLITE_RT_H

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <HalideBuffer.h>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

#include <tensorrt_provider_factory.h>

namespace {

class SessionManager {
public:
    SessionManager() {
        env = std::shared_ptr<Ort::Env>(new Ort::Env);
        session_map = std::shared_ptr<std::map<std::string, std::shared_ptr<Ort::Session>>>(new std::map<std::string, std::shared_ptr<Ort::Session>>{});
    }
    ~SessionManager() {
        session_map.reset();
        env.reset();
    }

    std::shared_ptr<Ort::Session> get_session(std::string session_id, const void *model, size_t model_size, const char *model_path, bool can_cuda) {
        auto &s_map = *session_map.get();
        if (s_map[session_id].get() == nullptr) {
            Ort::SessionOptions session_options;
            if (can_cuda && can_tensorrt()) {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
            }

            if (std::strlen(model_path) == 0) {
                s_map[session_id] = std::shared_ptr<Ort::Session>{new Ort::Session{*env.get(), model, model_size, session_options}};
            } else {
                s_map[session_id] = std::shared_ptr<Ort::Session>{new Ort::Session{*env.get(), model_path, session_options}};
            }
        }

        return s_map[session_id];
    }

private:
    std::shared_ptr<Ort::Env> env;
    std::shared_ptr<std::map<std::string, std::shared_ptr<Ort::Session>>> session_map;

    bool can_tensorrt() {
        bool can_tensorrt = false;
        void *handle = dlopen("libnvinfer.so", RTLD_LAZY);
        if (handle != NULL) {
            can_tensorrt = true;
            dlclose(handle);
        }

        return can_tensorrt;
    }
};

std::shared_ptr<SessionManager> get_session_manager(bool delete_manager) {
    static std::shared_ptr<SessionManager> session_manager;
    if (delete_manager == false && session_manager.get() == nullptr) {
        session_manager = std::shared_ptr<SessionManager>{new SessionManager};
    } else if (delete_manager == true && session_manager.get() != nullptr) {
        session_manager.reset();
    }
    return session_manager;
}

void release_onnxruntime() {
    get_session_manager(true);
}

void run_inference(Ort::Session &session, std::vector<halide_buffer_t *> &input_buffers, std::vector<int64_t> &dims, std::vector<halide_buffer_t *> &output_buffers) {
    // common settings
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // create input data
    std::size_t input_num = session.GetInputCount();
    std::vector<const char *> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<Ort::Value> input_tensors;
    for (std::size_t idx = 0; idx < input_num; idx++) {
        // name
        input_names.push_back(session.GetInputName(idx, allocator));

        // shape
        std::vector<int64_t> raw_input_shape = session.GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        for (size_t i = 0; i < raw_input_shape.size(); ++i) {
            // -1 means dynamic shape.
            if (raw_input_shape[i] == -1) {
                // Halide dimension order is reversed-
                raw_input_shape[i] = dims[dims.size() - 1 - i];
            }
        }
        input_shapes.push_back(raw_input_shape);

        // tensor
        auto &input_shape = input_shapes[idx];
        auto input_type_model = session.GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetElementType();
        if (input_type_model == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && input_buffers[idx]->type.code == halide_type_float && input_buffers[idx]->type.bits == 32) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float *>(input_buffers[idx]->host), input_buffers[idx]->number_of_elements(), input_shape.data(), input_shape.size()));
        } else if (input_type_model == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 && input_buffers[idx]->type.code == halide_type_uint && input_buffers[idx]->type.bits == 8) {
            input_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(memory_info, input_buffers[idx]->host, input_buffers[idx]->number_of_elements(), input_shape.data(), input_shape.size()));
        } else {
            // not support
        }
    }

    // create output data
    std::size_t output_num = session.GetOutputCount();
    std::vector<const char *> output_names;

    for (std::size_t idx = 0; idx < output_num; idx++) {
        // name
        output_names.push_back(session.GetOutputName(idx, allocator));
    }

    // run inference
    std::vector<Ort::Value> raw_outputs = session.Run(Ort::RunOptions{nullptr}, &input_names[0], &input_tensors[0], input_num, &output_names[0], output_num);

    std::size_t out_size = std::min(raw_outputs.size(), output_buffers.size());
    for (std::size_t idx = 0; idx < out_size; idx++) {
        // it's because Halide define_extern can return only Funcs
        std::size_t out_elem_num = std::min(raw_outputs[idx].GetTensorTypeAndShapeInfo().GetElementCount(), output_buffers[idx]->number_of_elements());

        memcpy(output_buffers[idx]->host, raw_outputs[idx].GetTensorMutableData<uint8_t>(), sizeof(float) * out_elem_num);
    }
}

int onnx_inference_base(std::vector<halide_buffer_t *> &input_buffers, halide_buffer_t *session_id_buf, halide_buffer_t *model, int model_size, halide_buffer_t *model_path_buf, std::vector<int64_t> &dims, bool can_cuda, std::vector<halide_buffer_t *> &output_buffers) {
    bool is_bounds_query = false;
    for (auto it = input_buffers.begin(); it != input_buffers.end(); it++) {
        if ((*it)->is_bounds_query()) {
            is_bounds_query = true;
            for (auto i = 0; i < dims.size(); ++i) {
                (*it)->dim[i].min = 0;
                (*it)->dim[i].extent = dims[i];
            }
        }
    }

    if (model->is_bounds_query()) {
        is_bounds_query = true;
        if (model->dimensions > 0) {
            model->dim[0].min = 0;
            model->dim[0].extent = model_size;
        }
    }

    if (!is_bounds_query) {
        std::shared_ptr<SessionManager> session_manager = get_session_manager(false);

        const char *model_path = reinterpret_cast<const char *>(model_path_buf->host);
        std::string session_id = reinterpret_cast<const char *>(session_id_buf->host);

        std::shared_ptr<Ort::Session> session = session_manager->get_session(session_id, model->host, static_cast<size_t>(model_size), model_path, can_cuda);
        run_inference(*session.get(), input_buffers, dims, output_buffers);
    }

    return 0;
}

}  // namespace

extern "C" ION_EXPORT int onnx_inference_out1(halide_buffer_t *in, halide_buffer_t *session_id_buf, halide_buffer_t *model, int model_size, halide_buffer_t *model_path_buf, int dim0, int dim1, int dim2, int dim3, bool can_cuda, halide_buffer_t *out) {
    std::vector<halide_buffer_t *> input_buffers{in};
    std::vector<halide_buffer_t *> output_buffers{out};
    std::vector<int64_t> dims{dim0, dim1, dim2, dim3};
    return onnx_inference_base(input_buffers, session_id_buf, model, model_size, model_path_buf, dims, can_cuda, output_buffers);
}

extern "C" ION_EXPORT int onnx_inference_out2(halide_buffer_t *in, halide_buffer_t *session_id_buf, halide_buffer_t *model, int model_size, halide_buffer_t *model_path_buf, int dim0, int dim1, int dim2, int dim3, bool can_cuda, halide_buffer_t *out0, halide_buffer_t *out1) {
    std::cout << "here" << std::endl;
    std::vector<halide_buffer_t *> input_buffers{in};
    std::vector<halide_buffer_t *> output_buffers{out0, out1};
    std::vector<int64_t> dims{dim0, dim1, dim2, dim3};
    return onnx_inference_base(input_buffers, session_id_buf, model, model_size, model_path_buf, dims, can_cuda, output_buffers);
}

extern "C" ION_EXPORT int onnx_inference_out3(halide_buffer_t *in, halide_buffer_t *session_id_buf, halide_buffer_t *model, int model_size, halide_buffer_t *model_path_buf, int dim0, int dim1, int dim2, int dim3, bool can_cuda, halide_buffer_t *out0, halide_buffer_t *out1, halide_buffer_t *out2) {
    std::vector<halide_buffer_t *> input_buffers{in};
    std::vector<halide_buffer_t *> output_buffers{out0, out1, out2};
    std::vector<int64_t> dims{dim0, dim1, dim2, dim3};
    return onnx_inference_base(input_buffers, session_id_buf, model, model_size, model_path_buf, dims, can_cuda, output_buffers);
}

extern "C" ION_EXPORT int onnx_inference_out4(halide_buffer_t *in, halide_buffer_t *session_id_buf, halide_buffer_t *model, int model_size, halide_buffer_t *model_path_buf, int dim0, int dim1, int dim2, int dim3, bool can_cuda, halide_buffer_t *out0, halide_buffer_t *out1, halide_buffer_t *out2, halide_buffer_t *out3) {
    std::vector<halide_buffer_t *> input_buffers{in};
    std::vector<halide_buffer_t *> output_buffers{out0, out1, out2, out3};
    std::vector<int64_t> dims{dim0, dim1, dim2, dim3};
    return onnx_inference_base(input_buffers, session_id_buf, model, model_size, model_path_buf, dims, can_cuda, output_buffers);
}

extern "C" ION_EXPORT int bounding_box_renderer(halide_buffer_t *in, halide_buffer_t *boxes, halide_buffer_t *classes, halide_buffer_t *scores, halide_buffer_t *nums, int32_t width, int32_t height, halide_buffer_t *out) {
    bool bounds_query = false;
    if (in->is_bounds_query()) {
        bounds_query = true;
        in->dim[0].min = 0;
        in->dim[0].extent = 3;
        in->dim[1].min = 0;
        in->dim[1].extent = width;
        in->dim[2].min = 0;
        in->dim[2].extent = height;
    }

    if (boxes->is_bounds_query()) {
        bounds_query = true;
        boxes->dim[0].min = 0;
        boxes->dim[0].extent = 400;
    }

    if (classes->is_bounds_query()) {
        bounds_query = true;
        classes->dim[0].min = 0;
        classes->dim[0].extent = 100;
    }

    if (scores->is_bounds_query()) {
        bounds_query = true;
        scores->dim[0].min = 0;
        scores->dim[0].extent = 100;
    }

    if (nums->is_bounds_query()) {
        bounds_query = true;
        nums->dim[0].min = 0;
        nums->dim[0].extent = 1;
    }

    if (!bounds_query) {
        const char *labels[] = {
            "background",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "12",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "26",
            "backpack",
            "umbrella",
            "29",
            "30",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "45",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "66",
            "dining table",
            "68",
            "69",
            "toilet",
            "71",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "83",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        };

        cv::Mat src(std::vector<int>{height, width}, CV_MAKETYPE(CV_8U, 3), in->host);
        cv::Mat dst(std::vector<int>{height, width}, CV_MAKETYPE(CV_8U, 3), out->host);

        for (int y = 0; y < dst.rows; ++y) {
            for (int x = 0; x < dst.cols; ++x) {
                dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y, x);
            }
        }

        for (int i = 0; i < static_cast<int>(reinterpret_cast<float *>(nums->host)[0]); ++i) {

            float top = reinterpret_cast<float *>(boxes->host)[4 * i + 0];
            float left = reinterpret_cast<float *>(boxes->host)[4 * i + 1];
            float bottom = reinterpret_cast<float *>(boxes->host)[4 * i + 2];
            float right = reinterpret_cast<float *>(boxes->host)[4 * i + 3];

            cv::Point p0(static_cast<int>(left * width), static_cast<int>(top * height));
            cv::Point p1(static_cast<int>(right * width), static_cast<int>(bottom * height));
            cv::rectangle(dst, p0, p1, cv::Scalar(0, 0, 255));
            std::string s(labels[static_cast<int>(reinterpret_cast<float *>(classes->host)[i])]);
            s += ":" + std::to_string(reinterpret_cast<float *>(scores->host)[i]);
            cv::putText(dst, s, cv::Point(p0.x, p0.y - 3), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255));
        }
    }

    return 0;
}

#undef ION_EXPORT

#endif  // ION_BB_DNN_RT_H
