#ifndef ION_BB_OPENCV_RT_H
#define ION_BB_OPENCV_RT_H

#include <Halide.h>
#include <HalideBuffer.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

namespace ion {
namespace bb {
namespace opencv {

std::map<std::string, Halide::ExternCFunction> extern_functions;

class RegisterExtern {
 public:
     RegisterExtern(std::string key, Halide::ExternCFunction f) {
         extern_functions[key] = f;
     }
};

} // image_io
} // bb
} // ion

#define ION_REGISTER_EXTERN(NAME) static auto ion_register_extern_##NAME = ion::bb::opencv::RegisterExtern(#NAME, NAME);

namespace {
int hl2cv_type(halide_type_t hl_type, int channel) {
    if (hl_type.code != halide_type_uint) {
        return -1;
    }
    if (hl_type.bits == 8) {
        return CV_MAKETYPE(CV_8U, channel);
    } else if (hl_type.bits == 16) {
        return CV_MAKETYPE(CV_16U, channel);
    } else {
        return -1;
    }
}
} // namespace


extern "C" ION_EXPORT
int median_blur(halide_buffer_t *in, int channel, int width, int height, int ksize, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = channel;
        in->dim[1].min = 0;
        in->dim[1].extent = width;
        in->dim[2].min = 0;
        in->dim[2].extent = height;
    } else {
        int cv_type = hl2cv_type(in->type, channel);
        if (cv_type == -1) {
            return -1;
        }
        cv::Mat src(std::vector<int>{height, width}, cv_type, in->host);
        cv::Mat dst(std::vector<int>{height, width}, cv_type, out->host);
        cv::medianBlur(src, dst, ksize);
    }

    return 0;
}

extern "C" ION_EXPORT
int display(halide_buffer_t *in, int width, int height, int idx, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = 3; // RGB
        in->dim[1].min = 0;
        in->dim[1].extent = width;
        in->dim[2].min = 0;
        in->dim[2].extent = height;
    } else {
        cv::Mat img(std::vector<int>{height, width}, CV_8UC3, in->host);
        cv::imshow("img" + std::to_string(idx), img);
        cv::waitKey(1);
    }

    return 0;
}

#undef ION_EXPORT
#undef ION_REGISTER_EXTERN

#endif // ION_BB_OPENCV_RT_H
