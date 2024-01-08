#ifndef ION_BB_OPENCV_RT_H
#define ION_BB_OPENCV_RT_H

#include <Halide.h>
#include <HalideBuffer.h>

#include "dynamic_module.h"

#include "log.h"
#include "opencv_loader.h"

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

extern "C" ION_EXPORT
int median_blur(halide_buffer_t *in, int ksize, halide_buffer_t *out) {
    auto& cv(ion::bb::OpenCV::get_instance());
    if (!cv.is_available()) {
        ion::log::error("OpenCV is not available");
        return -1;
    }

    if (in->is_bounds_query()) {
        for (auto i=0; i<in->dimensions; ++i) {
            in->dim[i].min = out->dim[i].min;
            in->dim[i].extent = out->dim[i].extent;
        }
    } else {
        int width = in->dim[1].extent;
        int height = in->dim[2].extent;
        int cv_type = ion::bb::hl2cv_type(in->type, 3);
        if (cv_type == -1) {
            return -1;
        }

        auto src = cv.cvCreateMatHeader(height, width, cv_type);
        cv.cvSetData(src, in->host, 3*width*sizeof(uint8_t));

        auto dst = cv.cvCreateMatHeader(height, width, cv_type);
        cv.cvSetData(dst, out->host, 3*width*sizeof(uint8_t));

        cv.cvSmooth(src, dst, CV_MEDIAN, ksize, ksize, 0, 0);

        cv.cvReleaseMat(&src);
        cv.cvReleaseMat(&dst);
    }

    return 0;
}
ION_REGISTER_EXTERN(median_blur);

extern "C" ION_EXPORT
int display(halide_buffer_t *in, int width, int height, int idx, halide_buffer_t *out) {
    auto& cv(ion::bb::OpenCV::get_instance());
    if (!cv.is_available()) {
        ion::log::error("OpenCV is not available");
        return -1;
    }

    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = 3; // RGB
        in->dim[1].min = 0;
        in->dim[1].extent = width;
        in->dim[2].min = 0;
        in->dim[2].extent = height;
    } else {
        auto img = cv.cvCreateMatHeader(height, width, CV_MAKETYPE(CV_8U, 3));
        cv.cvSetData(img, in->host, 3*width*sizeof(uint8_t));

        auto name = "img" + std::to_string(idx);
        cv.cvShowImage(name.c_str(), img);
        cv.cvWaitKey(1);

        cv.cvReleaseMat(&img);
    }

    return 0;
}
ION_REGISTER_EXTERN(display);

#undef ION_EXPORT
#undef ION_REGISTER_EXTERN

#endif // ION_BB_OPENCV_RT_H
