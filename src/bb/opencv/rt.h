#ifndef ION_BB_OPENCV_RT_H
#define ION_BB_OPENCV_RT_H

#include <HalideBuffer.h>

#include "dynamic_module.h"

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

#include <Halide.h>
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

#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_16F  7

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

enum SmoothMethod_c
{
    /** linear convolution with \f$\texttt{size1}\times\texttt{size2}\f$ box kernel (all 1's). If
    you want to smooth different pixels with different-size box kernels, you can use the integral
    image that is computed using integral */
    CV_BLUR_NO_SCALE =0,
    /** linear convolution with \f$\texttt{size1}\times\texttt{size2}\f$ box kernel (all
    1's) with subsequent scaling by \f$1/(\texttt{size1}\cdot\texttt{size2})\f$ */
    CV_BLUR  =1,
    /** linear convolution with a \f$\texttt{size1}\times\texttt{size2}\f$ Gaussian kernel */
    CV_GAUSSIAN  =2,
    /** median filter with a \f$\texttt{size1}\times\texttt{size1}\f$ square aperture */
    CV_MEDIAN =3,
    /** bilateral filter with a \f$\texttt{size1}\times\texttt{size1}\f$ square aperture, color
    sigma= sigma1 and spatial sigma= sigma2. If size1=0, the aperture square side is set to
    cvRound(sigma2\*1.5)\*2+1. See cv::bilateralFilter */
    CV_BILATERAL =4
};

using CvArr = void;
using CvMat = struct CvMat;

using cvCreateMatHeader_t = CvMat*(*)(int rows, int cols, int type);
using cvReleaseMat_t = void(*)(CvMat **mat);
using cvSetData_t = void(*)(CvArr* arr, void* data, int step);
using cvSmooth_t = void (*)(const CvArr* src, CvArr* dst, int smoothtype, int size1, int size2, double sigma1, double sigma2);
using cvShowImage_t = void(*)(const char* name, const CvArr* image);
using cvWaitKey_t = int(*)(int delay);

cvCreateMatHeader_t cvCreateMatHeader;
cvReleaseMat_t cvReleaseMat;
cvSetData_t cvSetData;
cvSmooth_t cvSmooth;
cvShowImage_t cvShowImage;
cvWaitKey_t cvWaitKey;

class Initializer {
public:

    Initializer() :
#ifdef _WIN32
        // TODO: Determine OpenCV version dynamically
        opencv_world_("opencv_world455")
#else
        opencv_core_("opencv_core"),
        opencv_imgproc_("opencv_imgproc"),
        opencv_highgui_("opencv_highgui")
#endif
    {
        init_symbols();
    }

    void init_symbols() {
#ifdef _WIN32
        #define GET_SYMBOL(LOCAL_VAR, TARGET_SYMBOL)                              \
            LOCAL_VAR = opencv_world_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL);   \
            if (LOCAL_VAR == nullptr) {                                           \
                throw ::std::runtime_error(                                       \
                    TARGET_SYMBOL " is unavailable on gobject-2.0");              \
            }

        GET_SYMBOL(cvCreateMatHeader, "cvCreateMatHeader");
        GET_SYMBOL(cvReleaseMat, "cvReleaseMat");
        GET_SYMBOL(cvSetData, "cvSetData");
        GET_SYMBOL(cvSmooth, "cvSmooth");
        GET_SYMBOL(cvShowImage, "cvShowImage");
        GET_SYMBOL(cvWaitKey, "cvWaitKey");

        #undef GET_SYMBOL
#else
        #define GET_SYMBOL(MODULE, LOCAL_VAR, TARGET_SYMBOL)                         \
            LOCAL_VAR = opencv_##MODULE##_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL); \
            if (LOCAL_VAR == nullptr) {                                              \
                throw ::std::runtime_error(                                          \
                    TARGET_SYMBOL " is unavailable on gobject-2.0");                 \
            }

        GET_SYMBOL(core, cvCreateMatHeader, "cvCreateMatHeader");
        GET_SYMBOL(core, cvReleaseMat, "cvReleaseMat");
        GET_SYMBOL(core, cvSetData, "cvSetData");
        GET_SYMBOL(imgproc, cvSmooth, "cvSmooth");
        GET_SYMBOL(highgui, cvShowImage, "cvShowImage");
        GET_SYMBOL(highgui, cvWaitKey, "cvWaitKey");

        #undef GET_SYMBOL

#endif
    }

private:
#if _WIN32
    ion::DynamicModule opencv_world_;
#else
    ion::DynamicModule opencv_core_;
    ion::DynamicModule opencv_imgproc_;
    ion::DynamicModule opencv_highgui_;
#endif
} initializer;

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
int median_blur(halide_buffer_t *in, int ksize, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        for (auto i=0; i<in->dimensions; ++i) {
            in->dim[i].min = out->dim[i].min;
            in->dim[i].extent = out->dim[i].extent;
        }
    } else {
        int width = in->dim[1].extent;
        int height = in->dim[2].extent;
        int cv_type = hl2cv_type(in->type, 3);
        if (cv_type == -1) {
            return -1;
        }

        CvMat *src = cvCreateMatHeader(height, width, cv_type);
        cvSetData(src, in->host, 3*width*sizeof(uint8_t));

        CvMat *dst = cvCreateMatHeader(height, width, cv_type);
        cvSetData(dst, out->host, 3*width*sizeof(uint8_t));

        cvSmooth(src, dst, CV_MEDIAN, ksize, ksize, 0, 0);

        cvReleaseMat(&src);
        cvReleaseMat(&dst);
    }

    return 0;
}
ION_REGISTER_EXTERN(median_blur);

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
        CvMat *img = cvCreateMatHeader(height, width, CV_8UC3);
        cvSetData(img, in->host, 3*width*sizeof(uint8_t));

        auto name = "img" + std::to_string(idx);
        cvShowImage(name.c_str(), img);
        cvWaitKey(1);

        cvReleaseMat(&img);
    }

    return 0;
}
ION_REGISTER_EXTERN(display);

#undef ION_EXPORT
#undef ION_REGISTER_EXTERN

#endif // ION_BB_OPENCV_RT_H
