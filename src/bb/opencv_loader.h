#ifndef ION_BB_OPENCV_LOADER_H
#define ION_BB_OPENCV_LOADER_H

#include "dynamic_module.h"

namespace {

#define CV_CN_MAX 512
#define CV_CN_SHIFT 3
#define CV_DEPTH_MAX (1 << CV_CN_SHIFT)

#define CV_8U 0
#define CV_8S 1
#define CV_16U 2
#define CV_16S 3
#define CV_32S 4
#define CV_32F 5
#define CV_64F 6
#define CV_16F 7

#define CV_MAT_DEPTH_MASK (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags) ((flags)&CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth, cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define IMREAD_GRAYSCALE 0
#define IMREAD_COLOR 1

enum SmoothMethod_c {
    /** linear convolution with \f$\texttt{size1}\times\texttt{size2}\f$ box kernel (all 1's). If
    you want to smooth different pixels with different-size box kernels, you can use the integral
    image that is computed using integral */
    CV_BLUR_NO_SCALE = 0,
    /** linear convolution with \f$\texttt{size1}\times\texttt{size2}\f$ box kernel (all
    1's) with subsequent scaling by \f$1/(\texttt{size1}\cdot\texttt{size2})\f$ */
    CV_BLUR = 1,
    /** linear convolution with a \f$\texttt{size1}\times\texttt{size2}\f$ Gaussian kernel */
    CV_GAUSSIAN = 2,
    /** median filter with a \f$\texttt{size1}\times\texttt{size1}\f$ square aperture */
    CV_MEDIAN = 3,
    /** bilateral filter with a \f$\texttt{size1}\times\texttt{size1}\f$ square aperture, color
    sigma= sigma1 and spatial sigma= sigma2. If size1=0, the aperture square side is set to
    cvRound(sigma2\*1.5)\*2+1. See cv::bilateralFilter */
    CV_BILATERAL = 4
};

}  // namespace

namespace ion {
namespace bb {

class OpenCV {

    using CvArr = void;
    using CvMat = struct CvMat;
    // To fo, remove c api function later, since it is deprecated
    using cvCreateMatHeader_t = CvMat *(*)(int rows, int cols, int type);
    using cvReleaseMat_t = void (*)(CvMat **mat);
    using cvSetData_t = void (*)(CvArr *arr, void *data, int step);
    using cvSplit_t = void (*)(const CvArr *src, CvArr *dst0, CvArr *dst1, CvArr *dst2, CvArr *dst3);
    using cvPow_t = void (*)(const CvArr *src, CvArr *dst, double power);
    using cvRepeat_t = void (*)(const CvArr *src, CvArr *dst);
    using cvSmooth_t = void (*)(const CvArr *src, CvArr *dst, int smoothtype, int size1, int size2, double sigma1, double sigma2);
    using cvShowImage_t = void (*)(const char *name, const CvArr *image);
    using cvCvtColor_t = void (*)(const CvArr *src, CvArr *dst, int code);
    //   using cvSaveImage_t = void(*)(const char *filename, const CvArr *image, const int *params);
    //    using cvLoadImageM_t =CvMat*(*)(const char* filename, int iscolor);
    using cvWaitKey_t = int (*)(int delay);

    using cvResize_t = void (*)(const CvArr *src, CvArr *dst, int interpolation);
    using cvNormalize_t = void (*)(const CvArr *src, CvArr *dst, double alpha, double beta, int normtype, int dtype, const CvArr *mask);

public:
    static OpenCV &get_instance() {
        static OpenCV instance;
        return instance;
    }

    OpenCV()
        :
#ifdef _WIN32
          // TODO: Determine OpenCV version dynamically
          opencv_world_("opencv_world455", false)
#else
          opencv_core_("opencv_core", false),
          opencv_imgproc_("opencv_imgproc", false),
          opencv_highgui_("opencv_highgui", false)
#endif
    {
        if (is_available()) {
            init_symbols();
        } else {
        }
    }

    void init_symbols() {
#ifdef _WIN32
#define GET_SYMBOL(LOCAL_VAR, TARGET_SYMBOL)                            \
    LOCAL_VAR = opencv_world_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL); \
    if (LOCAL_VAR == nullptr) {                                         \
        throw ::std::runtime_error(                                     \
            TARGET_SYMBOL " is unavailable on opencv_world");           \
    }

        GET_SYMBOL(cvCreateMatHeader, "cvCreateMatHeader");
        GET_SYMBOL(cvReleaseMat, "cvReleaseMat");
        GET_SYMBOL(cvSetData, "cvSetData");
        GET_SYMBOL(cvSplit, "cvSplit");
        GET_SYMBOL(cvPow, "cvPow");
        GET_SYMBOL(cvRepeat, "cvRepeat");
        GET_SYMBOL(cvSmooth, "cvSmooth");
        GET_SYMBOL(cvResize, "cvResize");
        GET_SYMBOL(cvCvtColor, "cvCvtColor");
        GET_SYMBOL(cvShowImage, "cvShowImage");
        GET_SYMBOL(cvWaitKey, "cvWaitKey");

#undef GET_SYMBOL
#else
#define GET_SYMBOL(MODULE, LOCAL_VAR, TARGET_SYMBOL)                         \
    LOCAL_VAR = opencv_##MODULE##_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL); \
    if (LOCAL_VAR == nullptr) {                                              \
        throw ::std::runtime_error(                                          \
            TARGET_SYMBOL " is unavailable on " #MODULE);                    \
    }

        GET_SYMBOL(core, cvCreateMatHeader, "cvCreateMatHeader");
        GET_SYMBOL(core, cvReleaseMat, "cvReleaseMat");
        GET_SYMBOL(core, cvSetData, "cvSetData");
        GET_SYMBOL(core, cvSplit, "cvSplit");
        GET_SYMBOL(core, cvPow, "cvPow");
        GET_SYMBOL(core, cvRepeat, "cvRepeat");
        GET_SYMBOL(imgproc, cvSmooth, "cvSmooth");
        GET_SYMBOL(imgproc, cvResize, "cvResize");
        GET_SYMBOL(imgproc, cvNormalize, "cvNormalize");  // obsolete
        GET_SYMBOL(imgproc, cvCvtColor, "cvCvtColor");
        GET_SYMBOL(highgui, cvShowImage, "cvShowImage");
        GET_SYMBOL(highgui, cvWaitKey, "cvWaitKey");
#undef GET_SYMBOL

#endif
    }

    bool is_available(void) const {
#if _WIN32
        return opencv_world_.is_available();
#else
        return opencv_core_.is_available() && opencv_imgproc_.is_available() && opencv_highgui_.is_available();
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

public:
    cvCreateMatHeader_t cvCreateMatHeader;
    cvReleaseMat_t cvReleaseMat;
    cvSetData_t cvSetData;
    cvSplit_t cvSplit;
    cvPow_t cvPow;
    cvRepeat_t cvRepeat;
    cvSmooth_t cvSmooth;
    cvResize_t cvResize;
    cvNormalize_t cvNormalize;
    cvShowImage_t cvShowImage;
    cvWaitKey_t cvWaitKey;
    cvCvtColor_t cvCvtColor;
};

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

}  // namespace bb
}  // namespace ion

#endif
