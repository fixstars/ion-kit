#ifndef ION_BB_IMAGE_IO_RT_DISPLAY_H
#define ION_BB_IMAGE_IO_RT_DISPLAY_H

#include <stdexcept>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#endif

#include <HalideBuffer.h>

#include "opencv_loader.h"
#include "rt_common.h"

#ifdef __linux__
namespace {

class FBDev {

public:
    FBDev(int32_t width, int32_t height)
        : width_(width), height_(height), device_is_available_(true) {

        using namespace std;

        const char *dev_name = "/dev/fb0";

        fd_ = open(dev_name, O_RDWR);
        if (fd_ == -1) {
            device_is_available_ = false;
            return;
        }

        // Get fixed screen information
        if (ioctl(fd_, FBIOGET_FSCREENINFO, &finfo_) == -1) {
            throw runtime_error("Error reading fixed information");
        }

        // Get variable screen information
        if (ioctl(fd_, FBIOGET_VSCREENINFO, &vinfo_) == -1) {
            throw runtime_error("Error reading variable information");
        }

        // std::cout << format("xoffset: %d, yoffset: %d\nxres: %d, yres: %d\nbits_per_pixel: %d, line_length: %d\n",
        //                    vinfo_.xoffset, vinfo_.yoffset, vinfo_.xres, vinfo_.yres, vinfo_.bits_per_pixel, finfo_.line_length);

        unsigned int screensize = (vinfo_.xres * vinfo_.yres * vinfo_.bits_per_pixel) >> 3; /* (>>3): bits to bytes */

        fbp_ = reinterpret_cast<uint8_t *>(mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (fbp_ == reinterpret_cast<uint8_t *>(-1)) {
            throw runtime_error("Error mapping framebuffer device to memory");
        }
    }

    void put(Halide::Runtime::Buffer<uint8_t> &buf) {
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                unsigned int location = (x + vinfo_.xoffset) * (vinfo_.bits_per_pixel >> 3) + (y + vinfo_.yoffset) * finfo_.line_length;
                fbp_[location + 0] = buf(2, x, y);  // B
                fbp_[location + 1] = buf(1, x, y);  // G
                fbp_[location + 2] = buf(0, x, y);  // R
            }
        }
    }

    bool is_available() {
        return device_is_available_;
    }

private:
    int32_t width_;
    int32_t height_;
    int fd_;
    uint8_t *fbp_;
    struct fb_fix_screeninfo finfo_;
    struct fb_var_screeninfo vinfo_;
    bool device_is_available_;
};

}  // namespace

extern "C" int ION_EXPORT ion_bb_image_io_fb_display(int32_t width, int32_t height, halide_buffer_t *in, halide_buffer_t *out) {
    try {
        static FBDev fbdev(width, height);
        if (in->is_bounds_query() || out->is_bounds_query()) {
            if (in->is_bounds_query()) {
                in->dim[0].min = 0;
                in->dim[0].extent = 3;
                in->dim[1].min = 0;
                in->dim[1].extent = width;
                in->dim[2].min = 0;
                in->dim[2].extent = height;
            }
        } else {
            Halide::Runtime::Buffer<uint8_t> ibuf(*in);
            if (fbdev.is_available()) {
                fbdev.put(ibuf);
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown" << std::endl;
        return -1;
    }
}
#endif

extern "C" ION_EXPORT int ion_bb_image_io_gui_display(halide_buffer_t *in, int width, int height, int idx, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = 3;  // RGB
        in->dim[1].min = 0;
        in->dim[1].extent = width;
        in->dim[2].min = 0;
        in->dim[2].extent = height;
    } else {
        if (getenv("DISPLAY")) {
            auto &cv(ion::bb::OpenCV::get_instance());
            Halide::Runtime::Buffer<uint8_t> ibuf(*in);
            ibuf.copy_to_host();

            auto img = cv.cvCreateMatHeader(height, width, CV_MAKETYPE(CV_8U, 3));
            cv.cvSetData(img, in->host, 3 * width * sizeof(uint8_t));

            auto name = "img" + std::to_string(idx);
            cv.cvShowImage(name.c_str(), img);
            cv.cvWaitKey(1);

            cv.cvReleaseMat(&img);
        } else {
            // This is shimulation mode. Just sleep 1/1000 second.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    return 0;
}
ION_REGISTER_EXTERN(ion_bb_image_io_gui_display);

#endif
