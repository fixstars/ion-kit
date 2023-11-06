#ifndef ION_BB_IMAGE_IO_RT_V4L2_H
#define ION_BB_IMAGE_IO_RT_V4L2_H

#include <cstdlib>
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <errno.h>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <HalideBuffer.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"

#include "rt_common.h"
#include "httplib.h"

namespace ion {
namespace bb {
namespace image_io {

std::unordered_map<int32_t, std::vector<uint8_t>> image_cache;

}  // namespace image_io
}  // namespace bb
}  // namespace ion

namespace ion {
namespace bb {
namespace image_io {

int xioctl(int fd, int request, void *arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

class V4L2 {

    struct Buffer {
        void *start;
        size_t length;
    };

public:

    static V4L2 &get_instance(int32_t id, int32_t index, int32_t fps, int32_t width, int32_t height, uint32_t pixel_format,
                              float gain_r, float gain_g, float gain_b, float offset, int32_t bit_width, int32_t bit_shift,
                              bool force_sim_mode, const std::string& url) {
        if (instances_.count(id) == 0) {
            instances_[id] = std::make_shared<V4L2>(id, index, fps, width, height, pixel_format, gain_r, gain_g, gain_b, offset, bit_width, bit_shift, force_sim_mode, url);
        }
        return *instances_[id];
    }

    V4L2(int32_t id, int32_t index, int32_t fps, int32_t width, int32_t height, uint32_t pixel_format,
         float gain_r, float gain_g, float gain_b, float offset, int32_t bit_width, int32_t bit_shift,
         bool force_sim_mode, const std::string& url)
        : id_(id), index_(index), fps_(fps), width_(width), height_(height), pixel_format_(pixel_format),
          gain_r_(gain_r), gain_g_(gain_g), gain_b_(gain_b), offset_(offset), bit_width_(bit_width), bit_shift_(bit_shift),
          sim_mode_(force_sim_mode), url_(url) {

        using namespace std;

        //
        // Initialize device
        //
        std::string dev_name_str = "/dev/video" + std::to_string(index);
        const char *dev_name = dev_name_str.c_str();
        struct stat st;
        if (-1 == stat(dev_name, &st)) {
            log::warn("Fallback to simulation mode: Could not find {}", dev_name);
            sim_mode_ = true;;
            return;
        }
        if (!S_ISCHR(st.st_mode)) {
            log::warn("Fallback to simulation mode: {} is not proper device", dev_name);
            sim_mode_ = true;;
            return;
        }

        fd_ = open(dev_name, O_RDWR | O_NONBLOCK, 0);
        if (-1 == fd_) {
            log::warn("Fallback to simulation mode: Cannot open {}: {}, {}", dev_name, errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }

        struct v4l2_capability cap;
        if (-1 == xioctl(fd_, VIDIOC_QUERYCAP, &cap)) {
            if (EINVAL == errno) {
                log::warn("Fallback to simulation mode: {} is not V4L2 device", dev_name);
                sim_mode_ = true;;
                return;
            } else {
                log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_QUERYCAP", errno, strerror(errno));
                sim_mode_ = true;;
                return;
            }
        }
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            log::warn("Fallback to simulation mode: {} is not video capture device", dev_name);
            sim_mode_ = true;;
            return;
        }
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            log::warn("Fallback to simulation mode: {} s does not support streaming i/o", dev_name);
            sim_mode_ = true;;
            return;
        }

        uint32_t desired_pixel_format = pixel_format;

        struct v4l2_fmtdesc fmtdesc;
        memset(&fmtdesc, 0, sizeof(fmtdesc));
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        bool supported = false;
        while (0 == xioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc)) {
            if (fmtdesc.pixelformat == desired_pixel_format) {
                supported = true;
            }
            fmtdesc.index++;
        }
        if (!supported) {
            log::warn("Fallback to simulation mode: {} does not support desired pixel format", dev_name);
            sim_mode_ = true;;
            return;
        }

        struct v4l2_format fmt {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .fmt = {
                .pix = {
                    .width = static_cast<__u32>(width),
                    .height = static_cast<__u32>(height),
                    .pixelformat = pixel_format,
                    .field = V4L2_FIELD_INTERLACED,
                }},
        };
        if (-1 == xioctl(fd_, VIDIOC_S_FMT, &fmt)) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_S_FMT", errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }
        if (width != fmt.fmt.pix.width || height != fmt.fmt.pix.height) {
            log::warn("Fallback to simulation mode: {} does not support desired resolution", dev_name);
            sim_mode_ = true;;
            return;
        }
        buffer_size_ = fmt.fmt.pix.sizeimage;

        struct v4l2_streamparm strmp {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,

        };
        if (-1 == xioctl(fd_, VIDIOC_G_PARM, &strmp)) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_G_PARM", errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }
        strmp.parm.capture.timeperframe.numerator = 1;
        strmp.parm.capture.timeperframe.denominator = fps;
        if (-1 == xioctl(fd_, VIDIOC_S_PARM, &strmp)) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_S_PARM", errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }

        //
        // Initialize mapped memory
        //
        struct v4l2_requestbuffers req;
        req.count = 2;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(fd_, VIDIOC_REQBUFS, &req)) {
            if (EINVAL == errno) {
                log::warn("Fallback to simulation mode: {} does not support memory mapping", dev_name);
                sim_mode_ = true;;
                return;
            } else {
                log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_REQBUFS", errno, strerror(errno));
                sim_mode_ = true;;
                return;
            }
        }

        const size_t align_unit = 64;
        const size_t size = (fmt.fmt.pix.sizeimage + align_unit - 1) / align_unit * align_unit;
        for (int i = 0; i < req.count; ++i) {
            Buffer buffer;
            buffer.start = aligned_alloc(align_unit, fmt.fmt.pix.sizeimage);
            buffer.length = fmt.fmt.pix.sizeimage;

            buffers_.push_back(buffer);
        }

        //
        // Start capture
        //
        for (int i = 0; i < buffers_.size() - 1; ++i) {
            struct v4l2_buffer buf;
            buf.index = static_cast<__u32>(i);
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_USERPTR;
            buf.m.userptr = reinterpret_cast<unsigned long>(buffers_[i].start);
            buf.length = buffers_[i].length;

            /* enqueue an empty (capturing) or filled (output) buffer in the driver's incoming queue */
            if (-1 == xioctl(fd_, VIDIOC_QBUF, &buf)) {
                log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_QBUF", errno, strerror(errno));
                sim_mode_ = true;;
                return;
            }
        }
        next_buffer_.index = static_cast<__u32>(buffers_.size() - 1);
        next_buffer_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        next_buffer_.memory = V4L2_MEMORY_USERPTR;
        next_buffer_.m.userptr = reinterpret_cast<unsigned long>(buffers_[buffers_.size() - 1].start);
        next_buffer_.length = buffers_[buffers_.size() - 1].length;

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        /* Start streaming I/O */
        if (-1 == xioctl(fd_, VIDIOC_STREAMON, &type)) {
            log::warn("Fallback to simulation mode: {} error {}, {}\n", "VIDIOC_STREAMON", errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }

        //
        // Initialize event
        //
        efd_ = epoll_create1(0);
        if (-1 == efd_) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "epoll_create1", errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }

        struct epoll_event event;
        event.events = EPOLLIN | EPOLLET;
        event.data.fd = fd_;

        if (-1 == epoll_ctl(efd_, EPOLL_CTL_ADD, fd_, &event)) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "epoll_ctl", errno, strerror(errno));
            sim_mode_ = true;;
            return;
        }
    }

    // 0: RGGB, 1:BGGR, 2: GRBG, 3:GBRG
    int bayer_pattern(uint32_t pixel_format) {
        switch (pixel_format) {
        case V4L2_PIX_FMT_SRGGB8:
        case V4L2_PIX_FMT_SRGGB10:
        case V4L2_PIX_FMT_SRGGB12:
            return 0;
        case V4L2_PIX_FMT_SBGGR8:
        case V4L2_PIX_FMT_SBGGR10:
        case V4L2_PIX_FMT_SBGGR12:
            return 1;
        case V4L2_PIX_FMT_SGRBG8:
        case V4L2_PIX_FMT_SGRBG10:
        case V4L2_PIX_FMT_SGRBG12:
            return 2;
        case V4L2_PIX_FMT_SGBRG8:
        case V4L2_PIX_FMT_SGBRG10:
        case V4L2_PIX_FMT_SGBRG12:
            return 3;
        deafult:
            throw std::runtime_error("Unreachable");
        }

        return -1;
    }

    template<typename T>
    void generate_bayer(Halide::Runtime::Buffer<T> &buf) {
        auto it = ion::bb::image_io::image_cache.find(id_);
        if (it != ion::bb::image_io::image_cache.end()) {
            memcpy(buf.data(), it->second.data(), it->second.size());
            return;
        }

        cv::Mat img = ion::bb::image_io::get_image(url_);

        if (img.empty()) {
            // Fill by dummy image
            cv::Mat pat = cv::Mat::zeros(2, 2, CV_16U);
            pat.at<uint16_t>((index_ / 2) % 2, index_ % 2) = 65535;

            img = cv::repeat(pat, height_ / 2, width_ / 2);

            std::vector<uint8_t> data(img.total() * img.elemSize());
            memcpy(data.data(), img.data, img.total() * img.elemSize());
            memcpy(buf.data(), img.data, img.total() * img.elemSize());

            ion::bb::image_io::image_cache[id_] = data;

            return;
        }

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(width_, height_));

        cv::Mat normalized_img;
        resized_img.convertTo(normalized_img, CV_32F, 1 / 255.f);

        cv::Mat linear_img;
        cv::pow(normalized_img, 2.2, linear_img);

        std::vector<cv::Mat> splitted_image(3);
        cv::split(linear_img, splitted_image);

        cv::Mat processed_img_r, processed_img_g, processed_img_b;
        processed_img_r = min(max(splitted_image[2] * gain_r_ + offset_, 0), 1);
        processed_img_g = min(max(splitted_image[1] * gain_g_ + offset_, 0), 1);
        processed_img_b = min(max(splitted_image[0] * gain_b_ + offset_, 0), 1);

        cv::Mat mask_00 = cv::repeat((cv::Mat_<float>(2, 2) << 1, 0, 0, 0), height_ / 2, width_ / 2);
        cv::Mat mask_10 = cv::repeat((cv::Mat_<float>(2, 2) << 0, 1, 0, 0), height_ / 2, width_ / 2);
        cv::Mat mask_01 = cv::repeat((cv::Mat_<float>(2, 2) << 0, 0, 1, 0), height_ / 2, width_ / 2);
        cv::Mat mask_11 = cv::repeat((cv::Mat_<float>(2, 2) << 0, 0, 0, 1), height_ / 2, width_ / 2);

        cv::Mat processed_img;
        switch (bayer_pattern(pixel_format_)) {
        case 0:  // RGGB
            processed_img = processed_img_r.mul(mask_00) + processed_img_g.mul(mask_01 + mask_10) + processed_img_b.mul(mask_11);
            break;
        case 1:  // BGGR
            processed_img = processed_img_r.mul(mask_11) + processed_img_g.mul(mask_01 + mask_10) + processed_img_b.mul(mask_00);
            break;
        case 2:  // GRBG
            processed_img = processed_img_r.mul(mask_10) + processed_img_g.mul(mask_00 + mask_11) + processed_img_b.mul(mask_01);
            break;
        case 3:  // GBRG
            processed_img = processed_img_r.mul(mask_01) + processed_img_g.mul(mask_00 + mask_11) + processed_img_b.mul(mask_10);
            break;
        }

        cv::Mat denormalized_img;
        processed_img.convertTo(denormalized_img, CV_16U, (1 << bit_width_) - 1);

        cv::Mat bit_shifted_img;
        bit_shifted_img = denormalized_img * (1 << bit_shift_) + denormalized_img / (1 << (bit_width_ - bit_shift_));

        std::vector<uint8_t> data(bit_shifted_img.total() * bit_shifted_img.elemSize());
        memcpy(data.data(), bit_shifted_img.data, bit_shifted_img.total() * bit_shifted_img.elemSize());
        memcpy(buf.data(), bit_shifted_img.data, bit_shifted_img.total() * bit_shifted_img.elemSize());

        ion::bb::image_io::image_cache[id_] = data;
    }

    template<typename T>
    void generate_yuyv(Halide::Runtime::Buffer<T> &buf) {

        auto it = ion::bb::image_io::image_cache.find(id_);
        if (it != ion::bb::image_io::image_cache.end()) {
            memcpy(buf.data(), it->second.data(), it->second.size());
            return;
        }

        cv::Mat img = ion::bb::image_io::get_image(url_);

        std::vector<uint8_t> yuyv_img(2 * width_ * height_);

        if (img.empty()) {
            // Fill by dummy image
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < 2 * width_; ++x) {
                    yuyv_img[2 * width_ * y + x] = (y * 2 * width_ + x) % 255;
                }
            }
        } else {
            cv::resize(img, img, cv::Size(width_, height_));
            cv::cvtColor(img, img, cv::COLOR_BGR2YCrCb);
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    // Y
                    yuyv_img[2 * width_ * y + 2 * x + 0] = img.at<cv::Vec3b>(y, x)[0];
                    // Cb or Cr
                    yuyv_img[2 * width_ * y + 2 * x + 1] = ((x % 2) == 1) ? img.at<cv::Vec3b>(y, x)[1] : img.at<cv::Vec3b>(y, x)[2];
                }
            }
        }
        memcpy(buf.data(), yuyv_img.data(), yuyv_img.size());
        ion::bb::image_io::image_cache[id_] = yuyv_img;

        return;
    }

    template<typename T>
    void generate(Halide::Runtime::Buffer<T> &buf) {

        // Simulate frame interval
        auto now = std::chrono::high_resolution_clock::now();
        auto actual_interval = std::chrono::duration_cast<std::chrono::microseconds>(now - checkpoint_).count();
        float expected_interval = 1e6f / static_cast<float>(fps_);
        if (actual_interval < expected_interval) {
            usleep(expected_interval - actual_interval);
        }

        if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
            generate_yuyv(buf);
        } else {
            generate_bayer(buf);
        }

        checkpoint_ = std::chrono::high_resolution_clock::now();
    }

    template<typename T>
    void get(Halide::Runtime::Buffer<T> &buf) {
        using namespace std;

        if (sim_mode_) {
            generate(buf);
            return;
        }

        if (buf.size_in_bytes() != buffer_size_) {
            throw runtime_error("Bad buffer size");
        }

        /* queue-in buffer */
        if (-1 == xioctl(fd_, VIDIOC_QBUF, &next_buffer_)) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_QBUF", errno, strerror(errno));
            sim_mode_ = true;
            return;
        }

        epoll_event event;
        if (-1 == epoll_wait(efd_, &event, 1, -1)) {
            log::warn("Fallback to simulation mode: {} error {}, {}", "epoll_wait", errno, strerror(errno));
            sim_mode_ = true;
            return;
        }

        if (event.data.fd != fd_) {
            throw runtime_error("Unreachable");
        }

        struct v4l2_buffer v4l2_buf;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_buf.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(fd_, VIDIOC_DQBUF, &next_buffer_)) {
            if (EAGAIN == errno) {
                return;
            } else {
                log::warn("Fallback to simulation mode: {} error {}, {}", "VIDIOC_DQBUF", errno, strerror(errno));
                sim_mode_ = true;
                return;
            }
        }

        memcpy(buf.data(), reinterpret_cast<void *>(next_buffer_.m.userptr), buf.size_in_bytes());
    }


private:
    int fd_;
    std::vector<Buffer> buffers_;
    v4l2_buffer next_buffer_;

    int32_t id_;
    int32_t index_;
    int32_t fps_;
    int32_t width_;
    int32_t height_;
    uint32_t pixel_format_;
    float gain_r_;
    float gain_g_;
    float gain_b_;
    float offset_;
    int32_t bit_width_;
    int32_t bit_shift_;
    bool sim_mode_;
    std::string url_;

    int efd_;

    uint32_t buffer_size_;

    std::chrono::time_point<std::chrono::high_resolution_clock> checkpoint_;

    static std::unordered_map<int32_t, std::shared_ptr<V4L2>> instances_;
};

std::unordered_map<int32_t, std::shared_ptr<V4L2>> V4L2::instances_;

}  // namespace image_io
}  // namespace bb
}  // namespace ion

extern "C" ION_EXPORT int ion_bb_image_io_v4l2(
    int32_t instance_id,
    // Parameters for V4L2
    int32_t index,
    int32_t fps,
    int32_t width, int32_t height,
    uint32_t pixel_format,
    uint32_t force_sim_mode, // Do not use bool to avoid LLVM codegen failure
    // Parameters for simulation
    halide_buffer_t *url_buf,
    float gain_r, float gain_g, float gain_b,
    float offset,
    int32_t bit_width, int32_t bit_shift,
    halide_buffer_t *out) {

    try {

        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = width;
            out->dim[1].min = 0;
            out->dim[1].extent = height;
            return 0;
        }

        auto &v4l2(ion::bb::image_io::V4L2::get_instance(instance_id, index, fps, width, height, pixel_format, gain_r, gain_g, gain_b, offset, bit_width, bit_shift, static_cast<bool>(force_sim_mode), reinterpret_cast<const char*>(url_buf->host)));
        Halide::Runtime::Buffer<uint16_t> obuf(*out);
        v4l2.get(obuf);

        return 0;

    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_v4l2)

extern "C" int ION_EXPORT ion_bb_image_io_camera(int32_t instance_id, int32_t index, int32_t fps, int32_t width, int32_t height, halide_buffer_t *url_buf, halide_buffer_t *out) {
    try {
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = 2 * width;  // YUYV
            out->dim[1].min = 0;
            out->dim[1].extent = height;
            return 0;
        }

        auto &v4l2(ion::bb::image_io::V4L2::get_instance(instance_id, index, fps, width, height, V4L2_PIX_FMT_YUYV, 1, 1, 1, 0, 8, 0, false, reinterpret_cast<const char*>(url_buf->host)));
        Halide::Runtime::Buffer<uint8_t> obuf(*out);
        v4l2.get(obuf);

        return 0;

    } catch (const std::exception &e) {
        ion::log::error("Exception was thrown: {}", e.what());
        return 1;
    } catch (...) {
        ion::log::error("Unknown exception was thrown");
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_image_io_camera)

#endif
