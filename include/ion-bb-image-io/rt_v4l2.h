#ifndef ION_BB_IMAGE_IO_RT_V4L2_H
#define ION_BB_IMAGE_IO_RT_V4L2_H

#include <cstdlib>
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

#include "rt_common.h"

#include "httplib.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    struct Setting {
        uint32_t index;
        int32_t width;
        int32_t height;
        uint32_t pixel_format;

        bool operator<(const Setting &s) const {
            if (index < s.index) {
                return true;
            } else if (index > s.index) {
                return false;
            }
            if (width < s.width) {
                return true;
            } else if (width > s.width) {
                return false;
            }
            if (height < s.height) {
                return true;
            } else if (height > s.height) {
                return false;
            }
            if (pixel_format < s.pixel_format) {
                return true;
            } else {
                return false;
            }
        }
    };

    static V4L2 &get_instance(uint32_t index, int32_t width, int32_t height, uint32_t pixel_format) {
        Setting setting;
        setting.index = index;
        setting.width = width;
        setting.height = height;
        setting.pixel_format = pixel_format;
        if (instances_.count(setting) == 0) {
            instances_[setting] = std::make_shared<V4L2>(index, width, height, pixel_format);
        }
        return *instances_[setting];
    }

    V4L2(uint32_t index, int32_t width, int32_t height, uint32_t pixel_format)
        : device_is_available_(true) {
        using namespace std;

        //
        // Initialize device
        //
        std::string dev_name_str = "/dev/video" + std::to_string(index);
        const char *dev_name = dev_name_str.c_str();
        struct stat st;
        if (-1 == stat(dev_name, &st)) {
            device_is_available_ = false;
            return;
        }
        if (!S_ISCHR(st.st_mode)) {
            std::cerr << format("%s is no device", dev_name) << std::endl;
            device_is_available_ = false;
            return;
        }

        fd_ = open(dev_name, O_RDWR | O_NONBLOCK, 0);
        if (-1 == fd_) {
            std::cerr << format("Cannot open '%s': %d, %s", dev_name, errno, strerror(errno)) << std::endl;
            device_is_available_ = false;
            return;
        }

        struct v4l2_capability cap;
        if (-1 == xioctl(fd_, VIDIOC_QUERYCAP, &cap)) {
            if (EINVAL == errno) {
                std::cerr << format("%s is no V4L2 device", dev_name) << std::endl;
                device_is_available_ = false;
                return;
            } else {
                std::cerr << format("%s error %d, %s\n", "VIDIOC_QUERYCAP", errno, strerror(errno)) << std::endl;
                device_is_available_ = false;
                return;
            }
        }
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            std::cerr << format("%s is no video capture device", dev_name) << std::endl;
            device_is_available_ = false;
            return;
        }
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            std::cerr << format("%s does not support streaming i/o", dev_name) << std::endl;
            device_is_available_ = false;
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
            std::cerr << format("%s does not support desired pixel format", dev_name) << std::endl;
            device_is_available_ = false;
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
            std::cerr << format("%s error %d, %s\n", "VIDIOC_S_FMT", errno, strerror(errno)) << std::endl;
            device_is_available_ = false;
            return;
        }
        if (width != fmt.fmt.pix.width || height != fmt.fmt.pix.height) {
            std::cerr << format("%s does not support desired resolution", dev_name) << std::endl;
            device_is_available_ = false;
            return;
        }

        buffer_size = fmt.fmt.pix.sizeimage;

        //
        // Initialize mapped memory
        //
        struct v4l2_requestbuffers req;
        req.count = 2;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(fd_, VIDIOC_REQBUFS, &req)) {
            if (EINVAL == errno) {
                std::cerr << format("%s does not support memory mapping\n", dev_name) << std::endl;
                device_is_available_ = false;
                return;
            } else {
                std::cerr << format("%s error %d, %s\n", "VIDIOC_REQBUFS", errno, strerror(errno)) << std::endl;
                device_is_available_ = false;
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
                std::cerr << format("%s error %d, %s\n", "VIDIOC_QBUF", errno, strerror(errno)) << std::endl;
                device_is_available_ = false;
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
            std::cerr << format("%s error %d, %s\n", "VIDIOC_STREAMON", errno, strerror(errno)) << std::endl;
            device_is_available_ = false;
            return;
        }

        //
        // Initialize event
        //
        efd_ = epoll_create1(0);
        if (-1 == efd_) {
            std::cerr << format("%s error %d, %s\n", "epoll_create1", errno, strerror(errno)) << std::endl;
            device_is_available_ = false;
            return;
        }

        struct epoll_event event;
        event.events = EPOLLIN | EPOLLET;
        event.data.fd = fd_;

        if (-1 == epoll_ctl(efd_, EPOLL_CTL_ADD, fd_, &event)) {
            std::cerr << format("%s error %d, %s\n", "epoll_ctl", errno, strerror(errno)) << std::endl;
            device_is_available_ = false;
            return;
        }
    }

    template<typename T>
    void get(Halide::Runtime::Buffer<T> &buf) {
        using namespace std;

        if (buf.size_in_bytes() != buffer_size) {
            throw runtime_error("Bad buffer size");
        }

        /* queue-in buffer */
        if (-1 == xioctl(fd_, VIDIOC_QBUF, &next_buffer_)) {
            throw runtime_error(format("%s error %d, %s\n", "VIDIOC_QBUF", errno, strerror(errno)));
        }

        epoll_event event;
        if (-1 == epoll_wait(efd_, &event, 1, -1)) {
            throw runtime_error(format("%s error %d, %s\n", "epoll_wait", errno, strerror(errno)));
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
                throw runtime_error(format("%s error %d, %s\n", "VIDIOC_DQBUF", errno, strerror(errno)));
            }
        }

        memcpy(buf.data(), reinterpret_cast<void *>(next_buffer_.m.userptr), buf.size_in_bytes());
    }

    void dispose() {
    }

    bool is_available() {
        return device_is_available_;
    }

private:
    int fd_;
    std::vector<Buffer> buffers_;
    v4l2_buffer next_buffer_;
    bool device_is_available_;

    int efd_;

    uint32_t buffer_size;

    static std::map<Setting, std::shared_ptr<V4L2>> instances_;
};

std::map<V4L2::Setting, std::shared_ptr<V4L2>> V4L2::instances_;

}  // namespace image_io
}  // namespace bb
}  // namespace ion

namespace ion {
namespace bb {
namespace image_io {

std::unordered_map<int32_t, std::vector<uint8_t>> image_cache;

}  // namespace image_io
}  // namespace bb
}  // namespace ion

extern "C" ION_EXPORT int ion_bb_image_io_v4l2(
    int32_t instance_id,
    // Parameters for V4L2
    int32_t width, int32_t height,
    int32_t index, uint32_t pixel_format,
    bool force_sim_mode,
    // Parameters for simulation
    halide_buffer_t *url_buf,
    float gain_r, float gain_g, float gain_b,
    float offset,
    int32_t bit_width, int32_t bit_shift,
    int32_t bayer_pattern,
    halide_buffer_t *out) {
    if (out->is_bounds_query()) {
        out->dim[0].min = 0;
        out->dim[0].extent = width;
        out->dim[1].min = 0;
        out->dim[1].extent = height;
        return 0;
    }

    if (!force_sim_mode) {
        auto &v4l2(ion::bb::image_io::V4L2::get_instance(index, width, height, pixel_format));
        if (v4l2.is_available()) {
            Halide::Runtime::Buffer<uint16_t> obuf(*out);
            v4l2.get(obuf);
            return 0;
        }
    }

    auto it = ion::bb::image_io::image_cache.find(instance_id);
    if (it != ion::bb::image_io::image_cache.end()) {
        memcpy(out->host, it->second.data(), it->second.size());
        return 0;
    }

    cv::Mat img = ion::bb::image_io::get_image(reinterpret_cast<const char *>(url_buf->host));

    if (img.empty()) {
        // Simulation mode
        cv::Mat pat = cv::Mat::zeros(2, 2, CV_16U);
        pat.at<uint16_t>((index / 2) % 2, index % 2) = 65535;

        img = cv::repeat(pat, height / 2, width / 2);

        std::vector<uint8_t> data(img.total() * img.elemSize());
        memcpy(data.data(), img.data, img.total() * img.elemSize());
        memcpy(out->host, img.data, img.total() * img.elemSize());

        ion::bb::image_io::image_cache[instance_id] = data;

        return 0;
    }

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(width, height));

    cv::Mat normalized_img;
    resized_img.convertTo(normalized_img, CV_32F, 1 / 255.f);

    cv::Mat linear_img;
    cv::pow(normalized_img, 2.2, linear_img);

    std::vector<cv::Mat> splitted_image(3);
    cv::split(linear_img, splitted_image);

    cv::Mat processed_img_r, processed_img_g, processed_img_b;
    processed_img_r = min(max(splitted_image[2] * gain_r + offset, 0), 1);
    processed_img_g = min(max(splitted_image[1] * gain_g + offset, 0), 1);
    processed_img_b = min(max(splitted_image[0] * gain_b + offset, 0), 1);

    cv::Mat mask_00 = cv::repeat((cv::Mat_<float>(2, 2) << 1, 0, 0, 0), height / 2, width / 2);
    cv::Mat mask_10 = cv::repeat((cv::Mat_<float>(2, 2) << 0, 1, 0, 0), height / 2, width / 2);
    cv::Mat mask_01 = cv::repeat((cv::Mat_<float>(2, 2) << 0, 0, 1, 0), height / 2, width / 2);
    cv::Mat mask_11 = cv::repeat((cv::Mat_<float>(2, 2) << 0, 0, 0, 1), height / 2, width / 2);

    cv::Mat processed_img;
    switch (bayer_pattern) {
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
    processed_img.convertTo(denormalized_img, CV_16U, (1 << bit_width) - 1);

    cv::Mat bit_shifted_img;
    bit_shifted_img = denormalized_img * (1 << bit_shift) + denormalized_img / (1 << (bit_width - bit_shift));

    std::vector<uint8_t> data(bit_shifted_img.total() * bit_shifted_img.elemSize());
    memcpy(data.data(), bit_shifted_img.data, bit_shifted_img.total() * bit_shifted_img.elemSize());
    memcpy(out->host, bit_shifted_img.data, bit_shifted_img.total() * bit_shifted_img.elemSize());

    ion::bb::image_io::image_cache[instance_id] = data;

    return 0;
}

extern "C" int ION_EXPORT ion_bb_image_io_camera(int32_t instance_id, int32_t width, int32_t height, int32_t index, halide_buffer_t *url_buf, halide_buffer_t *out) {
    if (out->is_bounds_query()) {
        out->dim[0].min = 0;
        out->dim[0].extent = 2 * width;  // YUYV
        out->dim[1].min = 0;
        out->dim[1].extent = height;
        return 0;
    }

    auto &v4l2(ion::bb::image_io::V4L2::get_instance(index, width, height, V4L2_PIX_FMT_YUYV));
    if (v4l2.is_available()) {
        Halide::Runtime::Buffer<uint16_t> obuf(*out);
        v4l2.get(obuf);
        return 0;
    }

    auto it = ion::bb::image_io::image_cache.find(instance_id);
    if (it != ion::bb::image_io::image_cache.end()) {
        memcpy(out->host, it->second.data(), it->second.size());
        return 0;
    }

    cv::Mat img = ion::bb::image_io::get_image(reinterpret_cast<const char *>(url_buf->host));

    std::vector<uint8_t> yuyv_img(2 * width * height);
    if (img.empty()) {
        // Simulation mode
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < 2 * width; ++x) {
                yuyv_img[2 * width * y + x] = (y * 2 * width + x) % 255;
            }
        }
    } else {
        cv::resize(img, img, cv::Size(width, height));
        cv::cvtColor(img, img, cv::COLOR_BGR2YCrCb);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Y
                yuyv_img[2 * width * y + 2 * x + 0] = img.at<cv::Vec3b>(y, x)[0];
                // Cb or Cr
                yuyv_img[2 * width * y + 2 * x + 1] = ((x % 2) == 1) ? img.at<cv::Vec3b>(y, x)[1] : img.at<cv::Vec3b>(y, x)[2];
            }
        }
    }
    memcpy(out->host, yuyv_img.data(), yuyv_img.size());
    ion::bb::image_io::image_cache[instance_id] = yuyv_img;

    return 0;
}

#endif
