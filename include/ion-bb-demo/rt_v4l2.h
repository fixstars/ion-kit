#ifndef ION_BB_DEMO_RT_V4L2_H
#define ION_BB_DEMO_RT_V4L2_H

#include <cstdlib>
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
namespace demo {

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
    static V4L2 &get_instance(uint32_t index, int32_t width, int32_t height, uint32_t pixel_format) {
        if (instances_.count(index) == 0) {
            instances_[index] = std::make_shared<V4L2>(index, width, height, pixel_format);
        }
        return *instances_[index];
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
        memset(&fmtdesc,0,sizeof(fmtdesc));
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        bool supported = false;
        while (0 == xioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc))
        {
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

        /* 10-bit Bayer is stored as 16-bit so bytes per pixel is 2 */

        unsigned int min;
        min = fmt.fmt.pix.width * 2;
        if (fmt.fmt.pix.bytesperline < min) {
            fmt.fmt.pix.bytesperline = min;
        }
        min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
        if (fmt.fmt.pix.sizeimage < min) {
            fmt.fmt.pix.sizeimage = min;
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

    void get(Halide::Runtime::Buffer<uint16_t> &buf) {
        using namespace std;

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

    static std::unordered_map<uint32_t, std::shared_ptr<V4L2>> instances_;
};

std::unordered_map<uint32_t, std::shared_ptr<V4L2>> V4L2::instances_;

}  // namespace demo
}  // namespace bb
}  // namespace ion

namespace ion {
namespace bb {
namespace demo {

std::tuple<std::string, std::string> parse_url(const std::string &url) {
    auto protocol_end_pos = url.find("://");
    if (protocol_end_pos == std::string::npos) {
        return std::tuple<std::string, std::string>("", "");
    }
    auto host_name_pos = protocol_end_pos + 3;
    auto path_name_pos = url.find("/", host_name_pos);
    auto host_name = url.substr(0, path_name_pos);
    auto path_name = url.substr(path_name_pos);
    return std::tuple<std::string, std::string>(host_name, path_name);
}

std::unordered_map<int32_t, cv::Mat> camera_stub_cache;

}  // namespace demo
}  // namespace bb
}  // namespace ion

extern "C" int ION_EXPORT ion_bb_demo_v4l2(int32_t index, int32_t width, int32_t height, uint32_t pixel_format, halide_buffer_t *out) {
    try {
        auto &v4l2(ion::bb::demo::V4L2::get_instance(index, width, height, pixel_format));
        Halide::Runtime::Buffer<uint16_t> obuf(*out);

        if (v4l2.is_available()) {
            v4l2.get(obuf);
        } else {
            return -1;
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

extern "C" ION_EXPORT int ion_bb_demo_camera_stub(halide_buffer_t *url_buf, int index, int width, int height, float gain_r, float gain_g, float gain_b, float offset, int bit_width, int bit_shift, int bayer_pattern, halide_buffer_t *out) {
    auto it = ion::bb::demo::camera_stub_cache.find(index);
    if (it != ion::bb::demo::camera_stub_cache.end()) {
        memcpy(out->host, it->second.data, it->second.total() * it->second.elemSize());
        return 0;
    }

    const char *url = reinterpret_cast<const char *>(url_buf->host);

    std::string host_name;
    std::string path_name;
    std::tie(host_name, path_name) = ion::bb::demo::parse_url(url);

    cv::Mat img;
    bool img_loaded = false;
    if (host_name.empty() || path_name.empty()) {
        // fallback to local file
        img = cv::imread(url);
        if (!img.empty()) {
            img_loaded = true;
        }
    } else {
        httplib::Client cli(host_name.c_str());
        cli.set_follow_location(true);
        auto res = cli.Get(path_name.c_str());
        if (res && res->status == 200) {
            std::vector<char> data(res->body.size());
            std::memcpy(data.data(), res->body.c_str(), res->body.size());
            img = cv::imdecode(cv::InputArray(data), cv::IMREAD_COLOR);
            img_loaded = true;
        }
    }

    if (!img_loaded) {
        // Simulation mode
        cv::Mat pat = cv::Mat::zeros(2, 2, CV_16U);
        pat.at<uint16_t>((index / 2) % 2, index % 2) = 65535;

        img = cv::repeat(pat, height / 2, width / 2);

        memcpy(out->host, img.data, img.total() * img.elemSize());

        ion::bb::demo::camera_stub_cache[index] = img;

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

    memcpy(out->host, bit_shifted_img.data, bit_shifted_img.total() * bit_shifted_img.elemSize());

    ion::bb::demo::camera_stub_cache[index] = bit_shifted_img;

    return 0;
}

extern "C" int ION_EXPORT ion_bb_demo_v4l2_imx219(halide_buffer_t *url_buf, int32_t index, bool force_sim_mode, halide_buffer_t *out) {
    bool bounds_query = false;
    const int width = 3264;
    const int height = 2464;

    if (url_buf->is_bounds_query()) {
        bounds_query = true;
    }
    if (out->is_bounds_query()) {
        out->dim[0].min = 0;
        out->dim[0].extent = width;
        out->dim[1].min = 0;
        out->dim[1].extent = height;
        bounds_query = true;
    }
    if (bounds_query) {
        return 0;
    }

    int result = -1;
    if (!force_sim_mode) {
        result = ion_bb_demo_v4l2(index, width, height, V4L2_PIX_FMT_SRGGB10, out);
    }
    if (result) {
        result = ion_bb_demo_camera_stub(url_buf, index, width, height, 0.4f, 0.5f, 0.3125f, 0.0625f, 10, 6, 0 /*RGGB*/, out);
    }

    return result;
}

#endif
