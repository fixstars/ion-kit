#ifndef ION_BB_IMAGE_IO_RT_V4L2_H
#define ION_BB_IMAGE_IO_RT_V4L2_H

// TODO: Remove OpenCV build dependency

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
#include "halide_image_io.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "log.h"

#include "rt_common.h"
#include "httplib.h"

namespace ion {
namespace bb {
namespace image_io {
#define BORDER_INTERPOLATE(x, l) (x < 0 ? 0 : (x >= l ? l - 1 : x))
float weight(float input){
            float alpha = -1;
            float x = (input < 0)? -input : input;
            float x2 = x * x;
            float x3 = x * x * x;

            if(x <= 1){
                return (alpha + 2) * x3 - (alpha + 3) * x2 + 1;
            }else if(x < 2){
                return alpha * x3 - 5 * alpha * x2 + 8 * alpha * x - 4 * alpha;
            }else{
                return 0x0;
            }
        }
template<typename T>
void resize_bicubic(Halide::Runtime::Buffer<T>& dst,
                                            const Halide::Runtime::Buffer<T>& src,
                                            const int32_t src_width, const int32_t src_height,
                                            const uint32_t dst_width, const uint32_t dst_height){
    double min_value = static_cast<double>(std::numeric_limits<T>::min());
    double max_value = static_cast<double>(std::numeric_limits<T>::max());

    for(int dh = 0; dh < dst_height; dh++){
        for(int dw = 0; dw < dst_width; dw++){
             for(int c = 0; c<3; c++){
                double value = 0;
                float totalWeight = 0;

                float x = ((static_cast<float>(dw)+ 0.5f)
                            *static_cast<float>(src_width)) / static_cast<float>(dst_width);
                x -= 0.5f;
                float y = (static_cast<float>(dh)+ 0.5f)
                            *static_cast<float>(src_height) / static_cast<float>(dst_height);
                y -= 0.5f;
                float dx = x - static_cast<float>(floor(x));
                float dy = y - static_cast<float>(floor(y));

                for(int i = -1; i < 3; i++){
                    for(int j = -1; j < 3; j++){

                        float wx = weight(j - dx);
                        float wy = weight(i - dy);
                        float w = wx * wy;

                        int sw = BORDER_INTERPOLATE((int)(x + j), src_width);
                        int sh = BORDER_INTERPOLATE((int)(y + i), src_height);
                        T s = src(sw, sh, c);

                        value += w*s;
                        totalWeight += w;
                    }

                }
                if(fabs(totalWeight)>0){
                    value /= fabs(totalWeight);
                }else{
                    value= 0;
                }
                value += 0.5;
                value = (value < min_value) ? min_value : value;
                value = (value > max_value) ? max_value : value;
                dst(dw, dh, c) = static_cast<T>(value);
             }
        }
    }
}


std::unordered_map<int32_t, std::vector<uint8_t>> image_cache;

template<typename T>
bool get_image(const std::string &url, Halide::Runtime::Buffer<T> &img, int width_, int height_) {
    bool img_loaded = false;
    if (url.empty()) {
        return img_loaded;
    }

    std::string host_name;
    std::string path_name;
    std::tie(host_name, path_name) = parse_url(url);

    Halide::Runtime::Buffer<uint8_t> img_buf;
    if (host_name.empty() || path_name.empty()){
        // fallback to local file
        if (std::filesystem::exists(url)){
            img_buf = Halide::Tools::load_and_convert_image(url);
            img_loaded = true;
        }
    }else{
        httplib::Client cli(host_name.c_str());
        cli.set_follow_location(true);
        auto res = cli.Get(path_name.c_str());
        if (res && res->status == 200) {
            std::vector<char> data(res->body.size());
            data.resize(res->body.size());
            std::memcpy(data.data(), res->body.c_str(), res->body.size());
            std::filesystem::path dir_path =  std::filesystem::temp_directory_path() / "simulation_camera";;
            if (!std::filesystem::exists(dir_path)) {
                if (!std::filesystem::create_directory(dir_path)) {
                    throw std::runtime_error("Failed to create temporary directory");
                }
            }
            std::ofstream ofs(dir_path / std::filesystem::path(url).filename(), std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(data.data()), data.size());

            img_buf  =  Halide::Tools::load_and_convert_image(dir_path / std::filesystem::path(url).filename());
            img_loaded = true;

        }
    }
    if (img_loaded){ //resize
        int ori_width = img_buf.width();
        int ori_height = img_buf.height();
        int channels =  img_buf.channels();
        Halide::Runtime::Buffer<uint8_t> resized (width_, height_, 3);
        resize_bicubic<uint8_t>(resized, img_buf, ori_width, ori_height, width_, height_);

        if (sizeof(T) == 4){ //float
            // Buffer<uint8_t> to Buffer<float> range(0-1)
            Halide::Runtime::Buffer<float> float_img = Halide::Tools::ImageTypeConversion::convert_image(resized, halide_type_of<float>());
            img.copy_from(float_img);
        }else if (sizeof(T) == 1){ //uint_8
            img.copy_from(resized);
        }else{
            throw std::runtime_error("Unsupported image format");
        }
    }
    return img_loaded;
}

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
        std::cout<<id_;
        if (it != ion::bb::image_io::image_cache.end()) {
            memcpy(buf.data(), it->second.data(), it->second.size() * sizeof(T));
            return;
        }

        Halide::Runtime::Buffer<float> img(width_, height_, 3);
        bool is_loaded = get_image<float>(url_, img, width_, height_);

        if (!is_loaded) {
            // Fill by dummy image
            Halide::Runtime::Buffer<uint16_t> img_16(width_, height_);
            img_16.fill(0);
             for(int y = (index_ / 2) % 2; y < height_ ; y+=2){
                for(int x = index_ % 2; x < width_; x+=2){
                     img_16(x, y) = 65535 ;
                }
             }
            buf.copy_from(img_16);
            auto size  = width_ * height_ * sizeof(uint16_t);
            std::vector<uint8_t> data(size);
            memcpy(data.data(), img_16.data(), size);
            ion::bb::image_io::image_cache[id_] = data;
            return;
        }

        img.for_each_element([&](int x, int y, int c) {
            img(x, y, c) = pow((float)(img(x, y, c)), 2.2) ;
        });

        std::vector<float> r_planar(width_*height_);
        std::vector<float> g_planar(width_*height_);
        std::vector<float> b_planar(width_*height_);

        memcpy(r_planar.data(), img.data(), width_* height_* sizeof(float));
        memcpy(g_planar.data(), img.data() + width_ * height_ * 1, width_* height_* sizeof(float));
        memcpy(b_planar.data(), img.data() + width_ * height_ * 2, width_* height_* sizeof(float));

        std::transform(r_planar.begin(), r_planar.end(), r_planar.begin(), [&](float x){return std::min(std::max(x * gain_r_ + offset_, 0.f), 1.f);});
        std::transform(g_planar.begin(), g_planar.end(), g_planar.begin(), [&](float x){return std::min(std::max(x * gain_g_ + offset_, 0.f), 1.f);});
        std::transform(b_planar.begin(), b_planar.end(), b_planar.begin(), [&](float x){return std::min(std::max(x * gain_b_ + offset_, 0.f), 1.f);});

        std::vector<float> processed_img_arr(width_*height_);
        int idx = 0;
        for (int j = 0; j < height_; j++) {
            int evenRow = j % 2 == 0;
            for (int i = 0; i < width_; i++) {
                int evenCol = i % 2 == 0;
                switch (bayer_pattern(pixel_format_)) {
                    case 0:  // RGGB
                        processed_img_arr[idx] = evenRow ? (evenCol ? r_planar[idx] : g_planar[idx]) : (evenCol ? g_planar[idx] : b_planar[idx]);
                        break;
                    case 1:  // BGGR
                        processed_img_arr[idx] = evenRow ? (evenCol ? b_planar[idx] : g_planar[idx]) : (evenCol ? g_planar[idx] : r_planar[idx]);
                        break;
                    case 2:  // GRBG
                        processed_img_arr[idx] = evenRow ? (evenCol ? g_planar[idx] : r_planar[idx]) : (evenCol ? b_planar[idx] : g_planar[idx]);
                        break;
                    case 3:  // GBRG
                        processed_img_arr[idx] = evenRow ? (evenCol ? g_planar[idx] : b_planar[idx]) : (evenCol ? r_planar[idx] : g_planar[idx]);
                        break;
                }
                idx+=1;
            }
        }

        std::vector<uint16_t> bit_shifted_img_arr(width_*height_);
        for (int i = 0;i<width_*height_;i++){
            float val = processed_img_arr[i];
            val *= (float)((1 << bit_width_) - 1);
            val = val * (float)(1 << bit_shift_) + val / (float)(1 << (bit_width_ - bit_shift_));
            bit_shifted_img_arr[i] = static_cast<uint16_t>(val);
        }
        std::vector<uint8_t> data(bit_shifted_img_arr.size() * sizeof(uint16_t));
        memcpy(data.data(), bit_shifted_img_arr.data(), bit_shifted_img_arr.size() * sizeof(uint16_t));
        memcpy(buf.data(), bit_shifted_img_arr.data(), bit_shifted_img_arr.size() * sizeof(uint16_t));
        ion::bb::image_io::image_cache[id_] = data;
    }


// Created by         :  Harris Zhu
// Filename           :  rgb2I420.cpp
// Avthor             :  Harris Zhu
//=======================================================================

#include <stdint.h>
#include <stddef.h>
#define BORDER_INTERPOLATE(x, l) (x < 0 ? 0 : (x >= l ? l - 1 : x))
    template<typename T>
    void rgb2YCrCb(T *destination, Halide::Runtime::Buffer<T> &rgb, int width, int height){
        for(int y = 0; y < height ; y++){
            for(int x = 0; x < width; x++){
                T r = rgb(x, y, 0);
                T g = rgb(x, y, 1);
                T b = rgb(x, y, 2);

                T Yy =  0.299 * r + 0.587 * g + 0.114 * b  ;
                T Cr =  (r-Yy) * 0.713 + 128;
                T Cb =  (b-Yy) * 0.564 + 128;
                destination[(x+y*width)*3] = Yy;
                destination[(x+y*width)*3+1] = Cr;
                destination[(x+y*width)*3+2] = Cb;
            }
        }

    }

    template<typename T>
    void generate_yuyv(Halide::Runtime::Buffer<T> &buf) {
        auto it = ion::bb::image_io::image_cache.find(id_);
        if (it != ion::bb::image_io::image_cache.end()) {
            memcpy(buf.data(), it->second.data(), it->second.size() * sizeof(T));
            return;
        }
        Halide::Runtime::Buffer<uint8_t> img (width_, height_, 3);
        bool is_loaded = get_image<uint8_t>(url_, img,  width_, height_);;
        std::vector<uint8_t> yuyv_img(2 * width_ * height_);
        if (!is_loaded) {
            // Fill by dummy image
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < 2 * width_; ++x) {
                    yuyv_img[2 * width_ * y + x] = (y * 2 * width_ + x) % 255;
                }
            }
        } else {

            std::vector<uint8_t> yuv(3 * width_ * width_);
            rgb2YCrCb<uint8_t>(yuv.data(), img, width_, height_);

            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    // Y
                    yuyv_img[2 * width_ * y + 2 * x + 0] = yuv[( x + y * width_) * 3];
                    // Cb or Cr
                    yuyv_img[2 * width_ * y + 2 * x + 1] = ((x % 2) == 1) ? yuv[(y * width_ + x) * 3 + 1]:yuv[(y * width_ + x) * 3 + 2];
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



#endif // ION_BB_IMAGE_IO_RT_V4L2_H

