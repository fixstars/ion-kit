#ifndef ION_BB_GENESIS_CLOUD_RT_CAMERA_H
#define ION_BB_GENESIS_CLOUD_RT_CAMERA_H

#include <stdexcept>
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

namespace {

int xioctl(int fd, int request, void * arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    }
    while (-1 == r && EINTR == errno);
    return r;
}

class V4L2 {

    struct Buffer {
        void *start;
        size_t length;
    };

 public:
     V4L2(int32_t width, int32_t height) : device_is_available_(true) {
         using namespace std;

         //
         // Initialize device
         //
         const char *dev_name = "/dev/video0";
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

         uint32_t desired_pixel_format = V4L2_PIX_FMT_YUYV;

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
                         .pixelformat = desired_pixel_format,
                         .field = V4L2_FIELD_INTERLACED,
                     }
                 },
         };
         if (-1 == xioctl(fd_, VIDIOC_S_FMT, &fmt)){
             std::cerr << format("%s error %d, %s\n", "VIDIOC_S_FMT", errno, strerror(errno)) << std::endl;
             device_is_available_ = false;
             return;
         }
         if (width != fmt.fmt.pix.width || height != fmt.fmt.pix.height) {
             std::cerr << format("%s does not support desired resolution", dev_name) << std::endl;
             device_is_available_ = false;
             return;
         }

         /* YUYV sampling 4 2 2, so bytes per pixel is 2*/

         unsigned int min;
         min = fmt.fmt.pix.width * 2;
         if (fmt.fmt.pix.bytesperline < min){
             fmt.fmt.pix.bytesperline = min;
         }
         min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
         if (fmt.fmt.pix.sizeimage < min){
             fmt.fmt.pix.sizeimage = min;
         }

         //
         // Initialize mapped memory
         //
         struct v4l2_requestbuffers req;
         req.count = 4;
         req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
         req.memory = V4L2_MEMORY_MMAP;

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
         /* video output requires at least two buffers, one displayed and one filled by the application */
         if (req.count < 2) {
             std::cerr << format("Insufficient buffer memory on %s\n", dev_name) << std::endl;
             device_is_available_ = false;
             return;
         }

         for (int i=0; i<req.count; ++i) {
             struct v4l2_buffer buf;
             buf.index = static_cast<__u32>(i);
             buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
             buf.memory = V4L2_MEMORY_MMAP;

             /* Query the status of a buffer */
             if (-1 == xioctl(fd_, VIDIOC_QUERYBUF, &buf)){
                 std::cerr << format("%s error %d, %s\n", "VIDIOC_QUERYBUF", errno, strerror(errno)) << std::endl;
                 device_is_available_ = false;
                 return;
             }

             Buffer buffer;
             buffer.start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
             buffer.length = buf.length;

             if (MAP_FAILED == buffer.start) {
                 std::cerr << format("%s error %d, %s\n", "mmap", errno, strerror(errno)) << std::endl;
                 device_is_available_ = false;
                 return;
             }

             buffers_.push_back(buffer);
         }

         //
         // Start capture
         //
         for (int i = 0; i < buffers_.size(); ++i) {
             struct v4l2_buffer buf;
             buf.index = static_cast<__u32>(i);
             buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
             buf.memory = V4L2_MEMORY_MMAP;

             /* enqueue an empty (capturing) or filled (output) buffer in the driver's incoming queue */
             if (-1 == xioctl(fd_, VIDIOC_QBUF, &buf)){
                 std::cerr << format("%s error %d, %s\n", "VIDIOC_QBUF", errno, strerror(errno)) << std::endl;
                 device_is_available_ = false;
                 return;
             }
         }
         enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
         /* Start streaming I/O */
         if (-1 == xioctl(fd_, VIDIOC_STREAMON, &type)){
             std::cerr << format("%s error %d, %s\n", "VIDIOC_STREAMON", errno, strerror(errno)) << std::endl;
             device_is_available_ = false;
             return;
         }

         //
         // Initialize event
         //
         efd_ = epoll_create1(0);
         if (-1 == efd_)
         {
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

     void get(Halide::Runtime::Buffer<uint8_t>& buf) {
         using namespace std;

         epoll_event event;
         if (-1 == epoll_wait(efd_, &event, 1, -1)) {
             throw runtime_error(format("%s error %d, %s\n", "epoll_wait", errno, strerror(errno)));
         }

         if (event.data.fd != fd_) {
             throw runtime_error("Unreachable");
         }

         struct v4l2_buffer v4l2_buf;
         v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
         v4l2_buf.memory = V4L2_MEMORY_MMAP;

         if (-1 == xioctl(fd_, VIDIOC_DQBUF, &v4l2_buf)) {
             if (EAGAIN == errno) {
                 return;
             } else {
                 throw runtime_error(format("%s error %d, %s\n", "VIDIOC_DQBUF", errno, strerror(errno)));
             }
         }

         memcpy(buf.data(), buffers_[v4l2_buf.index].start, buf.size_in_bytes());

         /* queue-in buffer */
         if (-1 == xioctl(fd_, VIDIOC_QBUF, &v4l2_buf)){
             throw runtime_error(format("%s error %d, %s\n", "VIDIOC_QBUF", errno, strerror(errno)));
         }
     }

     void dispose() {
     }

     bool is_available() {
         return device_is_available_;
     }

 private:
     int fd_;
     std::vector<Buffer> buffers_;
     bool device_is_available_;

     int efd_;
};

}

extern "C"
int ION_EXPORT ion_bb_genesis_cloud_camera(int32_t width, int32_t height, halide_buffer_t *out) {
    try {
        static V4L2 v4l2(width, height);

        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = 2 * width; // YUYV
            out->dim[1].min = 0;
            out->dim[1].extent = height;
        } else {
            Halide::Runtime::Buffer<uint8_t> obuf(*out);

            if (v4l2.is_available()) {
                v4l2.get(obuf);
            } else {
                // Simulation mode
                for (int y=0; y<height; ++y) {
                    for (int x=0; x<2*width; ++x) {
                        obuf(x, y) = (y * 2 * width + x) % 255;
                    }
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown" << std::endl;
        return -1;
    }
}

#endif
