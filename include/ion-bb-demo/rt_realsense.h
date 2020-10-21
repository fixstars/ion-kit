#ifndef ION_BB_DEMO_RT_REALSENSE_H
#define ION_BB_DEMO_RT_REALSENSE_H

#include <cstring>
#include <iostream>

#include <HalideBuffer.h>

#include "rt_common.h"

namespace ion {
namespace bb {
namespace demo {
class RealSense {
    // NOTE: Copied from librealsense2/rt_sensor.h
    typedef enum rs2_stream {
        RS2_STREAM_ANY,
        RS2_STREAM_DEPTH,      /**< Native stream of depth data produced by RealSense device */
        RS2_STREAM_COLOR,      /**< Native stream of color data captured by RealSense device */
        RS2_STREAM_INFRARED,   /**< Native stream of infrared data captured by RealSense device */
        RS2_STREAM_FISHEYE,    /**< Native stream of fish-eye (wide) data captured from the dedicate motion camera */
        RS2_STREAM_GYRO,       /**< Native stream of gyroscope motion data produced by RealSense device */
        RS2_STREAM_ACCEL,      /**< Native stream of accelerometer motion data produced by RealSense device */
        RS2_STREAM_GPIO,       /**< Signals from external device connected through GPIO */
        RS2_STREAM_POSE,       /**< 6 Degrees of Freedom pose data, calculated by RealSense device */
        RS2_STREAM_CONFIDENCE, /**< 4 bit per-pixel depth confidence level */
        RS2_STREAM_COUNT
    } rs2_stream_t;
    typedef enum rs2_format {
        RS2_FORMAT_ANY,           /**< When passed to enable stream, librealsense will try to provide best suited format */
        RS2_FORMAT_Z16,           /**< 16-bit linear depth values. The depth is meters is equal to depth scale * pixel value. */
        RS2_FORMAT_DISPARITY16,   /**< 16-bit float-point disparity values. Depth->Disparity conversion : Disparity = Baseline*FocalLength/Depth. */
        RS2_FORMAT_XYZ32F,        /**< 32-bit floating point 3D coordinates. */
        RS2_FORMAT_YUYV,          /**< 32-bit y0, u, y1, v data for every two pixels. Similar to YUV422 but packed in a different order - https://en.wikipedia.org/wiki/YUV */
        RS2_FORMAT_RGB8,          /**< 8-bit red, green and blue channels */
        RS2_FORMAT_BGR8,          /**< 8-bit blue, green, and red channels -- suitable for OpenCV */
        RS2_FORMAT_RGBA8,         /**< 8-bit red, green and blue channels + constant alpha channel equal to FF */
        RS2_FORMAT_BGRA8,         /**< 8-bit blue, green, and red channels + constant alpha channel equal to FF */
        RS2_FORMAT_Y8,            /**< 8-bit per-pixel grayscale image */
        RS2_FORMAT_Y16,           /**< 16-bit per-pixel grayscale image */
        RS2_FORMAT_RAW10,         /**< Four 10 bits per pixel luminance values packed into a 5-byte macropixel */
        RS2_FORMAT_RAW16,         /**< 16-bit raw image */
        RS2_FORMAT_RAW8,          /**< 8-bit raw image */
        RS2_FORMAT_UYVY,          /**< Similar to the standard YUYV pixel format, but packed in a different order */
        RS2_FORMAT_MOTION_RAW,    /**< Raw data from the motion sensor */
        RS2_FORMAT_MOTION_XYZ32F, /**< Motion data packed as 3 32-bit float values, for X, Y, and Z axis */
        RS2_FORMAT_GPIO_RAW,      /**< Raw data from the external sensors hooked to one of the GPIO's */
        RS2_FORMAT_6DOF,          /**< Pose data packed as floats array, containing translation vector, rotation quaternion and prediction velocities and accelerations vectors */
        RS2_FORMAT_DISPARITY32,   /**< 32-bit float-point disparity values. Depth->Disparity conversion : Disparity = Baseline*FocalLength/Depth */
        RS2_FORMAT_Y10BPACK,      /**< 16-bit per-pixel grayscale image unpacked from 10 bits per pixel packed ([8:8:8:8:2222]) grey-scale image. The data is unpacked to LSB and padded with 6 zero bits */
        RS2_FORMAT_DISTANCE,      /**< 32-bit float-point depth distance value.  */
        RS2_FORMAT_MJPEG,         /**< Bitstream encoding for video in which an image of each frame is encoded as JPEG-DIB   */
        RS2_FORMAT_Y8I,           /**< 8-bit per pixel interleaved. 8-bit left, 8-bit right.  */
        RS2_FORMAT_Y12I,          /**< 12-bit per pixel interleaved. 12-bit left, 12-bit right. Each pixel is stored in a 24-bit word in little-endian order. */
        RS2_FORMAT_INZI,          /**< multi-planar Depth 16bit + IR 10bit.  */
        RS2_FORMAT_INVI,          /**< 8-bit IR stream.  */
        RS2_FORMAT_W10,           /**< Grey-scale image as a bit-packed array. 4 pixel data stream taking 5 bytes */
        RS2_FORMAT_Z16H,          /**< Variable-length Huffman-compressed 16-bit depth values. */
        RS2_FORMAT_COUNT          /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
    } rs2_format_t;

    using rs2_error_t = struct rs2_error;
    using rs2_context_t = struct rs2_context;
    using rs2_device_list_t = struct rs2_device_list;
    using rs2_pipeline_t = struct rs2_pipeline;
    using rs2_pipeline_profile_t = struct rs2_pipeline_profile;
    using rs2_config_t = struct rs2_config;
    using rs2_frame_t = struct rs2_frame;

    using rs2_get_api_version_t = int (*)(rs2_error_t **error);
    using rs2_get_error_message_t = const char *(*)(const rs2_error_t *);
    using rs2_create_context_t = rs2_context_t *(*)(int, rs2_error_t **);
    using rs2_query_devices_t = rs2_device_list_t *(*)(const rs2_context_t *, rs2_error_t **);
    using rs2_get_device_count_t = int (*)(const rs2_device_list_t *, rs2_error_t **);
    using rs2_create_config_t = rs2_config_t *(*)(rs2_error_t **);
    using rs2_config_enable_stream_t = void (*)(rs2_config_t *, rs2_stream_t, int, int, int, rs2_format_t, int, rs2_error_t **);
    using rs2_create_pipeline_t = rs2_pipeline_t *(*)(rs2_context_t *, rs2_error_t **);
    using rs2_pipeline_start_with_config_t = rs2_pipeline_profile_t *(*)(rs2_pipeline_t *, rs2_config_t *, rs2_error_t **);
    using rs2_pipeline_stop_t = void (*)(rs2_pipeline_t *, rs2_error_t **);
    using rs2_pipeline_wait_for_frames_t = rs2_frame_t *(*)(rs2_pipeline_t *, unsigned int, rs2_error_t **);
    using rs2_extract_frame_t = rs2_frame_t *(*)(rs2_frame_t *, int, rs2_error_t **);
    using rs2_get_frame_data_t = const void *(*)(const rs2_frame_t *, rs2_error_t **);
    using rs2_release_frame_t = void (*)(const rs2_frame_t *);

    rs2_get_api_version_t rs2_get_api_version;
    rs2_get_error_message_t rs2_get_error_message;
    rs2_create_context_t rs2_create_context;
    rs2_query_devices_t rs2_query_devices;
    rs2_get_device_count_t rs2_get_device_count;
    rs2_create_config_t rs2_create_config;
    rs2_config_enable_stream_t rs2_config_enable_stream;
    rs2_create_pipeline_t rs2_create_pipeline;
    rs2_pipeline_start_with_config_t rs2_pipeline_start_with_config;
    rs2_pipeline_stop_t rs2_pipeline_stop;
    rs2_pipeline_wait_for_frames_t rs2_pipeline_wait_for_frames;
    rs2_extract_frame_t rs2_extract_frame;
    rs2_get_frame_data_t rs2_get_frame_data;
    rs2_release_frame_t rs2_release_frame;

public:
    static RealSense &get_instance(int32_t width, int32_t height) {
        static RealSense instance(width, height);
        return instance;
    }

    RealSense(int32_t width, int32_t height)
        : dm_("realsense2"), device_is_available_(false), ctx_(nullptr), devices_(nullptr), pipeline_(nullptr), config_(nullptr), frameset_(nullptr) {

        if (!init_symbols()) {
            return;
        }

        rs2_error_t *err = nullptr;

        int version = rs2_get_api_version(&err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        ctx_ = rs2_create_context(version, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        devices_ = rs2_query_devices(ctx_, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        int num_of_devices = rs2_get_device_count(devices_, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }
        if (num_of_devices == 0) {
            return;
        }

        int fps = 30;
        config_ = rs2_create_config(&err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        rs2_config_enable_stream(config_, RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        rs2_config_enable_stream(config_, RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        rs2_config_enable_stream(config_, RS2_STREAM_DEPTH, 0, width, height, RS2_FORMAT_Z16, fps, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        pipeline_ = rs2_create_pipeline(ctx_, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        rs2_pipeline_start_with_config(pipeline_, config_, &err);
        if (err) {
            std::cerr << rs2_get_error_message(err) << std::endl;
            return;
        }

        device_is_available_ = true;
    }

    ~RealSense() {
        rs2_error_t *err = nullptr;
        rs2_pipeline_stop(pipeline_, &err);
        // TODO: Delete recources
    }

    void get_frameset(Halide::Runtime::Buffer<uint64_t> &buf) {
        rs2_error_t *err = nullptr;
        if (frameset_) {
            rs2_release_frame(frameset_);
        }
        frameset_ = rs2_pipeline_wait_for_frames(pipeline_, 15000, &err);
        if (err) {
            throw std::runtime_error(rs2_get_error_message(err));
        }
        buf() = reinterpret_cast<uint64_t>(frameset_);
    }

    const void *get_frame_ptr(void *frameset, int index) {
        rs2_error_t *err = nullptr;
        rs2_frame_t *frame = rs2_extract_frame(static_cast<rs2_frame_t *>(frameset), index, &err);
        if (err) {
            throw std::runtime_error(rs2_get_error_message(err));
        }

        const void *frame_ptr = rs2_get_frame_data(frame, &err);
        if (err) {
            throw std::runtime_error(rs2_get_error_message(err));
        }

        return frame_ptr;
    }

    bool is_available() const {
        return device_is_available_;
    }

private:
    bool init_symbols() {
        if (!dm_.is_available()) {
            return false;
        }

#define GET_SYMBOL(LOCAL_VAR, TARGET_SYMBOL)                  \
    LOCAL_VAR = dm_.get_symbol<LOCAL_VAR##_t>(TARGET_SYMBOL); \
    if (LOCAL_VAR == nullptr) {                               \
        return false;                                         \
    }

        GET_SYMBOL(rs2_get_api_version, "rs2_get_api_version");
        GET_SYMBOL(rs2_get_error_message, "rs2_get_error_message");
        GET_SYMBOL(rs2_create_context, "rs2_create_context");
        GET_SYMBOL(rs2_query_devices, "rs2_query_devices");
        GET_SYMBOL(rs2_get_device_count, "rs2_get_device_count");
        GET_SYMBOL(rs2_create_config, "rs2_create_config");
        GET_SYMBOL(rs2_config_enable_stream, "rs2_config_enable_stream");
        GET_SYMBOL(rs2_create_pipeline, "rs2_create_pipeline");
        GET_SYMBOL(rs2_pipeline_start_with_config, "rs2_pipeline_start_with_config");
        GET_SYMBOL(rs2_pipeline_stop, "rs2_pipeline_stop");
        GET_SYMBOL(rs2_pipeline_wait_for_frames, "rs2_pipeline_wait_for_frames");
        GET_SYMBOL(rs2_extract_frame, "rs2_extract_frame");
        GET_SYMBOL(rs2_get_frame_data, "rs2_get_frame_data");
        GET_SYMBOL(rs2_release_frame, "rs2_release_frame");

#undef GET_SYMBOL

        return true;
    }

    DynamicModule dm_;
    bool device_is_available_;

    rs2_context_t *ctx_;
    rs2_device_list_t *devices_;
    rs2_pipeline_t *pipeline_;
    rs2_config_t *config_;
    rs2_frame_t *frameset_;
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

extern "C" int ION_EXPORT ion_bb_demo_realsense_d435_infrared(halide_buffer_t *in, halide_buffer_t *out_l, halide_buffer_t *out_r) {
    try {
        const int width = 1280;
        const int height = 720;

        auto &realsense(ion::bb::demo::RealSense::get_instance(width, height));

        if (out_l->is_bounds_query() || out_r->is_bounds_query()) {
            if (out_l->is_bounds_query()) {
                out_l->dim[0].min = 0;
                out_l->dim[0].extent = width;
                out_l->dim[1].min = 0;
                out_l->dim[1].extent = height;
            }
            if (out_r->is_bounds_query()) {
                out_r->dim[0].min = 0;
                out_r->dim[0].extent = width;
                out_r->dim[1].min = 0;
                out_r->dim[1].extent = height;
            }
        } else {
            Halide::Runtime::Buffer<uint8_t> obuf_l(*out_l);
            Halide::Runtime::Buffer<uint8_t> obuf_r(*out_r);

            void *frameset = reinterpret_cast<void *>(Halide::Runtime::Buffer<uint64_t>(*in)());
            if (frameset) {
                std::memcpy(obuf_l.data(), realsense.get_frame_ptr(frameset, 1), obuf_l.size_in_bytes());
                std::memcpy(obuf_r.data(), realsense.get_frame_ptr(frameset, 2), obuf_r.size_in_bytes());
            } else {
                // Simulation mode
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        obuf_l(x, y) = (y * width + x) % 256;
                        obuf_r(x, y) = (y * width + x) % 256;
                    }
                }
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

extern "C" int ION_EXPORT ion_bb_demo_realsense_d435_depth(halide_buffer_t *in, halide_buffer_t *out_d) {
    try {
        const int width = 1280;
        const int height = 720;

        auto &realsense(ion::bb::demo::RealSense::get_instance(width, height));

        if (out_d->is_bounds_query()) {
            if (out_d->is_bounds_query()) {
                out_d->dim[0].min = 0;
                out_d->dim[0].extent = width;
                out_d->dim[1].min = 0;
                out_d->dim[1].extent = height;
            }
        } else {
            Halide::Runtime::Buffer<uint16_t> obuf_d(*out_d);
            void *frameset = reinterpret_cast<void *>(Halide::Runtime::Buffer<uint64_t>(*in)());
            if (frameset) {
                std::memcpy(obuf_d.data(), realsense.get_frame_ptr(frameset, 0), obuf_d.size_in_bytes());
            } else {
                // Simulation mode
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        obuf_d(x, y) = static_cast<uint16_t>((y * width + x) % 65536);
                    }
                }
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

extern "C" int ION_EXPORT ion_bb_demo_realsense_d435_frameset(halide_buffer_t *out) {
    try {
        const int width = 1280;
        const int height = 720;

        auto &realsense(ion::bb::demo::RealSense::get_instance(width, height));

        if (out->is_bounds_query()) {
            // out->dim[0].min = 0;
            // out->dim[0].extent = 1;
        } else {
            Halide::Runtime::Buffer<uint64_t> obuf(*out);
            if (realsense.is_available()) {
                realsense.get_frameset(obuf);
            } else {
                obuf() = reinterpret_cast<uint64_t>(nullptr);
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
