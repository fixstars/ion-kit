#ifndef ION_BB_IMAGE_IO_RT_FILE_H
#define ION_BB_IMAGE_IO_RT_FILE_H

#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "rt_common.h"

#include "httplib.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

extern "C" int ION_EXPORT ion_bb_image_io_color_data_loader(halide_buffer_t *session_id_buf, halide_buffer_t *url_buf, int32_t width, int32_t height, halide_buffer_t *out) {

    using namespace ion::bb::image_io;

    try {

        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = width,
            out->dim[1].min = 0;
            out->dim[1].extent = height;
            out->dim[2].min = 0;
            out->dim[2].extent = 3;
        } else {
            const std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            const std::string url = reinterpret_cast<const char *>(url_buf->host);
            static std::unordered_map<std::string, std::unique_ptr<ImageSequence>> seqs;
            if (seqs.count(session_id) == 0) {
                seqs[session_id] = std::unique_ptr<ImageSequence>(new ImageSequence(session_id, url));
            }
            auto frame = seqs[session_id]->get(width, height, cv::IMREAD_COLOR);

            // Resize to desired width/height
            cv::resize(frame, frame, cv::Size(height, width), 0, 0);

            // Convert to RGB from BGR
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            // Reshape interleaved to planar
            frame = frame.reshape(1, width*height).t();

            std::memcpy(out->host, frame.data, width * height * 3 * sizeof(uint8_t));
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}

extern "C" int ION_EXPORT ion_bb_image_io_grayscale_data_loader(halide_buffer_t *session_id_buf, halide_buffer_t *url_buf, int32_t width, int32_t height, int32_t dynamic_range, halide_buffer_t *out) {

    using namespace ion::bb::image_io;

    try {
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = width;
            out->dim[1].min = 0;
            out->dim[1].extent = height;
        } else {
            const std::string session_id(reinterpret_cast<const char *>(session_id_buf->host));
            const std::string url = reinterpret_cast<const char *>(url_buf->host);
            static std::unordered_map<std::string, std::unique_ptr<ImageSequence>> seqs;
            if (seqs.count(session_id) == 0) {
                seqs[session_id] = std::unique_ptr<ImageSequence>(new ImageSequence(session_id, url));
            }
            auto frame = seqs[session_id]->get(width, height, cv::IMREAD_GRAYSCALE);

            // Normalize value range from 0-255 into 0-dynamic_range
            cv::normalize(frame, frame, 0, dynamic_range, cv::NORM_MINMAX, CV_16UC1);

            std::memcpy(out->host, frame.data, width * height * sizeof(uint16_t));
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}

extern "C" int ION_EXPORT ion_bb_image_io_saver(halide_buffer_t *in, int32_t in_extent_1, int32_t in_extent_2, halide_buffer_t *path, halide_buffer_t *out) {
    try {
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = 3;
            in->dim[1].min = 0;
            in->dim[1].extent = in_extent_1;
            in->dim[2].min = 0;
            in->dim[2].extent = in_extent_2;
        } else {
            cv::Mat img(std::vector<int>{in_extent_2, in_extent_1}, CV_8UC3, in->host);
            cv::imwrite(reinterpret_cast<const char *>(path->host), img);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}

#endif
