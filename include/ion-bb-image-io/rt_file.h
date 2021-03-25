#ifndef ION_BB_IMAGE_IO_RT_FILE_H
#define ION_BB_IMAGE_IO_RT_FILE_H

#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "rt_common.h"

#include "httplib.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

extern "C" int ION_EXPORT ion_bb_image_io_image_loader(halide_buffer_t *in, halide_buffer_t *out) {
    const char *url = reinterpret_cast<const char *>(in->host);

    static std::unordered_map<const char *, cv::Mat> decoded;
    if (decoded.count(url) == 0) {
        std::string host_name;
        std::string path_name;
        std::tie(host_name, path_name) = ion::bb::image_io::parse_url(url);
        bool img_loaded = false;
        cv::Mat frame;
        if (host_name.empty() || path_name.empty()) {
            // fallback to local file
            frame = cv::imread(url);
            if (!frame.empty()) {
                img_loaded = true;
            }
        } else {
            httplib::Client cli(host_name.c_str());
            cli.set_follow_location(true);
            auto res = cli.Get(path_name.c_str());
            if (res && res->status == 200) {
                std::vector<char> data(res->body.size());
                std::memcpy(data.data(), res->body.c_str(), res->body.size());
                frame = cv::imdecode(cv::InputArray(data), cv::IMREAD_COLOR);
                img_loaded = true;
            }
        }
        if (img_loaded) {
            decoded[url] = frame;
        } else {
            return -1;
        }
    }

    const cv::Mat &img(decoded[url]);

    if (out->is_bounds_query()) {
        out->dim[0].min = 0;
        out->dim[0].extent = 3;
        out->dim[1].min = 0;
        out->dim[1].extent = img.cols;
        out->dim[2].min = 0;
        out->dim[2].extent = img.rows;
    } else {
        std::memcpy(out->host, img.data, img.total() * img.elemSize());
    }

    return 0;
}

extern "C" int ION_EXPORT ion_bb_image_io_image_saver(halide_buffer_t *in, int32_t in_extent_1, int32_t in_extent_2, halide_buffer_t *path, halide_buffer_t *out) {
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

    return 0;
}

#endif
