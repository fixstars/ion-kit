#ifndef ION_BB_GENESIS_CLOUD_RT_H
#define ION_BB_GENESIS_CLOUD_RT_H

#include <iostream>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <HalideBuffer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "httplib.h"

#include "rt_common.h"
#include "rt_camera.h"
#include "rt_display.h"

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

namespace ion {
namespace bb {
namespace genesis_cloud {

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

} // namespace dnn
} // namespace bb
} // namespace ion

extern "C"
int ION_EXPORT ion_bb_genesis_cloud_image_loader(halide_buffer_t *in, halide_buffer_t *out) {

    if (in->is_bounds_query()) {
       // NOP
    } else {

        const char *url = reinterpret_cast<const char *>(in->host);

        static std::unordered_map<const char*, cv::Mat> decoded;
        if (decoded.count(url) == 0) {
            std::string host_name;
            std::string path_name;
            std::tie(host_name, path_name) = ion::bb::genesis_cloud::parse_url(url);
            if (host_name.empty() || path_name.empty()) {
                std::cerr << "Invalid URL is specified" << std::endl;
                return -1;
            }
            httplib::Client cli(host_name.c_str());
            cli.set_follow_location(true);
            auto res = cli.Get(path_name.c_str());
            if (res && res->status == 200) {
                std::vector<char> data(res->body.size());
                std::memcpy(data.data(), res->body.c_str(), res->body.size());
                decoded[url] = cv::imdecode(cv::InputArray(data), cv::IMREAD_COLOR);
            }
        }

        const cv::Mat& img(decoded[url]);

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
    }

    return 0;
}

extern "C"
int ION_EXPORT ion_bb_genesis_cloud_image_saver(halide_buffer_t *in, int32_t in_extent_1, int32_t in_extent_2, halide_buffer_t *path, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = 3;
        in->dim[1].min = 0;
        in->dim[1].extent = in_extent_1;
        in->dim[2].min = 0;
        in->dim[2].extent = in_extent_2;
    } else {
        cv::Mat img(std::vector<int>{in_extent_2, in_extent_1}, CV_8UC3, in->host);
        cv::imwrite(reinterpret_cast<const char*>(path->host), img);
    }

    return 0;
}

#undef ION_EXPORT

#endif
