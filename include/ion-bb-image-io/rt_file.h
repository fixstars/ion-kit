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
#include "zip_file.hpp"
#include "ghc/filesystem.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

bool end_with(const std::string& target_str, const std::string& end_str) {
    // It cannot contain ".zip"
    if (target_str.size() < end_str.size()) {
        return false;
    }
    return target_str.substr(target_str.size() - end_str.size()) == end_str;
}

} // anonymous

extern "C" int ION_EXPORT ion_bb_image_io_image_loader(halide_buffer_t *in, halide_buffer_t *out) {
    try {
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
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }

    return 0;
}

extern "C" int ION_EXPORT ion_bb_image_io_image_saver(halide_buffer_t *in, int32_t in_extent_1, int32_t in_extent_2, halide_buffer_t *path, halide_buffer_t *out) {
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

class ImageSequence {

 public:
     ImageSequence(const std::string& session_id, const std::string& url) : idx_(0) {
        namespace fs = ghc::filesystem;

        std::string host_name;
        std::string path_name;
        std::tie(host_name, path_name) = ion::bb::image_io::parse_url(url);

        std::vector<unsigned char> data;
        if (host_name.empty() || path_name.empty()) {
            // fallback to local file
            data.resize(fs::file_size(url));
            std::ifstream ifs(url, std::ios::binary);
            ifs.read(reinterpret_cast<char *>(data.data()), data.size());
        } else {
            httplib::Client cli(host_name.c_str());
            cli.set_follow_location(true);
            auto res = cli.Get(path_name.c_str());
            if (res && res->status == 200) {
                data.resize(res->body.size());
                std::memcpy(data.data(), res->body.c_str(), res->body.size());
            } else {
                throw std::runtime_error("Failed to download");
            }
        }

        auto dir_path = fs::temp_directory_path() / session_id;
        if (!fs::exists(dir_path)) {
            if (!fs::create_directory(dir_path)) {
                throw std::runtime_error("Failed to create temporary directory");
            }
        }

        if (end_with(url, ".zip")) {
            miniz_cpp::zip_file zf(data);
            zf.extractall(dir_path);
        } else {
            std::ofstream ofs(dir_path / fs::path(url).filename(), std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
        }

        for (auto& d : fs::directory_iterator(dir_path)) {
            paths_.push_back(d.path());
        }
        // Dictionary order
        std::sort(paths_.begin(), paths_.end());

     }

     cv::Mat get() {
        namespace fs = ghc::filesystem;

        auto path = paths_[idx_];

        auto frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) {
            throw std::runtime_error("Failed to load data : " + path.string());
        }
        idx_ = ((idx_+1) % paths_.size());

        return frame;
    }

 private:
    int32_t idx_;
    std::vector<ghc::filesystem::path> paths_;
};

extern "C" int ION_EXPORT ion_bb_image_io_grayscale_data_loader(halide_buffer_t *session_id_buf, halide_buffer_t *url_buf, int32_t width, int32_t height, halide_buffer_t *out) {

    namespace fs = ghc::filesystem;

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
            auto frame = seqs[session_id]->get();
            cv::normalize(frame, frame, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
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

#endif
