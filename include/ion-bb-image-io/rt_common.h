#ifndef ION_BB_IMAGE_IO_RT_COMMON_H
#define ION_BB_IMAGE_IO_RT_COMMON_H

#include <algorithm>
#include <cstdio>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "httplib.h"
#include "zip_file.hpp"
#include "ghc/filesystem.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

namespace ion {
namespace bb {
namespace image_io {

template<typename... Rest>
std::string format(const char *fmt, const Rest &... rest) {
    int length = snprintf(NULL, 0, fmt, rest...) + 1;  // Explicit place for null termination
    std::vector<char> buf(length, 0);
    snprintf(&buf[0], length, fmt, rest...);
    std::string s(buf.begin(), std::find(buf.begin(), buf.end(), '\0'));
    return s;
}

class DynamicModule {
public:
#ifdef _WIN32
    using Handle = HMODULE;
#else
    using Handle = void *;
#endif

    DynamicModule(const std::string &module_name, bool essential) {
        if (module_name == "") {
            handle_ = nullptr;
            return;
        }

#ifdef _WIN32
        auto file_name = module_name + ".dll";
        handle_ = LoadLibraryA(file_name.c_str());

        if (handle_ != nullptr){
            //successfully loaded the module without the prefix of "lib".
            return;
        }

        file_name = "lib" + file_name;
        handle_ = LoadLibraryA(file_name.c_str());

#else
        auto file_name = "lib" + module_name + ".so";
        handle_ = dlopen(file_name.c_str(), RTLD_NOW);
#endif

        if (handle_ == nullptr) {
            if (essential) {
                throw std::runtime_error(get_error_string());
            } else {
                std::cerr << format("WARNING: Not found the not essential dynamic library: %s, it may work as a simulation mode", module_name.c_str());
            }
        }
    }

    ~DynamicModule() {
        if (handle_ != nullptr) {
#ifdef _WIN32
            FreeLibrary(handle_);
#else
            dlclose(handle_);
#endif
        }
    }

    DynamicModule(const std::string &module_name)
        : DynamicModule(module_name, true) {
    }

    bool is_available(void) const {
        return handle_ != NULL;
    }

    template<typename T>
    T get_symbol(const std::string &symbol_name) const {
#if defined(_WIN32)
        return reinterpret_cast<T>(GetProcAddress(handle_, symbol_name.c_str()));
#else
        return reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
#endif
    }

private:
    std::string get_error_string(void) const {
        std::string error_msg;

#ifdef _WIN32
        LPVOID lpMsgBuf;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                          FORMAT_MESSAGE_IGNORE_INSERTS,
                      nullptr, GetLastError(),
                      MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
                      (LPTSTR)&lpMsgBuf, 0, nullptr);
        std::size_t len = 0;
        wcstombs_s(&len, nullptr, 0, reinterpret_cast<const wchar_t *>(lpMsgBuf), _TRUNCATE);
        std::vector<char> buf(len + 1, 0);
        wcstombs_s(nullptr, &buf[0], len, reinterpret_cast<const wchar_t *>(lpMsgBuf), _TRUNCATE);
        error_msg.assign(buf.begin(), buf.end());
        LocalFree(lpMsgBuf);
#else
        const char *buf(dlerror());
        error_msg.assign(buf ? buf : "none");
#endif
        return error_msg;
    }

    Handle handle_;
};

std::tuple<std::string, std::string> parse_url(const std::string &url) {
    if (url.rfind("http://", 0) != 0) {  // not starts_with
        return std::tuple<std::string, std::string>("", "");
    }
    auto path_name_pos = url.find("/", 7);
    auto host_name = url.substr(0, path_name_pos);
    auto path_name = url.substr(path_name_pos);
    return std::tuple<std::string, std::string>(host_name, path_name);
}

#ifndef _WIN32
cv::Mat get_image(const std::string &url) {
    if (url.empty()) {
        return {};
    }

    std::string host_name;
    std::string path_name;
    std::tie(host_name, path_name) = parse_url(url);

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
            if (!img.empty()) {
                img_loaded = true;
            }
        }
    }

    if (img_loaded) {
        return img;
    } else {
        return {};
    }
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

        if (fs::path(url).extension() == ".zip") {
            miniz_cpp::zip_file zf(data);
            zf.extractall(dir_path.string());
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

     cv::Mat get(int width, int height, int imread_flags) {
        namespace fs = ghc::filesystem;

        cv::Mat frame;

        auto path = paths_[idx_];

        if (path.extension() == ".raw") {
            auto size = fs::file_size(path);
            switch (imread_flags) {
                case cv::IMREAD_GRAYSCALE:
                    if (size == width * height * sizeof(uint8_t)) {
                        frame = cv::Mat(height, width, CV_8UC1);
                    } else if (size == width * height * sizeof(uint16_t)) {
                        frame = cv::Mat(height, width, CV_16UC1);
                    } else {
                        throw std::runtime_error("Unsupported raw format");
                    }
                    break;
                case cv::IMREAD_COLOR:
                    if (size == 3 * width * height * sizeof(uint8_t)) {
                        // Expect interleaved RGB
                        frame = cv::Mat(height, width, CV_8UC3);
                    } else {
                        throw std::runtime_error("Unsupported raw format");
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported flags");
            }
            std::ifstream ifs(path, std::ios::binary);
            ifs.read(reinterpret_cast<char*>(frame.ptr()), size);
        } else {
            frame = cv::imread(path.string(), imread_flags);
            if (frame.empty()) {
                throw std::runtime_error("Failed to load data : " + path.string());
            }
        }
        idx_ = ((idx_+1) % paths_.size());

        return frame;
    }

 private:
    int32_t idx_;
    std::vector<ghc::filesystem::path> paths_;
};
#endif // _WIN32

}  // namespace image_io
}  // namespace bb
}  // namespace ion

#endif
