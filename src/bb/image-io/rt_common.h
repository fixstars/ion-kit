#ifndef ION_BB_IMAGE_IO_RT_COMMON_H
#define ION_BB_IMAGE_IO_RT_COMMON_H

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "opencv_loader.h"

#include <Halide.h>
#include <HalideBuffer.h>
#include "halide_image_io.h"

#include "ion/export.h"


#include "log.h"

#include "httplib.h"

namespace zip_file {
#include "zip_file.hpp"
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
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

// class DynamicModule {
// public:
// #ifdef _WIN32
//     using Handle = HMODULE;
// #else
//     using Handle = void *;
// #endif
//
//     DynamicModule(const std::string &module_name, bool essential) {
//         ion::log::debug("Load module : Trying to load {}", module_name);
//         if (module_name == "") {
//             handle_ = nullptr;
//             return;
//         }
//
// #ifdef _WIN32
//         auto file_name = module_name + ".dll";
//         ion::log::debug("Load module : Looking for {}", file_name);
//         handle_ = LoadLibraryA(file_name.c_str());
//
//         if (handle_ != nullptr){
//             //successfully loaded the module without the prefix of "lib".
//             return;
//         }
//
//         file_name = "lib" + file_name;
//         ion::log::debug("Load module : Looking for {}", file_name);
//         handle_ = LoadLibraryA(file_name.c_str());
//
// #else
//         auto file_name = "lib" + module_name + ".so";
//         ion::log::debug("Load module : Looking for {}", file_name);
//         handle_ = dlopen(file_name.c_str(), RTLD_NOW);
// #endif
//
//         if (handle_ == nullptr) {
//             if (essential) {
//                 throw std::runtime_error(get_error_string());
//             } else {
//                 std::cerr << format("WARNING: Not found the not essential dynamic library: %s, it may work as a simulation mode", module_name.c_str());
//             }
//         }
//     }
//
//     ~DynamicModule() {
//         if (handle_ != nullptr) {
// #ifdef _WIN32
//             FreeLibrary(handle_);
// #else
//             dlclose(handle_);
// #endif
//         }
//     }
//
//     DynamicModule(const std::string &module_name)
//         : DynamicModule(module_name, true) {
//     }
//
//     bool is_available(void) const {
//         return handle_ != NULL;
//     }
//
//     template<typename T>
//     T get_symbol(const std::string &symbol_name) const {
// #if defined(_WIN32)
//         return reinterpret_cast<T>(GetProcAddress(handle_, symbol_name.c_str()));
// #else
//         return reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
// #endif
//     }
//
// private:
//     std::string get_error_string(void) const {
//         std::string error_msg;
//
// #ifdef _WIN32
//         LPVOID lpMsgBuf;
//         FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
//                           FORMAT_MESSAGE_IGNORE_INSERTS,
//                       nullptr, GetLastError(),
//                       MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
//                       (LPTSTR)&lpMsgBuf, 0, nullptr);
//         std::size_t len = 0;
//         wcstombs_s(&len, nullptr, 0, reinterpret_cast<const wchar_t *>(lpMsgBuf), _TRUNCATE);
//         std::vector<char> buf(len + 1, 0);
//         wcstombs_s(nullptr, &buf[0], len, reinterpret_cast<const wchar_t *>(lpMsgBuf), _TRUNCATE);
//         error_msg.assign(buf.begin(), buf.end());
//         LocalFree(lpMsgBuf);
// #else
//         const char *buf(dlerror());
//         error_msg.assign(buf ? buf : "none");
// #endif
//         return error_msg;
//     }
//
//     Handle handle_;
// };

std::tuple<std::string, std::string> parse_url(const std::string &url) {
    auto protocol_end_pos = url.find("://");
    if (protocol_end_pos == std::string::npos)
        return std::tuple<std::string, std::string>("", "");
    auto host_name_pos = protocol_end_pos + 3;
    auto path_name_pos = url.find("/", host_name_pos);
    auto host_name = url.substr(0, path_name_pos);
    auto path_name = url.substr(path_name_pos);
    return std::tuple<std::string, std::string>(host_name, path_name);
}

template<typename T>
class ImageSequence {

 public:
     ImageSequence(const std::string& session_id, const std::string& url) : idx_(0) {
        namespace fs = std::filesystem;

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
            zip_file::miniz_cpp::zip_file zf(data);
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

     void get(int width, int height, int imread_flags, Halide::Runtime::Buffer<T> &buf) {
        namespace fs = std::filesystem;

        auto path = paths_[idx_];
        auto size = fs::file_size(path);

        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> img_data(size);
        ifs.read(reinterpret_cast<char*>(img_data.data()), size);
        if (path.extension() == ".raw") {
            switch (imread_flags) {
                case IMREAD_GRAYSCALE:
                    if (size == width * height * sizeof(uint8_t)) {
                        Halide::Runtime::Buffer<uint8_t> buf_8(std::vector<int>{width, height}); //read in 8 bit
                        std::memcpy(buf_8.data(), img_data.data(), size);   // set_img_data
                        auto buf_16 = Halide::Tools::ImageTypeConversion::convert_image(buf_8, halide_type_of<uint16_t>());
                        buf.copy_from(buf_16);
                    } else if (size == width * height * sizeof(uint16_t)) {
                        std::memcpy(buf.data(), img_data.data(), size);
                    } else {
                        throw std::runtime_error("Unsupported raw format");
                    }
                    break;
                case IMREAD_COLOR:
                    if (size == 3 * width * height * sizeof(uint8_t)) {
                        // Expect interleaved RGB
                        Halide::Runtime::Buffer <uint8_t> buf_interleaved = Halide::Runtime::Buffer <uint8_t>::make_interleaved(width, height, 3); ;
                        std::memcpy(buf_interleaved.data(), img_data.data(), size);   // set_img_data
                        auto buffer_planar = buf_interleaved.copy_to_planar();
                        buf.copy_from(buffer_planar);
                    } else {
                        throw std::runtime_error("Unsupported raw format");
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported flags");
            }
        } else {
            switch (imread_flags) {
                case IMREAD_GRAYSCALE:
                {
                    Halide::Runtime::Buffer<T> img_buf = Halide::Tools::load_and_convert_image(path.string());
                    buf.copy_from(img_buf);
                }
                    break;
                case IMREAD_COLOR:
                  {
                    Halide::Runtime::Buffer<uint8_t> img_buf = Halide::Tools::load_image(path.string());
                    buf.copy_from(img_buf);
                  }
                    break;
                default:
                    throw std::runtime_error("Unsupported flags");
            }

        }
        idx_ = ((idx_+1) % paths_.size());
        return;
    }

 private:
    int32_t idx_;
    std::vector<std::filesystem::path> paths_;
};


struct rawHeader {

    // ---------- 0
    int version_;
    // ---------- 4
    int width_;
    int height_;
    // ---------- 12
    float r_gain0_;
    float g_gain0_;
    float b_gain0_;
    // ---------- 24
    float r_gain1_;
    float g_gain1_;
    float b_gain1_;
    // ---------- 36
    int offset0_x_;
    int offset0_y_;
    int offset1_x_;
    int offset1_y_;
    // ---------- 52
    int outputsize0_x_;
    int outputsize0_y_;
    int outputsize1_x_;
    int outputsize1_y_;
    // ---------- 68
    float fps_;
    // ---------- 72
    int pfnc_pixelformat;
    int group_id;
};

// PFNC
// https://www.emva.org/wp-content/uploads/GenICamPixelFormatValues.pdf
#define PFNC_Mono8 0x01080001 //PFNC Monochrome 8-bit
#define PFNC_Mono10 0x01100003 //PFNC Monochrome 10-bit unpacked
#define PFNC_Mono12 0x01100005 //PFNC Monochrome 12-bit unpacked
#define PFNC_RGB8 0x02180014 //PFNC Red-Green-Blue 8-bit
#define PFNC_BGR8 0x02180015 //PFNC Blue-Green-Red 8-bit

#define PFNC_BayerBG8 0x0108000B //PFNC Bayer Blue-Green 8-bit
#define PFNC_BayerBG10 0x0110000F //PFNC Bayer Blue-Green 10-bit unpacked
#define PFNC_BayerBG12 0x01100013 //PFNC Bayer Blue-Green 12-bit unpacked

#define PFNC_BayerGR8 0x01080008 //PFNC Bayer Green-Red 8-bit
#define PFNC_BayerGR12 0x01100010 //PFNC Bayer Green-Red 12-bit unpacked
#define PFNC_YCbCr422_8 0x0210003B //PFNC YCbCr 4:2:2 8-bit

}  // namespace image_io
}  // namespace bb
}  // namespace ion

#endif
