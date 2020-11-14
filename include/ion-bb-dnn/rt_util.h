#ifndef ION_BB_DNN_UTIL_H
#define ION_BB_DNN_UTIL_H

#include <dlfcn.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ion {
namespace bb {
namespace dnn {

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

    DynamicModule(const std::string &module_name, bool with_extension = false) {
        if (module_name == "") {
            handle_ = nullptr;
            return;
        }

#ifdef _WIN32
        auto file_name = with_extension ? module_name : module_name + ".dll";
        handle_ = LoadLibraryA(file_name.c_str());
        if (handle_ == nullptr) {
            return;
        }
#else
        auto file_name = with_extension ? module_name : "lib" + module_name + ".so";
        handle_ = dlopen(file_name.c_str(), RTLD_NOW);
        if (handle_ == nullptr) {
            return;
        }
#endif
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

    bool is_available(void) const {
        return handle_ != NULL;
    }

    template<typename T>
    T get_symbol(const std::string &symbol_name) const {
        void *func_handle;
#if defined(_WIN32)
        func_handle = GetProcAddress(handle_, symbol_name.c_str());
#else
        func_handle = dlsym(handle_, symbol_name.c_str());
#endif
        if (func_handle == nullptr) {
            throw std::runtime_error(get_error_string());
        }
        return reinterpret_cast<T>(func_handle);
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

typedef struct DetectionBox {
    float max_conf;
    int max_id;
    float x1, x2, y1, y2;
} DetectionBox;

float area(const DetectionBox &b) {
    return (b.x2 - b.x1) * (b.y2 - b.y1);
}

float intersection(const DetectionBox &a, const DetectionBox &b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = x2 - x1;
    const float h = y2 - y1;
    if (w <= 0 || h <= 0) return 0;
    return w * h;
}

float union_(const DetectionBox &a, const DetectionBox &b) {
    const auto area1 = area(a);
    const auto area2 = area(b);
    const auto inter = intersection(a, b);
    return area1 + area2 - inter;
}

void coco_render_boxes(cv::Mat &frame, const std::vector<DetectionBox> &boxes, const int w, const int h, int id_offset = 0) {

    static const std::map<int, std::pair<const char *, cv::Scalar>> label_color_map = {
        { 0, {"person", cv::Scalar(111, 221, 142)}},
        { 1, {"bicycle", cv::Scalar(199, 151, 121)}},
        { 2, {"car", cv::Scalar(145, 233, 34)}},
        { 3, {"motorbike", cv::Scalar(110, 131, 63)}},
        { 4, {"aeroplane", cv::Scalar(251, 141, 195)}},
        { 5, {"bus", cv::Scalar(136, 137, 194)}},
        { 6, {"train", cv::Scalar(114, 27, 34)}},
        { 7, {"truck", cv::Scalar(172, 221, 65)}},
        { 8, {"boat", cv::Scalar(7, 30, 178)}},
        { 9, {"traffic light", cv::Scalar(31, 28, 230)}},
        {10, {"fire hydrant", cv::Scalar(66, 214, 26)}},
        {12, {"stop sign", cv::Scalar(133, 39, 182)}},
        {13, {"parking meter", cv::Scalar(33, 20, 48)}},
        {14, {"bench", cv::Scalar(174, 253, 25)}},
        {15, {"bird", cv::Scalar(212, 160, 0)}},
        {16, {"cat", cv::Scalar(88, 78, 255)}},
        {17, {"dog", cv::Scalar(183, 35, 220)}},
        {18, {"horse", cv::Scalar(118, 157, 99)}},
        {19, {"sheep", cv::Scalar(81, 39, 129)}},
        {20, {"cow", cv::Scalar(253, 97, 253)}},
        {21, {"elephant", cv::Scalar(208, 170, 203)}},
        {22, {"bear", cv::Scalar(209, 175, 193)}},
        {23, {"zebra", cv::Scalar(43, 32, 163)}},
        {24, {"giraffe", cv::Scalar(246, 162, 213)}},
        {26, {"backpack", cv::Scalar(150, 199, 251)}},
        {27, {"umbrella", cv::Scalar(225, 165, 42)}},
        {30, {"handbag", cv::Scalar(56, 139, 51)}},
        {31, {"tie", cv::Scalar(235, 82, 61)}},
        {32, {"suitcase", cv::Scalar(219, 129, 248)}},
        {33, {"frisbee", cv::Scalar(120, 74, 139)}},
        {34, {"skis", cv::Scalar(164, 201, 240)}},
        {35, {"snowboard", cv::Scalar(238, 83, 85)}},
        {36, {"sports ball", cv::Scalar(134, 120, 102)}},
        {37, {"kite", cv::Scalar(166, 149, 183)}},
        {38, {"baseball bat", cv::Scalar(243, 13, 18)}},
        {39, {"baseball glove", cv::Scalar(56, 182, 85)}},
        {40, {"skateboard", cv::Scalar(117, 60, 48)}},
        {41, {"surfboard", cv::Scalar(109, 204, 30)}},
        {42, {"tennis racket", cv::Scalar(245, 221, 109)}},
        {43, {"bottle", cv::Scalar(74, 27, 47)}},
        {45, {"wine glass", cv::Scalar(229, 166, 29)}},
        {46, {"cup", cv::Scalar(158, 219, 241)}},
        {47, {"fork", cv::Scalar(95, 153, 84)}},
        {48, {"knife", cv::Scalar(218, 183, 12)}},
        {49, {"spoon", cv::Scalar(146, 37, 136)}},
        {50, {"bowl", cv::Scalar(63, 212, 25)}},
        {51, {"banana", cv::Scalar(174, 9, 96)}},
        {52, {"apple", cv::Scalar(180, 104, 193)}},
        {53, {"sandwich", cv::Scalar(160, 117, 33)}},
        {54, {"orange", cv::Scalar(224, 42, 115)}},
        {55, {"broccoli", cv::Scalar(9, 49, 96)}},
        {56, {"carrot", cv::Scalar(124, 213, 203)}},
        {57, {"hot dog", cv::Scalar(187, 193, 196)}},
        {58, {"pizza", cv::Scalar(57, 25, 171)}},
        {59, {"donut", cv::Scalar(189, 74, 145)}},
        {60, {"cake", cv::Scalar(73, 119, 11)}},
        {61, {"chair", cv::Scalar(37, 253, 178)}},
        {62, {"sofa", cv::Scalar(83, 223, 49)}},
        {63, {"pottedplant", cv::Scalar(111, 216, 113)}},
        {64, {"bed", cv::Scalar(167, 152, 203)}},
        {66, {"diningtable", cv::Scalar(99, 144, 184)}},
        {69, {"toilet", cv::Scalar(100, 204, 167)}},
        {71, {"tvmonitor", cv::Scalar(203, 87, 87)}},
        {72, {"laptop", cv::Scalar(139, 188, 41)}},
        {73, {"mouse", cv::Scalar(23, 84, 185)}},
        {74, {"remote", cv::Scalar(79, 160, 205)}},
        {75, {"keyboard", cv::Scalar(63, 7, 87)}},
        {76, {"cell phone", cv::Scalar(197, 255, 152)}},
        {77, {"microwave", cv::Scalar(199, 123, 207)}},
        {78, {"oven", cv::Scalar(211, 86, 200)}},
        {79, {"toaster", cv::Scalar(232, 184, 61)}},
        {80, {"sink", cv::Scalar(226, 254, 156)}},
        {81, {"refrigerator", cv::Scalar(195, 207, 141)}},
        {83, {"book", cv::Scalar(238, 101, 223)}},
        {84, {"clock", cv::Scalar(24, 84, 233)}},
        {85, {"vase", cv::Scalar(39, 104, 233)}},
        {86, {"scissors", cv::Scalar(49, 115, 78)}},
        {87, {"teddy bear", cv::Scalar(199, 193, 20)}},
        {88, {"hair drier", cv::Scalar(156, 85, 108)}},
        {89, {"toothbrush", cv::Scalar(189, 59, 8)}},
    };

    for (const auto &b : boxes) {
        const auto lc = label_color_map.at(b.max_id + id_offset);
        const auto label = lc.first;
        const auto color = lc.second / 255.0;
        const int x1 = b.x1 * w;
        const int y1 = b.y1 * h;
        const int x2 = b.x2 * w;
        const int y2 = b.y2 * h;
        const cv::Point2d p1(x1, y1);
        const cv::Point2d p2(x2, y2);
        cv::rectangle(frame, p1, p2, color);
        cv::putText(frame, label, cv::Point(x1, y1 - 3), cv::FONT_HERSHEY_COMPLEX, 0.5, color);
    }
}

}  // namespace dnn
}  // namespace bb
}  // namespace ion

#endif
