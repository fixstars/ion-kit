#ifndef ION_BB_CORE_RT_H
#define ION_BB_CORE_RT_H

#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <HalideBuffer.h>

#include "httplib.h"

#ifdef _WIN32
#define ION_EXPORT __declspec(dllexport)
#else
#define ION_EXPORT
#endif

#if defined(ION_ENABLE_JIT_EXTERN)
#include <Halide.h>
namespace ion {
namespace bb {
namespace core {

std::map<std::string, Halide::ExternCFunction> extern_functions;

class RegisterExtern {
 public:
     RegisterExtern(std::string key, Halide::ExternCFunction f) {
         extern_functions[key] = f;
     }
};

} // image_io
} // bb
} // ion
#define ION_REGISTER_EXTERN(NAME) static auto ion_register_extern_##NAME = ion::bb::core::RegisterExtern(#NAME, NAME);
#else
#define ION_REGISTER_EXTERN(NAME)
#endif

namespace ion {
namespace bb {
namespace core {

std::tuple<std::string, std::string> parse_url(const std::string &url) {
    if (url.rfind("http://", 0) != 0) {  // not starts_with
        return std::tuple<std::string, std::string>("", "");
    }
    auto path_name_pos = url.find("/", 7);
    auto host_name = url.substr(0, path_name_pos);
    auto path_name = url.substr(path_name_pos);
    return std::tuple<std::string, std::string>(host_name, path_name);
}

template<typename T>
void fill_by_rng(std::mt19937 &rng, halide_buffer_t *range, halide_buffer_t *out) {
    T *p = reinterpret_cast<T *>(range->host);
    typename std::conditional<
        std::is_floating_point<T>::value,
        std::uniform_real_distribution<T>,
        std::uniform_int_distribution<T>>::type dist(p[0], p[1]);
    std::generate_n(reinterpret_cast<T *>(out->host), out->number_of_elements(), [&dist, &rng]() { return dist(rng); });
}

/* std::uniform_int_distribution doesn't accept uint8_t/int8_t as a template parameter under strict C++ standard */
template<>
void fill_by_rng<uint8_t>(std::mt19937 &rng, halide_buffer_t *range, halide_buffer_t *out) {
    uint8_t *p = reinterpret_cast<uint8_t *>(range->host);
    std::uniform_int_distribution<uint16_t> dist(p[0], p[1]);
    std::generate_n(reinterpret_cast<uint8_t *>(out->host), out->number_of_elements(), [&dist, &rng]() { return static_cast<uint8_t>(dist(rng)); });
}

template<>
void fill_by_rng<int8_t>(std::mt19937 &rng, halide_buffer_t *range, halide_buffer_t *out) {
    int8_t *p = reinterpret_cast<int8_t *>(range->host);
    std::uniform_int_distribution<int16_t> dist(p[0], p[1]);
    std::generate_n(reinterpret_cast<int8_t *>(out->host), out->number_of_elements(), [&dist, &rng]() { return static_cast<int8_t>(dist(rng)); });
}

std::unordered_map<std::string, std::vector<uint8_t>> buffer_cache;
std::unordered_map<int32_t, std::mt19937> rng_map;

}  // namespace core
}  // namespace bb
}  // namespace ion

extern "C" ION_EXPORT int ion_bb_core_buffer_loader(halide_buffer_t *url_buf, int32_t extent0, int32_t extent1, int32_t extent2, int32_t extent3, halide_buffer_t *out) {
    std::string url = std::string(reinterpret_cast<const char *>(url_buf->host));

    if (!out->is_bounds_query()) {
        auto it = ion::bb::core::buffer_cache.find(url);
        if (it != ion::bb::core::buffer_cache.end() && it->second.size() == out->size_in_bytes()) {
            memcpy(out->host, it->second.data(), it->second.size());
            return 0;
        } else {
            return -1;
        }
    }

    out->dim[0].min = 0;
    out->dim[0].extent = extent0;
    if (out->dimensions > 1) {
        out->dim[1].min = 0;
        out->dim[1].extent = extent1;
    }
    if (out->dimensions > 2) {
        out->dim[2].min = 0;
        out->dim[2].extent = extent2;
    }
    if (out->dimensions > 3) {
        out->dim[3].min = 0;
        out->dim[3].extent = extent3;
    }
    auto size = out->size_in_bytes();

    std::string host_name;
    std::string path_name;
    std::tie(host_name, path_name) = ion::bb::core::parse_url(url);

    bool img_loaded = false;
    std::vector<uint8_t> data(size);
    if (url.empty()) {
        // return all 0
    } else if (host_name.empty() || path_name.empty()) {
        // load local file
        std::ifstream ifs(url, std::ios::in | std::ios::binary);
        if (!ifs) {
            ifs.seekg(0, std::ios_base::end);
            if (ifs.tellg() == size) {  // check size
                ifs.seekg(0, std::ios_base::beg);
                ifs.read(reinterpret_cast<char *>(data.data()), size);
                img_loaded = true;
            }
        }
    } else {
        httplib::Client cli(host_name.c_str());
        cli.set_follow_location(true);
        auto res = cli.Get(path_name.c_str());
        if (res && res->status == 200 && res->body.size() == size) {
            std::memcpy(data.data(), res->body.c_str(), res->body.size());
            img_loaded = true;
        }
    }

    if (img_loaded) {
        ion::bb::core::buffer_cache[url] = data;
        return 0;
    } else {
        return -1;
    }
}

extern "C" int ION_EXPORT ion_bb_core_buffer_saver(halide_buffer_t *in, halide_buffer_t *path_buf, int32_t extent0, int32_t extent1, int32_t extent2, int32_t extent3, halide_buffer_t *out) {
    std::string path = std::string(reinterpret_cast<const char *>(path_buf->host));
    if (in->is_bounds_query()) {
        in->dim[0].min = 0;
        in->dim[0].extent = extent0;
        if (in->dimensions > 1) {
            in->dim[1].min = 0;
            in->dim[1].extent = extent1;
        }
        if (in->dimensions > 2) {
            in->dim[2].min = 0;
            in->dim[2].extent = extent2;
        }
        if (in->dimensions > 3) {
            in->dim[3].min = 0;
            in->dim[3].extent = extent3;
        }
    } else if (!path.empty()) {
        std::ofstream ofs(reinterpret_cast<const char *>(path_buf->host), std::ios::out | std::ios::binary);
        if (!ofs) {
            return -1;
        }
        ofs.write(reinterpret_cast<char *>(in->host), in->size_in_bytes());
    }

    return 0;
}

extern "C" ION_EXPORT int ion_bb_core_random_buffer(int32_t instance_id, int32_t seed, halide_buffer_t *range, int32_t extent0, int32_t extent1, int32_t extent2, int32_t extent3, halide_buffer_t *out) {
    if (out->is_bounds_query()) {
        out->dim[0].min = 0;
        out->dim[0].extent = extent0;
        if (out->dimensions > 1) {
            out->dim[1].min = 0;
            out->dim[1].extent = extent1;
        }
        if (out->dimensions > 2) {
            out->dim[2].min = 0;
            out->dim[2].extent = extent2;
        }
        if (out->dimensions > 3) {
            out->dim[3].min = 0;
            out->dim[3].extent = extent3;
        }

        return 0;
    }

    std::mt19937 rng;
    auto it = ion::bb::core::rng_map.find(instance_id);
    if (it != ion::bb::core::rng_map.end()) {
        rng = it->second;
    } else {
        rng = std::mt19937(seed);
    }

    if (out->type == halide_type_t(halide_type_uint, 8)) {
        ion::bb::core::fill_by_rng<uint8_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_uint, 16)) {
        ion::bb::core::fill_by_rng<uint16_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_uint, 32)) {
        ion::bb::core::fill_by_rng<uint32_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_uint, 64)) {
        ion::bb::core::fill_by_rng<uint64_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_int, 8)) {
        ion::bb::core::fill_by_rng<int8_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_int, 16)) {
        ion::bb::core::fill_by_rng<int16_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_int, 32)) {
        ion::bb::core::fill_by_rng<int32_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_int, 64)) {
        ion::bb::core::fill_by_rng<int64_t>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_float, 32)) {
        ion::bb::core::fill_by_rng<float>(rng, range, out);
    } else if (out->type == halide_type_t(halide_type_float, 64)) {
        ion::bb::core::fill_by_rng<double>(rng, range, out);
    } else {
        return -1;
    }

    return 0;
}

#undef ION_REGISTER_EXTERN
#undef ION_EXPORT

#endif
