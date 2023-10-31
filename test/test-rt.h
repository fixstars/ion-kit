#ifndef TEST_RT_H
#define TEST_RT_H

#include <iostream>
#include <cstdio>

#include <Halide.h>
#include <HalideBuffer.h>

#include "log.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

class DynamicModule {
public:
#ifdef _WIN32
    using Handle = HMODULE;
#else
    using Handle = void *;
#endif

    DynamicModule()
        : DynamicModule("") {
    }

    DynamicModule(const std::string &module_name, bool rtld_global = false, bool with_extension = false) {
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
        const int flag = RTLD_NOW | (rtld_global ? RTLD_GLOBAL : 0);
        handle_ = dlopen(file_name.c_str(), flag);
        if (handle_ == nullptr) {
            return;
        }
#endif
    }

    ~DynamicModule() {
        if (handle_ != nullptr) {
#ifdef _WIN32
            // NOTE: Do not closing DLL explicitly
            // to avoid SEGV at the ORT threadpoool.
#else
            // dlclose(handle_);
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

extern "C" DLLEXPORT
int consume(halide_buffer_t *in, int desired_min0, int desired_extent0, int desired_min1, int desired_extent1, int32_t v, halide_buffer_t *out) {
    if (in->is_bounds_query()) {
        in->dim[0].min = desired_min0;
        in->dim[0].extent = desired_extent0;
        in->dim[1].min = desired_min1;
        in->dim[1].extent = desired_extent1;
    } else {
        Halide::Runtime::Buffer<int32_t> ibuf(*in);
        for (int y=0; y<in->dim[1].extent; ++y) {
            for (int x=0; x<in->dim[0].extent; ++x) {
                std::cout << ibuf(x, y) + v << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}

extern "C" DLLEXPORT
int branch(halide_buffer_t *in, int32_t input_width, int32_t input_height, halide_buffer_t *out0, halide_buffer_t *out1) {
    if (in->is_bounds_query() || out0->is_bounds_query() || out1->is_bounds_query()) {
        if (out0->is_bounds_query()) {
            out0->dim[0].min = 0;
            out0->dim[0].extent = input_width;
            out0->dim[1].min = 0;
            out0->dim[1].extent = input_height/2;
        }
        if (out1->is_bounds_query()) {
            out1->dim[0].min = 0;
            out1->dim[0].extent = input_width;
            out1->dim[1].min = 0;
            out1->dim[1].extent = input_height/2;
        }
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = input_width;
            in->dim[1].min = 0;
            in->dim[1].extent = input_height;
        }
    } else {
        Halide::Runtime::Buffer<int32_t> ibuf(*in);
        Halide::Runtime::Buffer<int32_t> obuf0(*out0);
        Halide::Runtime::Buffer<int32_t> obuf1(*out1);
        for (int y=0; y<input_height/2; ++y) {
            for (int x=0; x<input_width; ++x) {
                obuf0(x, y) = ibuf(x, y);
                obuf1(x, y) = ibuf(x, y + input_height/2);
            }
        }
    }

    return 0;
}

using call_inc_kernel_t = void (*)(int32_t *in, int32_t width, int32_t height, int32_t v, int32_t *out);

extern "C" DLLEXPORT
int inc(halide_buffer_t *in, int32_t width, int32_t height, int32_t v, bool use_gpu, halide_buffer_t *out) {
    using namespace Halide;

    if (in->is_bounds_query() || out->is_bounds_query()) {
        if (out->is_bounds_query()) {
            out->dim[0].min = 0;
            out->dim[0].extent = width;
            out->dim[1].min = 0;
            out->dim[1].extent = height;
        }
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = width;
            in->dim[1].min = 0;
            in->dim[1].extent = height;
        }
    } else {

        ion::log::debug("in->host({:#x}), in->device({:#x}), out->host({:#x}), out->device({:#x})", reinterpret_cast<uint64_t>(in->host), in->device, reinterpret_cast<uint64_t>(out->host), out->device);

        Runtime::Buffer<int32_t> ibuf(*in);
        Runtime::Buffer<int32_t> obuf(*out);

        if (use_gpu) {
            auto device_api = get_device_interface_for_device_api(DeviceAPI::CUDA, get_host_target().with_feature(Target::CUDA));

            if (!ibuf.has_device_allocation()) {
                ibuf.device_malloc(device_api);
                ibuf.copy_to_device(device_api);
            }

            bool copy_to_host = false;
            if (!obuf.has_device_allocation()) {
                obuf.device_malloc(device_api);
                copy_to_host = true;
            }

            static DynamicModule dm("gpu-extern-lib");
            call_inc_kernel_t call_inc_kernel = dm.get_symbol<call_inc_kernel_t>("call_inc_kernel");
            call_inc_kernel(reinterpret_cast<int32_t*>(ibuf.raw_buffer()->device), width, height, v,
                            reinterpret_cast<int32_t*>(obuf.raw_buffer()->device));

            if (copy_to_host) {
                obuf.set_host_dirty(false);
                obuf.set_device_dirty();
                obuf.copy_to_host();
            }
        } else {
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    obuf(x, y) = ibuf(x, y) + v;
                }
            }
        }
    }

    return 0;
}
#undef DLLEXPORT

#endif
