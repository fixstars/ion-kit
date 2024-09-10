#ifndef ION_DYNAMIC_MODULE_H
#define ION_DYNAMIC_MODULE_H

#include <filesystem>

#if _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#define ION_DYNAMIC_MODULE_PREFIX ""
#define ION_DYNAMIC_MODULE_EXT ".dll"
#elif __APPLE__
#include <dlfcn.h>
#define ION_DYNAMIC_MODULE_PREFIX "lib"
#define ION_DYNAMIC_MODULE_EXT ".dylib"
#else
#include <dlfcn.h>
#define ION_DYNAMIC_MODULE_PREFIX "lib"
#define ION_DYNAMIC_MODULE_EXT ".so"
#endif

#include "log.h"

namespace {
bool has_prefix_and_ext(const std::string &n) {
    return n.find(ION_DYNAMIC_MODULE_PREFIX) != std::string::npos && n.find(ION_DYNAMIC_MODULE_EXT) != std::string::npos;
}
}  // namespace

namespace ion {

class DynamicModule {
public:
#ifdef _WIN32
    using Handle = HMODULE;
#else
    using Handle = void *;
#endif

    DynamicModule(const std::string &module_name_or_path, bool essential = true, bool lazy_load = false) {
        if (module_name_or_path == "") {
            handle_ = nullptr;
            return;
        }

        // set target
        std::string target;
        if (std::filesystem::exists(module_name_or_path) || has_prefix_and_ext(module_name_or_path)) {
            // This is absolute path or file name
            target = module_name_or_path;
        } else {
            target = std::string(ION_DYNAMIC_MODULE_PREFIX) + module_name_or_path + std::string(ION_DYNAMIC_MODULE_EXT);
        }
        target_ = target;
        essential_ = essential;

        if (lazy_load) {
#ifdef _WIN32
            handle_ = nullptr;
#else
            handle_ = RTLD_DEFAULT;
#endif
        } else {

            // TODO: WIP: test moduel_name_or_path using std::filesystem
#ifdef _WIN32
            handle_ = LoadLibraryA(target.c_str());
#else
            handle_ = dlopen(target.c_str(), RTLD_NOW);
#endif
            if (handle_ == nullptr) {
                if (essential) {
                    throw std::runtime_error(getErrorString());
                } else {
                    log::warn("Not found inessential library {} : {}", target, getErrorString());
                }
            }
        }
    }

    ~DynamicModule() {
        if (handle_ != nullptr) {
            // NOTE: DSO which is loaded by with_bb_module should not be unloaded even if Builder is destructed.
            // Loading more than twice does not have any side effects.
        }
    }

    bool is_available(void) const {
        return handle_ != nullptr;
    }

    template<typename T>
    T get_symbol(const std::string &symbol_name) {
#ifdef _WIN32
        if (handle_ != nullptr) {
            return reinterpret_cast<T>(GetProcAddress(handle_, symbol_name.c_str()));
        } else {
            //  search symbol globally first
            Handle hmods[1024];  // the array size should be large enough
            DWORD cb_needed;
            if (EnumProcessModules(GetCurrentProcess(), hmods, sizeof(hmods), &cb_needed)) {
                for (unsigned int i = 0; i < (cb_needed / sizeof(HMODULE)); i++) {
                    char path[MAX_PATH];
                    // Get the module name
                    if (GetModuleFileNameA(hmods[i], path, sizeof(path) / sizeof(char))) {
                        // Try to get the address of the symbol if module path includes target_
                        std::string module_path(path);
                        if (module_path.find(target_) != std::string::npos) {
                            handle_ = hmods[i];
                            return reinterpret_cast<T>(GetProcAddress(hmods[i], symbol_name.c_str()));
                        }
                    }
                }
            }
            // failed to load symbol gloablly, load it explicitly
            handle_ = LoadLibraryA(target_.c_str());
            if (handle_ != nullptr) {
                log::info("Lazy loading library {}", target_);
            } else {
                if (essential_) {
                    throw std::runtime_error(getErrorString());
                }
            }
            return reinterpret_cast<T>(GetProcAddress(handle_, symbol_name.c_str()));
        }
#else
        if (handle_ == RTLD_DEFAULT) {
            void *func_ptr = dlsym(handle_, symbol_name.c_str());
            if (func_ptr != nullptr) {
                return reinterpret_cast<T>(func_ptr);
            } else {
                handle_ = dlopen(target_.c_str(), RTLD_NOW);
                if (handle_ != nullptr) {
                    log::info("Lazy loading library {}", target_);
                } else {
                    if (essential_) {
                        throw std::runtime_error(getErrorString());
                    }
                }
                return reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
            }
        } else {
            return reinterpret_cast<T>(dlsym(handle_, symbol_name.c_str()));
        }
#endif
    }

private:
    std::string getErrorString(void) const {
        std::string error_msg;

#ifdef _WIN32
        DWORD error = GetLastError();
        if (error) {
            LPVOID lpMsgBuf;
            DWORD bufLen = FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER |
                    FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr,
                error,
                MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
                (LPTSTR)&lpMsgBuf, 0, nullptr);
            if (bufLen) {
                LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
                error_msg = std::string(lpMsgStr, lpMsgStr + bufLen);
                LocalFree(lpMsgBuf);
            }
        }
#else
        const char *buf(dlerror());
        error_msg.assign(buf ? buf : "none");
#endif
        return error_msg;
    }

    Handle handle_;
    std::string target_;
    bool essential_;
};

}  // namespace ion

#endif  // ION_DYNAMIC_MODULE_H
