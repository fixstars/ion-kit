#ifndef ION_BB_DEMO_RT_COMMON_H
#define ION_BB_DEMO_RT_COMMON_H

#include <algorithm>
#include <cstdio>
#include <string>

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
namespace demo {

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
  using Handle = void*;
#endif

     DynamicModule(const std::string& module_name) {
         if (module_name == "") {
             handle_ = nullptr;
             return;
         }

#ifdef _WIN32
         auto file_name = module_name + ".dll";
         handle_ = LoadLibraryA(file_name.c_str());
         if (handle_ == nullptr) {
             throw std::runtime_error(get_error_string());
         }
#else
         auto file_name = "lib" + module_name + ".so";
         handle_ = dlopen(file_name.c_str(), RTLD_NOW);
         if (handle_ == nullptr) {
             throw std::runtime_error(get_error_string());
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

     bool is_available(void) const { return handle_ != NULL; }

     template <typename T>
     T get_symbol(const std::string& symbol_name) const {
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
         wcstombs_s(&len, nullptr, 0, reinterpret_cast<const wchar_t*>(lpMsgBuf), _TRUNCATE);
         std::vector<char> buf(len + 1, 0);
         wcstombs_s(nullptr, &buf[0], len, reinterpret_cast<const wchar_t*>(lpMsgBuf), _TRUNCATE);
         error_msg.assign(buf.begin(), buf.end());
         LocalFree(lpMsgBuf);
#else
         const char* buf(dlerror());
         error_msg.assign(buf ? buf : "none");
#endif
         return error_msg;
     }


     Handle handle_;
};

}  // namespace demo
}  // namespace bb
}  // namespace ion

#endif
