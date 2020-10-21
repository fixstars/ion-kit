#ifndef ION_DYNAMIC_MODULE_H
#define ION_DYNAMIC_MODULE_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {}

namespace ion {

class DynamicModule {
 public:

#ifdef _WIN32
  using Handle = HMODULE;
#else
  using Handle = void*;
#endif

     DynamicModule(const std::string& module_path) {
         if (module_path == "") {
             handle_ = nullptr;
             return;
         }

#ifdef _WIN32
         handle_ = LoadLibraryA(module_path.c_str());
         if (handle_ == nullptr) {
             throw std::runtime_error(getErrorString());
         }
#else
         handle_ = dlopen(module_path.c_str(), RTLD_NOW);
         if (handle_ == nullptr) {
             throw std::runtime_error(getErrorString());
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

 private:

     std::string getErrorString(void) const {
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

} // namespace ion

#endif // ION_DYNAMIC_MODULE_H
