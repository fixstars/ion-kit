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
             // NOTE: DSO which is loaded by with_bb_module should not be unloaded even if Builder is destructed.
             // Loading more than twice does not have any side effects.
#ifdef _WIN32
//             FreeLibrary(handle_);
#else
//             dlclose(handle_);
#endif
         }
     }

 private:

     std::string getErrorString(void) const {
         std::string error_msg;

#ifdef _WIN32
         DWORD error = GetLastError();
         if (error)
         {
             LPVOID lpMsgBuf;
             DWORD bufLen = FormatMessage(
                 FORMAT_MESSAGE_ALLOCATE_BUFFER |
                 FORMAT_MESSAGE_FROM_SYSTEM |
                 FORMAT_MESSAGE_IGNORE_INSERTS,
                 nullptr,
                 error,
                 MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
                 (LPTSTR) &lpMsgBuf, 0, nullptr );
             if (bufLen)
             {
                 LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
                 error_msg = std::string(lpMsgStr, lpMsgStr+bufLen);
                 LocalFree(lpMsgBuf);
             }
         }
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
