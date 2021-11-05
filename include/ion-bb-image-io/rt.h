#ifndef ION_BB_IMAGE_IO_RT_H
#define ION_BB_IMAGE_IO_RT_H

#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

#include <HalideBuffer.h>

#if defined(ION_ENABLE_JIT_EXTERN)
#include <Halide.h>
namespace ion {
namespace bb {
namespace image_io {

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
#define ION_REGISTER_EXTERN(NAME) static auto ion_register_extern_##NAME = ion::bb::image_io::RegisterExtern(#NAME, NAME);
#else
#define ION_REGISTER_EXTERN(NAME)
#endif

#include "rt_display.h"
#include "rt_file.h"
#include "rt_realsense.h"
#include "rt_v4l2.h"
#include "rt_u3v.h"

#undef ION_REGISTER_EXTERN

#endif
