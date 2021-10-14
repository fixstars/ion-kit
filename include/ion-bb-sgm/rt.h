#ifndef ION_BB_SGB_RT_H
#define ION_BB_SGB_RT_H

namespace ion {
namespace bb {
namespace sgm {

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

#endif
