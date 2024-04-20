#ifndef ION_BB_LLM_RT_H
#define ION_BB_LLM_RT_H

#include <Halide.h>

#include <llama.h>

#include "ion/export.h"

#include "log.h"
#include "json/json.hpp"

namespace ion {
namespace bb {
namespace llm {

std::map<std::string, Halide::ExternCFunction> extern_functions;

class RegisterExtern {
 public:
     RegisterExtern(std::string key, Halide::ExternCFunction f) {
         extern_functions[key] = f;
     }
};

} // llm
} // bb
} // ion

#define ION_REGISTER_EXTERN(NAME) static auto ion_register_extern_##NAME = ion::bb::llm::RegisterExtern(#NAME, NAME);

extern "C" 
ION_EXPORT int ion_bb_llm_llava(halide_buffer_t *in, halide_buffer_t *prompt, halide_buffer_t *out) {
    try {
        ion::log::info("ion_bb_llm_llava");
        // if (in->is_bounds_query()) {
        //     in->dim[0] = out->dim[0];
        //     in->dim[1] = out->dim[1];
        //     in->dim[2] = out->dim[2];
        //     return 0;
        // }

        return 0;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}
ION_REGISTER_EXTERN(ion_bb_llm_llava)

#undef ION_REGISTER_EXTERN

#endif  // ION_BB_LLM_BB_H
