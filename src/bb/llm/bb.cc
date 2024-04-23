#include <fstream>

#include <Halide.h>

#include <llama.h>

#include "ion/export.h"

#include "log.h"
#include "json/json.hpp"

#include "clip.h"
#include "llava.h"

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
ION_EXPORT int ion_bb_llm_llava(halide_buffer_t *in, halide_buffer_t *prompt, int32_t width, int32_t height, halide_buffer_t *out) {
    try {
        if (in->is_bounds_query()) {
            in->dim[0].min = 0;
            in->dim[0].extent = 3;
            in->dim[1].min = 0;
            in->dim[1].extent = width;
            in->dim[2].min = 0;
            in->dim[2].extent = height;
            return 0;
        }

        Halide::Runtime::Buffer<int8_t> obuf(*out);

        std::ofstream ofs("test.bin");
        Halide::Runtime::Buffer<uint8_t> ibuf(*in);
        ofs.write(reinterpret_cast<const char*>(ibuf.data()), in->size_in_bytes());

        auto verbosity = 1;
        auto ctx_clip = clip_model_load("pasu", verbosity);

        llama_backend_init();
        // llama_numa_init(params->numa);

        auto n_threads = 1;
        auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, ibuf.data(), ibuf.size_in_bytes());
        if (!embed) {
            ion::log::error("Could not load image");
            return 1;
        }

        obuf.fill(0);
        obuf(0) = 'x';

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
