#ifndef ION_BB_LLM_BB_H
#define ION_BB_LLM_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace llm {

class Llava : public BuildingBlock<Llava> {
public:
    Input<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    Input<Halide::Func> prompt{"prompt", Halide::type_of<uint8_t>(), 1};
    Output<Halide::Func> output{"output", Halide::type_of<uint8_t>(), 1};

    void generate() {
        std::vector<ExternFuncArgument> params = {prompt};
        Func llava("llava");
        llava.define_extern("ion_llm_llava", params, Halide::type_of<uint8_t>(), 1);
        llava.compute_root();

        output = llava;
    }

private:
    Halide::Var c, x, y;
};

}  // namespace llm
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::llm::Llava, llm_llava);

#endif  // ION_BB_LLM_BB_H
