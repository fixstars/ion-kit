#ifndef ION_BB_LLM_BB_H
#define ION_BB_LLM_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace llm {

extern std::map<std::string, Halide::ExternCFunction> extern_functions;

class Llava : public BuildingBlock<Llava> {
public:
    Input<Halide::Func> input{"input", Halide::type_of<uint8_t>(), 3};
    Input<Halide::Func> prompt{"prompt", Halide::type_of<int8_t>(), 1};
    Output<Halide::Func> output{"output", Halide::type_of<int8_t>(), 1};
    BuildingBlockParam<int32_t> width{"width", 640};
    BuildingBlockParam<int32_t> height{"height", 480};

    void generate() {
        using namespace Halide; 

        // NOTE: These tricks is required for the input parameter which is passed as an external function argument
        Func input_;
        input_(_) = input(_);
        input_.compute_root();
        
        Func prompt_;
        prompt_(_) = prompt(_);
        prompt_.compute_root();

        std::vector<ExternFuncArgument> params = {input_, prompt_, static_cast<int32_t>(width), static_cast<int32_t>(height)};
        Func llava("llava");
        llava.define_extern("ion_bb_llm_llava", params, type_of<int8_t>(), 1);
        llava.compute_root();

        output = llava;

        this->register_disposer("ion_bb_llm_llava_dispose");
    }
};

}  // namespace llm
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::llm::Llava, llm_llava);

#endif  // ION_BB_LLM_BB_H
