#ifndef ION_BB_LLM_BB_H
#define ION_BB_LLM_BB_H

#include <ion/ion.h>

namespace ion {
namespace bb {
namespace llm {

class X : public BuildingBlock<X> {
public:
    Input<Halide::Func> input{"input", Halide::type_of<float>(), 3};
    Output<Halide::Func> output{"output", Halide::type_of<float>(), 3};

    void generate() {
        output(c, x, y) = input(x, y, c);
    }

private:
    Halide::Var c, x, y;
};

}  // namespace llm
}  // namespace bb
}  // namespace ion

ION_REGISTER_BUILDING_BLOCK(ion::bb::llm::X, llm_x);

#endif  // ION_BB_LLM_BB_H
