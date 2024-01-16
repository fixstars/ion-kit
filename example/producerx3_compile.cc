#include <ion/ion.h>

#include <vector>


#define STRINGIFY(N) #N

// #ifndef NAME_PREFIX
// #define NAME_PREFIX "producerx3"
// #endif

using namespace ion;

int main() {
    try {
        Port input{"input", Halide::type_of<uint8_t>()};
        Builder b;
        b.set_target(Halide::get_target_from_environment());
        b.with_bb_module("ion-bb");

        Node n = b.add("base_scalar_to_func_uint8")(input);
        n = b.add("base_extend_dimension_0d_uint8")(n["output"]);
        n = b.add("base_extend_dimension_1d_uint8")(n["output"]);
        n = b.add("base_extend_dimension_2d_uint8")(n["output"]);

        b.compile( STRINGIFY(NAME_PREFIX) );
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
