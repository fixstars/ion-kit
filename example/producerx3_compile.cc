#include <ion/ion.h>

#include "ion-bb-std/bb.h"

#include <vector>

using namespace ion;

int main() {
    try {
        Port input{"input", Halide::type_of<uint8_t>()};
        Builder b;
        b.set_target(Halide::get_target_from_environment());

        Node n = b.add("std_producer_u8x3")(input);

        b.compile("producerx3");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
