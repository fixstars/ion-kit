//#if (NAME_PREFIX == producerx3)
//#include "producerx3.h"
// #include NAME_PREFIX ".h"
//#include "producerx3_gpu.h"
//#endif

#include <HalideBuffer.h>

#include <iostream>

extern int NAME_PREFIX(uint8_t, halide_buffer_t *);

int main() {
    try {
        const std::vector<int> output_extents = {3, 512, 384};
        Halide::Runtime::Buffer<uint8_t> out(output_extents);

        const uint8_t in = 1;
        NAME_PREFIX(in, out);

        for (int y = 0; y < output_extents.at(2); ++y) {
            for (int x = 0; x < output_extents.at(1); ++x) {
                for (int c = 0; c < output_extents.at(0); ++c) {
                    if (out(c, x, y) != in) {
                        throw std::runtime_error("Output is not correct!");
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }

    return 0;
}
