#include <iostream>
#include <HalideBuffer.h>

#include "test-rt.h"
#include "complex_graph.h"

using namespace Halide::Runtime;

int main()
{
    int32_t size = 16;
    int32_t split_n = 2;
    Buffer<int32_t> in(std::vector<int32_t>{size, size});
    for (int32_t y=0; y<size; ++y) {
         for (int32_t x=0; x<size; ++x) {
             in(x, y) = 40;
         }
    }
    Buffer<int32_t> out0(std::vector<int32_t>{size, size/split_n});
    Buffer<int32_t> out1(std::vector<int32_t>{size, size/split_n});
    for (int32_t y=0; y<size/split_n; ++y) {
         for (int32_t x=0; x<size; ++x) {
             out0(x, y) = 0;
             out1(x, y) = 0;
         }
    }
    int ret = complex_graph(in, size, size, out0, out1);

    for (int y=0; y<size/split_n; ++y) {
        for (int x=0; x<size; ++x) {
            std::cerr << out0(x, y) << " ";
            if (out0(x, y) != 41) {
                return -1;
            }
        }
        std::cerr << std::endl;
    }

    for (int y=0; y<size/split_n; ++y) {
        for (int x=0; x<size; ++x) {
            std::cerr << out1(x, y) << " ";
            if (out1(x, y) != 42) {
                return -1;
            }
        }
        std::cerr << std::endl;
    }

    return ret;
}
