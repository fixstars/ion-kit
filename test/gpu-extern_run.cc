#include <vector>
#include <HalideBuffer.h>

#include "test-rt.h"
#include "gpu_extern.h"

using namespace Halide::Runtime;

int main()
{
    try {
        int size = 32;

        Buffer<int32_t> ibuf(std::vector<int32_t>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                ibuf(x, y) = 42;
            }
        }

        Buffer<int32_t> obuf(std::vector<int32_t>{size, size});
        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                obuf(x, y) = 0;
            }
        }

        auto ret = gpu_extern(ibuf, obuf);

        for (int y=0; y<size; ++y) {
            for (int x=0; x<size; ++x) {
                if (obuf(x, y) != 44) {
                    throw std::runtime_error("Invalid value");
                }
            }
        }

        std::cout << "Passed" << std::endl;

        return ret;

    } catch (const Halide::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
