#include <HalideBuffer.h>

#include "test-rt.h"
#include "simple_graph.h"

using namespace Halide::Runtime;

int main()
{
    Buffer<int32_t> out = Buffer<int32_t>::make_scalar();
    return simple_graph(2, 2, 0, 0, 1, out);
}
