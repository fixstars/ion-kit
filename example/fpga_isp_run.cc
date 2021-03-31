#include <iostream>
#include <vector>

#include "fpga_isp.h"

#include "ion-bb-core/rt.h"
#include "ion-bb-fpga/rt.h"
#include "ion-bb-image-io/rt.h"
#include "ion-bb-image-processing/rt.h"

#include <HalideBuffer.h>

int main() {
    Halide::Runtime::Buffer<int32_t> output_buf(std::vector<int>{});

    fpga_isp(output_buf);
    halide_profiler_reset();
    fpga_isp(output_buf);

    return 0;
}
