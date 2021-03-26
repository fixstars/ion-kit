#include <iostream>

#include "ion/ion.h"

#include "ion-bb-core/bb.h"
#include "ion-bb-dnn/bb.h"
#ifdef HALIDE_FOR_FPGA
#include "ion-bb-fpga/bb.h"
#endif
#include "ion-bb-image-io/bb.h"
#include "ion-bb-image-processing/bb.h"
#include "ion-bb-internal/bb.h"
#include "ion-bb-opencv/bb.h"
#include "ion-bb-sgm/bb.h"

using namespace ion;

int main()
{
    Builder b;
    b.with_bb_module("libion-bb-test.so");
    std::string md = b.bb_metadata();
    std::cout << md << std::endl;
    if (md.empty()) {
        return -1;
    }

    return 0;
}
