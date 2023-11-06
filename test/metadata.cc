#include <iostream>

#include "ion/ion.h"

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
