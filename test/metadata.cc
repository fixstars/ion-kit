#include <iostream>

#include "ion/ion.h"

using namespace ion;

int main()
{
    try {
        Builder b;
        b.with_bb_module("ion-bb-test");
        std::string md = b.bb_metadata();
        std::cout << md << std::endl;
        if (md.empty()) {
            return 1;
        }

    } catch (const Halide::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
