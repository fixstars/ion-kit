#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    try {
        Builder b;
        b.add("sonzai_shinai_bb");
        b.compile("sonzai_shinai_graph");
    } catch (...) {
        // Expected
        return 0;
    }
    return 1;
}
