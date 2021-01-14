#include "ion/ion.h"

//#include "test-bb.h"
//#include "test-rt.h"
#include "harmony_bb/bb.h"
#include "harmony_bb/rt.h"

using namespace ion;

int main()
{
    Port rp_8x3{"rp_8x3", Halide::type_of<uint8_t>(), 3};
    auto t = Halide::type_of<int32_t>();
    Port out0_w_p{"out0_w_p", t}, out0_h_p{"out0_h_p", t}, seq_num_p{"seq_num_p", t};
    Param output_directory_ptr{"output_directory_ptr", "/tmp/out"};
    Param ext_name_ptr{"ext_name_ptr", ".jpg"};
    Builder b;
    b.set_target(Halide::get_host_target());
    Node n;
    n = b.add("harmony_imagesaver")(rp_8x3, out0_w_p, out0_h_p, seq_num_p).set_param(output_directory_ptr, ext_name_ptr);

    Halide::Buffer<uint8_t> ibuf(std::vector<int>{3, 16, 16});

    PortMap pm;
    pm.set(rp_8x3, ibuf);
    pm.set(out0_w_p, 16);
    pm.set(out0_h_p, 16);
    pm.set(seq_num_p, 0);

    b.run({}, pm);

    return 0;
}
