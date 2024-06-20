#include <exception>

#include "ion/ion.h"

#include "test-bb.h"
#include "test-rt.h"

using namespace std;
using namespace ion;

int main() {
    try {

        {
            constexpr size_t h = 4, w = 4;

            Halide::Buffer<int32_t> in(w, h);
            in.fill(42);

            Halide::Buffer<int32_t> out(w, h);
            out.fill(0);

            Builder b;
            Target target = Halide::get_host_target();
            target.set_feature(Target::Debug);
            b.set_target(target);

            auto n = b.add("test_inc_by_offset")(in);

            n = b.add("test_inc_by_offset")(n["output"]);

            n["output"].bind(out);

            Halide::Buffer<int32_t> param_buf = Halide::Buffer<int32_t>::make_scalar();
            param_buf.fill(1);

//            std::vector< int > sizes;
//            Halide::Buffer<int32_t> param_buf1(param_buf.data(),sizes);

            for(auto &n:b.nodes()){
                for (auto& [pn, port] : n.unbound_iports()) {
                      port.bind(param_buf);
                      n.set_iport(port);
                }

                for (auto& [pn, port] : n.unbound_oports()) {
                    port.bind(param_buf);
                    n.set_oport(port);
                }

            }

           b.run();
           for (int y = 0; y < h; ++y) {
               for (int x = 0; x < w; ++x) {
                   if (out(0,0) !=  45  ) {
                        throw runtime_error("Unexpected out value");
                   }
                }
            }

           if (param_buf(0) !=  3  ) {
               throw runtime_error("Unexpected value");
           }
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
