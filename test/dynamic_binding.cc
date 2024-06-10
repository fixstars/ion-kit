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

            b.set_target(Halide::get_host_target());
            Graph g = b.add_graph("graph0");
            auto n = g.add("test_inc_by_offset")(in);
            n["output"].bind(out);

            Halide::Buffer<int32_t> out_buf = Halide::Buffer<int32_t>::make_scalar();
            out_buf.fill(0);
            for (auto& [pn, port] : n.dynamic_iports()) {
                port.bind(out_buf.data());
            }

           for (auto& [pn, port] : n.dynamic_oports()) {
                port.bind(out_buf);
            }

           for(int i = 0;i < 9;i++){
               g.run();
               if (out(0,0) !=  in(0,0)+ i) {
                   throw runtime_error("Unexpected out value");
               }
           }
        }
        {
            constexpr size_t h = 4, w = 4;

            Halide::Buffer<int32_t> in(w, h);
            in.fill(42);

            Halide::Buffer<int32_t> out(w, h);
            out.fill(0);

            Builder b;
            b.set_target(Halide::get_host_target());
            auto n = b.add("test_inc_by_offset")(in);
            n["output"].bind(out);

            Halide::Buffer<int32_t> out_buf = Halide::Buffer<int32_t>::make_scalar();
            out_buf.fill(0);
            for (auto& [pn, port] : n.dynamic_iports()) {
                port.bind(out_buf.data());
            }

           for (auto& [pn, port] : n.dynamic_oports()) {
                port.bind(out_buf);
            }

           for(int i = 0;i < 9;i++){
               b.run();
               if (out(0,0) !=  in(0,0)+ i) {
                   throw runtime_error("Unexpected out value");
               }
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
