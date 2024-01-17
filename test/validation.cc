#include "ion/ion.h"

#include "spdlog/cfg/helpers.h"
#include "spdlog/details/os.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

using namespace ion;

int main()
{
    try {
        Buffer<int32_t> input(2, 2);
        Buffer<int32_t> output(2, 2);

        // Unknown parameter
        {
            Builder b;
            b.with_bb_module("ion-bb-test");
            b.set_target(Halide::get_host_target());
            Node n;
            n = b.add("test_inc_i32x2")(input).set_param(Param("unknown-parameter", 1));
            n = b.add("test_inc_i32x2")(n["output"]);
            n["output"].bind(output);

            try {
                b.run();
            } catch (const std::exception& e) {
                // The error should thrown as runtime_error, not Halide::Error
                std::cerr << e.what() << std::endl;
            }
        }

        // Unknown output port 1
        {
            Builder b;
            b.with_bb_module("ion-bb-test");
            b.set_target(Halide::get_host_target());
            Node n;
            n = b.add("test_inc_i32x2")(input).set_param(Param("v", 41));
            n = b.add("test_inc_i32x2")(n["unknown-port"]);
            n["output"].bind(output);

            try {
                b.run();
            } catch (const std::exception& e) {
                // The error should thrown as runtime_error, not Halide::Error
                std::cerr << e.what() << std::endl;
            }
        }

        // Unknown output port 2
        {
            Builder b;
            b.with_bb_module("ion-bb-test");
            b.set_target(Halide::get_host_target());
            Node n;
            n = b.add("test_inc_i32x2")(input).set_param(Param("v", 41));
            n = b.add("test_inc_i32x2")(n["output"]);
            n["unknown-port"].bind(output);

            try {
                b.run();
            } catch (const std::exception& e) {
                // The error should thrown as runtime_error, not Halide::Error
                std::cerr << e.what() << std::endl;
            }
        }

        // Unknown input port 1
        {
            Builder b;
            b.with_bb_module("ion-bb-test");
            b.set_target(Halide::get_host_target());

            Buffer<int32_t> unknown(2, 2);

            Node n;
            n = b.add("test_inc_i32x2")(input, unknown).set_param(Param("v", 41));
            n = b.add("test_inc_i32x2")(n["output"]);
            n["output"].bind(output);

            try {
                b.run();
            } catch (const std::exception& e) {
                // The error should thrown as runtime_error, not Halide::Error
                std::cerr << e.what() << std::endl;
            }
        }

    } catch (Halide::Error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
