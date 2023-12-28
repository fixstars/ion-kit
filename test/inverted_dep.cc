#include <iostream>
#include <fstream>

#include "ion/ion.h"

#include "test-rt.h"

using namespace ion;

int main()
{
    try {
        const char *file_name = "test.graph";

        {
            std::string graph = R"(
                {
                  "nodes": [
                    {
                      "id": "158cea31-23df-45ef-a036-8bf209271804",
                      "name": "test_consumer",
                      "params": [],
                      "ports": [
                        {
                          "size": 1,
                          "dimensions": 0,
                          "impl_ptr": 93843319831024,
                          "index": -1,
                          "name": "output",
                          "node_id": "54f036a3-0b98-4d42-a343-6c7421d15f2f",
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        },
                        {
                          "size": 1,
                          "dimensions": 0,
                          "impl_ptr": 93843319821872,
                          "index": -1,
                          "name": "min0",
                          "node_id": "",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "size": 1,
                          "dimensions": 0,
                          "impl_ptr": 93843319822176,
                          "index": -1,
                          "name": "extent0",
                          "node_id": "",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "size": 1,
                          "dimensions": 0,
                          "impl_ptr": 93843319822480,
                          "index": -1,
                          "name": "min1",
                          "node_id": "",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "size": 1,
                          "dimensions": 0,
                          "impl_ptr": 93843319822784,
                          "index": -1,
                          "name": "extent1",
                          "node_id": "",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "size": 1,
                          "dimensions": 0,
                          "impl_ptr": 93843319823088,
                          "index": -1,
                          "name": "v",
                          "node_id": "",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        }
                      ],
                      "target": "x86-64-linux-avx-avx2-avx512-avx512_cannonlake-avx512_skylake-debug-f16c-fma-sse41-trace_pipeline"
                    },
                    {
                      "id": "54f036a3-0b98-4d42-a343-6c7421d15f2f",
                      "name": "test_producer",
                      "params": [
                        {
                          "key": "v",
                          "val": "41"
                        }
                      ],
                      "ports": [],
                      "target": "x86-64-linux-avx-avx2-avx512-avx512_cannonlake-avx512_skylake-debug-f16c-fma-sse41-trace_pipeline"
                    }
                  ],
                  "target": "x86-64-linux-avx-avx2-avx512-avx512_cannonlake-avx512_skylake-debug-f16c-fma-sse41-trace_pipeline"
                }
            )";
            std::ofstream ofs(file_name);
            ofs << graph;
        }

        Halide::Type t = Halide::type_of<int32_t>();

        Builder b;

        b.set_target(Halide::get_host_target());

        b.with_bb_module("ion-bb-test");

        b.load(file_name);

        PortMap pm;

        Halide::Buffer<int32_t> r = Halide::Buffer<int32_t>::make_scalar();
        for (auto& n : b.nodes()) {
            if (n.name() == "test_consumer") {
                pm.set(n["min0"], 0);
                pm.set(n["extent0"], 1);
                pm.set(n["min1"], 0);
                pm.set(n["extent1"], 2);
                pm.set(n["v"], 1);
                pm.set(n["output"], r);
                break;
            }
        }

        b.run(pm);
    } catch (const Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }


    return 0;
}
