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
                      "id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                      "name": "test_consumer",
                      "params": [],
                      "ports": [
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171704963648,
                          "index": -1,
                          "pred_id": "842e6227-6960-4d05-9f6b-592b63dfd834",
                          "pred_name": "output",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "input",
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171705995072,
                          "index": -1,
                          "pred_id": "",
                          "pred_name": "",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "min0",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171703972800,
                          "index": -1,
                          "pred_id": "",
                          "pred_name": "",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "extent0",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171703640896,
                          "index": -1,
                          "pred_id": "",
                          "pred_name": "",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "min1",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171703968704,
                          "index": -1,
                          "pred_id": "",
                          "pred_name": "",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "extent1",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171703962192,
                          "index": -1,
                          "pred_id": "",
                          "pred_name": "",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "v",
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171704333952,
                          "index": -1,
                          "pred_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "pred_name": "output",
                          "size": 1,
                          "succ_id": "",
                          "succ_name": "",
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        }
                      ],
                      "target": "x86-64-linux-avx-avx2-debug-f16c-fma-sse41-trace_pipeline"
                    },
                    {
                      "id": "842e6227-6960-4d05-9f6b-592b63dfd834",
                      "name": "test_producer",
                      "params": [
                        {
                          "key": "v",
                          "val": "41"
                        }
                      ],
                      "ports": [
                        {
                          "dimensions": 0,
                          "impl_ptr": 94171704963648,
                          "index": -1,
                          "pred_id": "842e6227-6960-4d05-9f6b-592b63dfd834",
                          "pred_name": "output",
                          "size": 1,
                          "succ_id": "00a1e8eb-29bd-4cc3-b724-8551e09f60ac",
                          "succ_name": "input",
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        }
                      ],
                      "target": "x86-64-linux-avx-avx2-debug-f16c-fma-sse41-trace_pipeline"
                    }
                  ],
                  "target": "x86-64-linux-avx-avx2-debug-f16c-fma-sse41-trace_pipeline"
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

    std::cout << "Passed" << std::endl;

    return 0;
}
