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
                      "id": "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                      "name": "test_consumer",
                      "params": [],
                      "ports": [
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067312766832,
                          "index": -1,
                          "pred_chan": [
                            "2c706f47-6f51-4f1e-82de-f87f2dd0e9ab",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "input"
                            ]
                          ],
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067313268224,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "min0"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "min0"
                            ]
                          ],
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067313265008,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "extent0"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "extent0"
                            ]
                          ],
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067313264752,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "min1"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "min1"
                            ]
                          ],
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067312830512,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "extent1"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "extent1"
                            ]
                          ],
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067312767360,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "v"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "v"
                            ]
                          ],
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "impl_ptr": 94067312830256,
                          "index": -1,
                          "pred_chan": [
                            "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [],
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        }
                      ],
                      "target": "host-trace_pipeline"
                    },
                    {
                      "id": "2c706f47-6f51-4f1e-82de-f87f2dd0e9ab",
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
                          "impl_ptr": 94067312766832,
                          "index": -1,
                          "pred_chan": [
                            "2c706f47-6f51-4f1e-82de-f87f2dd0e9ab",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "9ebf9c1e-25bf-451d-b92e-54322c72476f",
                              "input"
                            ]
                          ],
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        }
                      ],
                      "target": "host-trace_pipeline"
                    }
                  ],
                  "target": "host-trace_pipeline"
                }
            )";
            std::ofstream ofs(file_name);
            ofs << graph;
        }

        int32_t min0 = 0, extent0 = 1, min1 = 0, extent1 = 2, v = 1;
        Halide::Buffer<int32_t> r = Halide::Buffer<int32_t>::make_scalar();

        Builder b;
        b.with_bb_module("ion-bb-test");
        b.set_target(Halide::get_host_target());
        b.load(file_name);

        for (auto& n : b.nodes()) {
            std::cout << n.name() << std::endl;
            if (n.name() == "test_consumer") {
                n["min0"].bind(&min0);
                n["extent0"].bind(&extent0);
                n["min1"].bind(&min1);
                n["extent1"].bind(&extent1);
                n["v"].bind(&v);
                n["output"].bind(r);
            }
        }

        b.run();
    
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
