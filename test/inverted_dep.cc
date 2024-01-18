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
                      "id": "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
                      "name": "test_consumer",
                      "params": [],
                      "ports": [
                        {
                          "dimensions": 0,
                          "impl_ptr": 93824992939168,
                          "index": -1,
                          "pred_chan": [
                            "51917e77-d626-47ff-b1be-37957a7d0706",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
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
                          "impl_ptr": 93824992925424,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "min0"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
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
                          "impl_ptr": 93824992925968,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "extent0"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
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
                          "impl_ptr": 93824992926512,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "min1"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
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
                          "impl_ptr": 93824992927056,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "extent1"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
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
                          "impl_ptr": 93824992927664,
                          "index": -1,
                          "pred_chan": [
                            "",
                            "v"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
                              "v"
                            ]
                          ],
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        }
                      ],
                      "target": "host-profile"
                    },
                    {
                      "id": "51917e77-d626-47ff-b1be-37957a7d0706",
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
                          "impl_ptr": 93824992939168,
                          "index": -1,
                          "pred_chan": [
                            "51917e77-d626-47ff-b1be-37957a7d0706",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "827bd8eb-b51c-4f0a-b94d-58dd3c521464",
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
                      "target": "host-profile"
                    }
                  ],
                  "target": "host-profile"
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
                n.iport("min0").bind(&min0);
                n.iport("extent0").bind(&extent0);
                n.iport("min1").bind(&min1);
                n.iport("extent1").bind(&extent1);
                n.iport("v").bind(&v);
                n.oport("output").bind(r);
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
