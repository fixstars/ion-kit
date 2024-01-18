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
                      "id": "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
                      "name": "test_consumer",
                      "params": [],
                      "ports": [
                        {
                          "dimensions": 0,
                          "id": "2792b187-a42f-4c02-9399-25fc3acddd8e",
                          "index": -1,
                          "pred_chan": [
                            "c4fcbdba-7da4-4149-80ab-4ad5da37b435",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
                          "id": "b44a2f84-b7a2-40a4-9fbf-ed80078b6123",
                          "index": -1,
                          "pred_chan": [
                            "",
                            "min0"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
                          "id": "2f9ab162-f72a-42c8-8b92-2cbcf5ce71f7",
                          "index": -1,
                          "pred_chan": [
                            "",
                            "extent0"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
                          "id": "ba2f373c-2dd7-436f-b816-0ca59ca83037",
                          "index": -1,
                          "pred_chan": [
                            "",
                            "min1"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
                          "id": "537fd4b2-eef1-4c69-a04f-bd09adf3c93f",
                          "index": -1,
                          "pred_chan": [
                            "",
                            "extent1"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
                          "id": "80f24262-a521-43b7-8063-3b410fb5c509",
                          "index": -1,
                          "pred_chan": [
                            "",
                            "v"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
                      "id": "c4fcbdba-7da4-4149-80ab-4ad5da37b435",
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
                          "id": "2792b187-a42f-4c02-9399-25fc3acddd8e",
                          "index": -1,
                          "pred_chan": [
                            "c4fcbdba-7da4-4149-80ab-4ad5da37b435",
                            "output"
                          ],
                          "size": 1,
                          "succ_chans": [
                            [
                              "3de72ac3-d7e4-4de1-b73e-49856f8b5fc7",
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
