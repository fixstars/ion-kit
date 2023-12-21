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
                      "id": "dc4b82f0-497c-4f32-b210-3ad5cfd91143",
                      "name": "test_consumer",
                      "params": [],
                      "ports": [
                        {
                          "dimensions": 0,
                          "name": "output",
                          "index" : -1,
                          "node_id": "1310589d-1448-4107-ac1d-67c32f482906",
                          "array_size": 1,
                          "impl": 1,
                          "type": {
                            "bits": 0,
                            "code": 3,
                            "lanes": 0
                          }
                        },
                        {
                          "dimensions": 0,
                          "name": "min0",
                          "index" : -1,
                          "node_id": "",
                          "array_size": 1,
                          "impl": 2,
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "name": "extent0",
                          "index" : -1,
                          "node_id": "",
                          "array_size": 1,
                          "impl": 3,
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "name": "min1",
                          "index" : -1,
                          "node_id": "",
                          "array_size": 1,
                          "impl": 5,
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "name": "extent1",
                          "index" : -1,
                          "node_id": "",
                          "array_size": 1,
                          "impl": 6,
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        },
                        {
                          "dimensions": 0,
                          "name": "v",
                          "index" : -1,
                          "node_id": "",
                          "array_size": 1,
                          "impl": 7,
                          "type": {
                            "bits": 32,
                            "code": 0,
                            "lanes": 1
                          }
                        }
                      ],
                      "target": "host"
                    },
                    {
                      "id": "1310589d-1448-4107-ac1d-67c32f482906",
                      "name": "test_producer",
                      "params": [],
                      "ports": [],
                      "target": "host"
                    }
                  ],
                  "target": "host"
                }
                )";
            std::ofstream ofs(file_name);
            ofs << graph;
        }

        Halide::Type t = Halide::type_of<int32_t>();
        Port min0{"min0", t}, extent0{"extent0", t}, min1{"min1", t}, extent1{"extent1", t}, v{"v", t};

        Builder b;

        b.set_target(Halide::get_host_target());

        b.with_bb_module("ion-bb-test");

        b.load(file_name);

        PortMap pm;
        pm.set(min0, 0);
        pm.set(extent0, 2);
        pm.set(min1, 0);
        pm.set(extent1, 2);
        pm.set(v, 1);

        Halide::Buffer<int32_t> r = Halide::Buffer<int32_t>::make_scalar();
        for (auto& n : b.nodes()) {
            if (n.name() == "test_consumer") {
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
