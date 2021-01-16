#include <iostream>
#include <fstream>

#include "ion/ion.h"

#include "test-rt.h"

using namespace ion;

int main()
{
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
                      "dimensions_": 0,
                      "key_": "output",
                      "node_id_": "1310589d-1448-4107-ac1d-67c32f482906",
                      "type_": {
                        "bits": 0,
                        "code": 3,
                        "lanes": 0
                      }
                    },
                    {
                      "dimensions_": 0,
                      "key_": "min0",
                      "node_id_": "",
                      "type_": {
                        "bits": 32,
                        "code": 0,
                        "lanes": 1
                      }
                    },
                    {
                      "dimensions_": 0,
                      "key_": "extent0",
                      "node_id_": "",
                      "type_": {
                        "bits": 32,
                        "code": 0,
                        "lanes": 1
                      }
                    },
                    {
                      "dimensions_": 0,
                      "key_": "min1",
                      "node_id_": "",
                      "type_": {
                        "bits": 32,
                        "code": 0,
                        "lanes": 1
                      }
                    },
                    {
                      "dimensions_": 0,
                      "key_": "extent1",
                      "node_id_": "",
                      "type_": {
                        "bits": 32,
                        "code": 0,
                        "lanes": 1
                      }
                    },
                    {
                      "dimensions_": 0,
                      "key_": "v",
                      "node_id_": "",
                      "type_": {
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

    b.with_bb_module("libion-bb-test.so");

    Node n;
    n = b.add("test_producer");
    n = b.add("test_consumer")(n["output"], min0, extent0, min1, extent1, v);

    b.load(file_name);

    PortMap pm;
    pm.set(min0, 0);
    pm.set(extent0, 2);
    pm.set(min1, 0);
    pm.set(extent1, 2);
    pm.set(v, 1);

    Halide::Buffer<int32_t> r = Halide::Buffer<int32_t>::make_scalar();
    pm.set(n["output"], r);

    b.run(pm);
    return 0;
}
