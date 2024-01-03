#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Halide.h>

#include "ion/c_ion.h"

int main()
{
    try {
        int ret = 0;
        {
            ion_type_t t = {.code=ion_type_int, .bits=32, .lanes=1};

            ion_port_t p;

            ret = ion_port_create(&p, "port", t, 0);
            if (ret != 0)
                return ret;

            ion_port_t p2;
            ret = ion_port_create_with_index(&p2, p, 1);
            if (ret != 0)
                return ret;

        }

        {
            ion_type_t t = {.code=ion_type_int, .bits=32, .lanes=1};

            ion_port_t min0, extent0, min1, extent1, v;
            ret = ion_port_create(&min0, "min0", t, 0);
            if (ret != 0)
                return ret;

            ret = ion_port_create(&extent0, "extent0", t, 0);
            if (ret != 0)
                return ret;

            ret = ion_port_create(&min1, "min1", t, 0);
            if (ret != 0)
                return ret;

            ret = ion_port_create(&extent1, "extent1", t, 0);
            if (ret != 0)
                return ret;

            ret = ion_port_create(&v, "v", t, 0);
            if (ret != 0)
                return ret;

            ion_param_t v41;
            ret = ion_param_create(&v41, "v", "41");
            if (ret != 0)
                return ret;

            ion_builder_t b;
            ret = ion_builder_create(&b);
            if (ret != 0)
                return ret;

            ret = ion_builder_set_target(b, "host");
            if (ret != 0)
                return ret;

            ret = ion_builder_with_bb_module(b, "ion-bb-test");
            if (ret != 0)
                return ret;

            ion_node_t n0;
            ret = ion_builder_add_node(b, "test_producer", &n0);
            if (ret != 0)
                return ret;

            ret = ion_node_set_param(n0, &v41, 1);
            if (ret != 0)
                return ret;

            ion_node_t n1;
            ret = ion_builder_add_node(b, "test_consumer", &n1);
            if (ret != 0)
                return ret;

            ion_port_t *ports = (ion_port_t*)malloc(6*sizeof(ion_port_t));

            ret = ion_node_get_port(n0, "output", &ports[0]);
            if (ret != 0)
                return ret;

            ports[1] = min0;
            ports[2] = extent0;
            ports[3] = min1;
            ports[4] = extent1;
            ports[5] = v;

            ret = ion_node_set_iport(n1, ports, 6);
            if (ret != 0)
                return ret;

            ret = ion_builder_save(b, "simple_graph.json");
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(min0);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(extent0);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(min1);
            if (ret != 0)
                return ret;


            ret = ion_port_destroy(extent1);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(v);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(ports[0]);
            if (ret != 0)
                return ret;

            ret = ion_node_destroy(n0);
            if (ret != 0)
                return ret;

            ret = ion_node_destroy(n1);
            if (ret != 0)
                return ret;

            ret = ion_builder_destroy(b);
            if (ret != 0)
                return ret;

            free(ports);
        }

        {
            ion_builder_t b;
            ret = ion_builder_create(&b);
            if (ret != 0)
                return ret;

            ret = ion_builder_load(b, "simple_graph.json");
            if (ret != 0)
                return ret;

            ret = ion_builder_with_bb_module(b, "ion-bb-test");
            if (ret != 0)
                return ret;

            ion_builder_compile_option_t op;
            op.output_directory = ".";

            ret = ion_builder_compile(b, "simple_graph", op);
            if (ret != 0)
                return ret;

            ret = ion_builder_destroy(b);
            if (ret != 0)
                return ret;
        }

        {
            ion_type_t t = {.code=ion_type_int, .bits=32, .lanes=1};
            ion_buffer_t b;
            int sizes[3] = {3, 2, 4};
            ret = ion_buffer_create(&b, t, sizes, 3);
            if (ret != 0) {
                return ret;
            }

            int buf1[3*2*4] = {
                0,  1,  2,  3,  4,  5,
                6,  7,  8,  9, 10, 11,
                12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23
            };

            ret = ion_buffer_write(b, buf1, 3 * 2 * 4 * sizeof(int));
            if (ret != 0) {
                return ret;
            }

            int buf2 [3 * 2 * 4] = {0};

            ret = ion_buffer_read(b, buf2, 3 * 2 * 4 * sizeof(int));
            if (ret != 0) {
                return ret;
            }

            for (int i=0; i<3*2*4; ++i) {
                if (buf1[i] != buf2[i]) {
                    return -1;
                }
            }
            ret = ion_buffer_destroy(b);
            if (ret != 0) {
                return ret;
            }
        }

        {
            ion_type_t t = {.code=ion_type_int, .bits=32, .lanes=1};

            ion_port_t ip;
            ret = ion_port_create(&ip, "input", t, 2);
            if (ret != 0)
                return ret;

            ion_param_t v41;
            ret = ion_param_create(&v41, "v", "41");
            if (ret != 0)
                return ret;

            ion_builder_t b;
            ret = ion_builder_create(&b);
            if (ret != 0)
                return ret;

            ret = ion_builder_set_target(b, "host");
            if (ret != 0)
                return ret;

            ret = ion_builder_with_bb_module(b, "ion-bb-test");
            if (ret != 0)
                return ret;

            ion_node_t n;
            ret = ion_builder_add_node(b, "test_inc_i32x2", &n);
            if (ret != 0)
                return ret;

            ret = ion_node_set_iport(n, &ip, 1);
            if (ret != 0)
                return ret;

            ret = ion_node_set_param(n, &v41, 1);
            if (ret != 0)
                return ret;

            int sizes[] = {16, 16};
            ion_buffer_t ibuf;
            ret = ion_buffer_create(&ibuf, t, sizes, 2);
            if (ret != 0)
                return ret;

            int in[16*16];
            for (int i=0; i<16*16; ++i) {
                in[i] = 1;
            }
            ret = ion_buffer_write(ibuf, in, 16*16*sizeof(int));
            if (ret != 0)
                return ret;

            ion_port_t op;
            ret = ion_node_get_port(n, "output", &op);
            if (ret != 0)
                return ret;

            ion_buffer_t obuf;
            ret = ion_buffer_create(&obuf, t, sizes, 2);
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(ip, ibuf);
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(op, obuf);
            if (ret != 0)
                return ret;

            ret = ion_builder_run(b);
            if (ret != 0)
                return ret;

            int out[16*16] = {0};
            ret = ion_buffer_read(obuf, out, 16*16*sizeof(int));
            for (int i=0;i<16*16; ++i) {
                if (out[i] != 42) {
                    printf("%d\n", out[i]);
                    return -1;
                }
            }

            ret = ion_buffer_destroy(ibuf);
            if (ret != 0)
                return ret;

            ret = ion_buffer_destroy(obuf);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(ip);
            if (ret != 0)
                return ret;

            ret = ion_node_destroy(n);
            if (ret != 0)
                return ret;

            ret = ion_builder_destroy(b);
            if (ret != 0)
                return ret;
        }

    } catch (Halide::Error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
