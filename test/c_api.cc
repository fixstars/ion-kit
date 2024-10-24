#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Halide.h>

#include "ion/c_ion.h"

int main() {
    try {
        int ret = 0;
        {
            ion_type_t t = {.code = ion_type_int, .bits = 32, .lanes = 1};

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
            ion_type_t t = {.code = ion_type_int, .bits = 32, .lanes = 1};

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

            ret = ion_node_set_params(n0, &v41, 1);
            if (ret != 0)
                return ret;

            ion_node_t n1;
            ret = ion_builder_add_node(b, "test_consumer", &n1);
            if (ret != 0)
                return ret;

            ion_port_t *ports = (ion_port_t *)malloc(6 * sizeof(ion_port_t));

            ret = ion_node_get_port(n0, "output", &ports[0]);
            if (ret != 0)
                return ret;

            ports[1] = min0;
            ports[2] = extent0;
            ports[3] = min1;
            ports[4] = extent1;
            ports[5] = v;

            ret = ion_node_set_iports(n1, ports, 6);
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
            ion_type_t t = {.code = ion_type_int, .bits = 32, .lanes = 1};
            ion_buffer_t b;
            int sizes[3] = {3, 2, 4};
            ret = ion_buffer_create(&b, t, sizes, 3);
            if (ret != 0) {
                return ret;
            }

            int buf1[3 * 2 * 4] = {
                0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23};

            ret = ion_buffer_write(b, buf1, 3 * 2 * 4 * sizeof(int));
            if (ret != 0) {
                return ret;
            }

            int buf2[3 * 2 * 4] = {0};

            ret = ion_buffer_read(b, buf2, 3 * 2 * 4 * sizeof(int));
            if (ret != 0) {
                return ret;
            }

            for (int i = 0; i < 3 * 2 * 4; ++i) {
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
            ion_type_t t = {.code = ion_type_int, .bits = 32, .lanes = 1};

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

            ret = ion_node_set_iports(n, &ip, 1);
            if (ret != 0)
                return ret;

            ret = ion_node_set_params(n, &v41, 1);
            if (ret != 0)
                return ret;

            int sizes[] = {16, 16};
            ion_buffer_t ibuf;
            ret = ion_buffer_create(&ibuf, t, sizes, 2);
            if (ret != 0)
                return ret;

            int in[16 * 16];
            for (int i = 0; i < 16 * 16; ++i) {
                in[i] = 1;
            }
            ret = ion_buffer_write(ibuf, in, 16 * 16 * sizeof(int));
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

            int out[16 * 16] = {0};
            ret = ion_buffer_read(obuf, out, 16 * 16 * sizeof(int));
            for (int i = 0; i < 16 * 16; ++i) {
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

        {
            ion_type_t t = {.code = ion_type_int, .bits = 32, .lanes = 1};

            ion_port_t ip;
            ret = ion_port_create(&ip, "input", t, 2);
            if (ret != 0)
                return ret;


            ion_port_t offsets_p;
            ret = ion_port_create(&offsets_p, "input_offsets", t, 0);
            if (ret != 0)
                return ret;

            ion_param_t len;
            ret = ion_param_create(&len, "input_offsets.size", "4");
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
            ret = ion_builder_add_node(b, "test_scalar_array", &n);
            if (ret != 0)
                return ret;

            ret = ion_node_set_params(n, &len, 1);
            if (ret != 0)
                return ret;

            ion_port_t *ports = (ion_port_t *) malloc(2 * sizeof(ion_port_t));
            ports[0] = ip;
            ports[1] = offsets_p;
            ret = ion_node_set_iports(n, ports, 2);
            if (ret != 0)
                return ret;

            int sizes[] = {4, 4};
            ion_buffer_t ibuf;
            ret = ion_buffer_create(&ibuf, t, sizes, 2);
            if (ret != 0)
                return ret;

            int in[4 * 4];
            for (int i = 0; i < 4 * 4; ++i) {
                in[i] = 42;
            }
            ret = ion_buffer_write(ibuf, in, 4 * 4 * sizeof(int));
            if (ret != 0)
                return ret;

            ion_port_t op;
            ret = ion_node_get_port(n, "output", &op);
            if (ret != 0)
                return ret;

            ion_buffer_t *obufs = (ion_buffer_t *) malloc(4 * sizeof(ion_buffer_t));
            for (int i = 0; i < 4; ++i) {
                ret = ion_buffer_create(obufs + i, t, sizes, 2);
                if (ret != 0)
                    return ret;
            }

            int in_offsets[4];
            for (int i = 0; i < 4; ++i) {
                in_offsets[i] = i;
            }

            ret = ion_port_bind_i32_array(offsets_p, (int *) (&in_offsets), 4);
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(ip, ibuf);
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer_array(op, obufs, 4);
            if (ret != 0)
                return ret;

            ret = ion_builder_run(b);
            if (ret != 0)
                return ret;

            for (int i = 0;i < 4 ;i++){
                 int out[4 * 4] = {0};
                ret = ion_buffer_read(*(obufs + i), out, 4 * 4 * sizeof(int));
                if (ret != 0)
                    return ret;
                if (out[0] != 42 + i) {
                    printf("%d\n", out[0]);
                    return -1;
                }
            }

            ret = ion_port_destroy(ip);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(offsets_p);
            if (ret != 0)
                return ret;

            ret = ion_port_destroy(op);
            if (ret != 0)
                return ret;

            ret = ion_builder_destroy(b);
            if (ret != 0)
                return ret;

            free(ports);
        }

        {

            ion_type_t t = {.code = ion_type_int, .bits = 32, .lanes = 1};

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

            ion_graph_t g0;
            ret = ion_builder_add_graph(b, "graph0", &g0);
            if (ret != 0)
                return ret;

            ion_param_t v41;
            ret = ion_param_create(&v41, "v", "41");
            if (ret != 0)
                return ret;

            ion_node_t n0;
            ret = ion_graph_add_node(g0, "test_inc_i32x2", &n0);
            if (ret != 0)
                return ret;
            int sizes[] = {16, 16};

            ret = ion_node_set_params(n0, &v41, 1);
            if (ret != 0)
                return ret;

            ion_port_t ip0;
            ret = ion_port_create(&ip0, "input0", t, 2);
            if (ret != 0)
                return ret;
            ret = ion_node_set_iports(n0, &ip0, 1);
            if (ret != 0)
                return ret;

            ion_buffer_t ibuf0;
            ret = ion_buffer_create(&ibuf0, t, sizes, 2);
            if (ret != 0)
                return ret;

            int in0[16 * 16];
            for (int i = 0; i < 16 * 16; ++i) {
                in0[i] = 0;
            }
            ret = ion_buffer_write(ibuf0, in0, 16 * 16 * sizeof(int));
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(ip0, ibuf0);
            if (ret != 0)
                return ret;

            ion_port_t op0;
            ret = ion_node_get_port(n0, "output", &op0);
            if (ret != 0)
                return ret;

            ion_buffer_t obuf0;
            ret = ion_buffer_create(&obuf0, t, sizes, 2);
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(op0, obuf0);
            if (ret != 0)
                return ret;
            //
            //            ret = ion_graph_run(g0);
            //            if (ret != 0)
            //                return ret;

            int out0[16 * 16] = {0};
            //            ret = ion_buffer_read(obuf0, out0, 16*16*sizeof(int));
            //            for (int i=0;i<16*16; ++i) {
            //                if (out0[i] != 41) {
            //                    printf("out0: %d\n", out0[i]);
            //
            //                }
            //            }

            ion_graph_t g1;
            ret = ion_builder_add_graph(b, "graph1", &g1);
            if (ret != 0)
                return ret;

            ion_node_t n1;
            ret = ion_graph_add_node(g1, "test_inc_i32x2", &n1);
            if (ret != 0)
                return ret;

            ret = ion_node_set_params(n1, &v41, 1);
            if (ret != 0)
                return ret;

            ion_port_t ip1;
            ret = ion_port_create(&ip1, "input1", t, 2);
            if (ret != 0)
                return ret;
            ret = ion_node_set_iports(n1, &ip1, 1);
            if (ret != 0)
                return ret;

            ion_buffer_t ibuf1;
            ret = ion_buffer_create(&ibuf1, t, sizes, 2);
            if (ret != 0)
                return ret;

            int in1[16 * 16];
            for (int i = 0; i < 16 * 16; ++i) {
                in1[i] = 1;
            }
            ret = ion_buffer_write(ibuf1, in1, 16 * 16 * sizeof(int));
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(ip1, ibuf1);
            if (ret != 0)
                return ret;

            ion_port_t op1;
            ret = ion_node_get_port(n1, "output", &op1);
            if (ret != 0)
                return ret;

            ion_buffer_t obuf1;
            ret = ion_buffer_create(&obuf1, t, sizes, 2);
            if (ret != 0)
                return ret;

            ret = ion_port_bind_buffer(op1, obuf1);
            if (ret != 0)
                return ret;

            //            ret = ion_graph_run(g1);
            if (ret != 0)
                return ret;

            int out1[16 * 16] = {0};
            //            ret = ion_buffer_read(obuf1, out1, 16*16*sizeof(int));
            //            for (int i=0;i<16*16; ++i) {
            //                if (out1[i] != 42) {
            //                    printf("out1: %d\n", out1[i]);
            //                }
            //            }

            for (int i = 0; i < 16 * 16; ++i) {
                out0[i] = 0;
                out1[i] = 0;
            }

            ion_graph_t g2;
            ion_graph_create(&g2, b, "graph2");

            ion_graph_t *graphs = (ion_graph_t *)malloc(2 * sizeof(ion_graph_t));
            graphs[0] = g0;
            graphs[1] = g1;

            ret = ion_graph_create_with_multiple(&g2, graphs, 2);
            if (ret != 0)
                return ret;
            ret = ion_graph_run(g2);
            if (ret != 0)
                return ret;
            ret = ion_buffer_read(obuf0, out0, 16 * 16 * sizeof(int));
            ret = ion_buffer_read(obuf1, out1, 16 * 16 * sizeof(int));

            for (int i = 0; i < 16 * 16; ++i) {
                if (out0[i] != 41) {
                    printf("out0: %d\n", out0[i]);
                    ;
                }
                if (out1[i] != 42) {
                    printf("out1: %d\n", out1[i]);
                }
            }
            ret = ion_graph_destroy(g0);
            ret = ion_graph_destroy(g1);
            ret = ion_graph_destroy(g2);
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
