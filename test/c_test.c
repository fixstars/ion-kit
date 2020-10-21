#include <stdlib.h>

#include "ion/c_ion.h"

int main()
{
    int ret = 0;

    {
        ion_type_t t = {.code=ion_type_int, .bits=32, .lanes=1};

        ion_port_t min0, extent0, min1, extent1, v;
        ret = ion_port_create(&min0, "min0", t);
        if (ret != 0)
            return ret;

        ret = ion_port_create(&extent0, "extent0", t);
        if (ret != 0)
            return ret;

        ret = ion_port_create(&min1, "min1", t);
        if (ret != 0)
            return ret;

        ret = ion_port_create(&extent1, "extent1", t);
        if (ret != 0)
            return ret;

        ret = ion_port_create(&v, "v", t);
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

        // ret = ion_builder_with_bb_module(b, "./libion-bb-test.so");
        // if (ret != 0)
        //     return ret;

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

        ret = ion_node_set_port(n1, ports, 6);
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

        ret = ion_builder_with_bb_module(b, "./libion-bb-test.so");
        if (ret != 0)
            return ret;

        ion_builder_compile_option_t op;
        op.output_directory = ".";

        ret = ion_builder_compile(b, "simple_graph", &op);
        if (ret != 0)
            return ret;

        ret = ion_builder_destroy(b);
        if (ret != 0)
            return ret;
    }

    return 0;
}
