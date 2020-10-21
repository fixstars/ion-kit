#ifndef ION_C_ION_H
#define ION_C_ION_H

#include <stdint.h>

#if defined __cplusplus
extern "C" {
#endif

typedef enum {
    ion_type_int = 0,   //!< signed integers
    ion_type_uint = 1,  //!< unsigned integers
    ion_type_float = 2, //!< floating point numbers
    ion_type_handle = 3 //!< opaque pointer type (void *)
} ion_type_code_t;

typedef struct {
    ion_type_code_t code;
    uint8_t bits;
    uint8_t lanes;
} ion_type_t;

typedef struct {
    const char *output_directory;
} ion_builder_compile_option_t;

typedef struct ion_port_t_ *ion_port_t;
typedef struct ion_param_t_ *ion_param_t;
typedef struct ion_node_t_ *ion_node_t;
typedef struct ion_builder_t_ *ion_builder_t;

int ion_port_create(ion_port_t *, const char *, ion_type_t);
int ion_port_destroy(ion_port_t);

int ion_param_create(ion_param_t *, const char *, const char *);
int ion_param_destroy(ion_param_t);

int ion_node_create(ion_node_t *);
int ion_node_destroy(ion_node_t);
int ion_node_get_port(ion_node_t, const char *, ion_port_t *);
int ion_node_set_port(ion_node_t, ion_port_t *, int);
int ion_node_set_param(ion_node_t, ion_param_t *, int);

int ion_builder_create(ion_builder_t *);
int ion_builder_destroy(ion_builder_t);
int ion_builder_set_target(ion_builder_t, const char *);
int ion_builder_with_bb_module(ion_builder_t, const char *);
int ion_builder_add_node(ion_builder_t, const char *, ion_node_t *);
int ion_builder_compile(ion_builder_t, const char *, ion_builder_compile_option_t *option);
int ion_builder_save(ion_builder_t, const char *);
int ion_builder_load(ion_builder_t, const char *);
int ion_builder_bb_metadata(ion_builder_t, char *, int, int *);

#if defined __cplusplus
}
#endif

#endif // ION_C_ION_H
