#ifndef ION_C_ION_H
#define ION_C_ION_H

#include <stdint.h>
#include <stdbool.h>

#if defined __cplusplus
extern "C" {
#endif

typedef enum {
    ion_type_int = 0,    //!< signed integers
    ion_type_uint = 1,   //!< unsigned integers
    ion_type_float = 2,  //!< floating point numbers
    ion_type_handle = 3  //!< opaque pointer type (void *)
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
typedef struct ion_buffer_t_ *ion_buffer_t;
typedef struct ion_port_map_t_ *ion_port_map_t;
typedef struct ion_graph_t_ *ion_graph_t;

int ion_port_create(ion_port_t *, const char *, ion_type_t, int);
int ion_port_create_with_index(ion_port_t *, ion_port_t, int);
int ion_port_destroy(ion_port_t);
int ion_port_bind_i8(ion_port_t, int8_t *);
int ion_port_bind_i16(ion_port_t, int16_t *);
int ion_port_bind_i32(ion_port_t, int32_t *);
int ion_port_bind_i64(ion_port_t, int64_t *);
int ion_port_bind_u1(ion_port_t, bool *);
int ion_port_bind_u8(ion_port_t, uint8_t *);
int ion_port_bind_u16(ion_port_t, uint16_t *);
int ion_port_bind_u32(ion_port_t, uint32_t *);
int ion_port_bind_u64(ion_port_t, uint64_t *);
int ion_port_bind_f32(ion_port_t, float *);
int ion_port_bind_f64(ion_port_t, double *);

int ion_port_bind_i8_array(ion_port_t, int8_t *, int);
int ion_port_bind_i16_array(ion_port_t, int16_t *, int);
int ion_port_bind_i32_array(ion_port_t, int32_t *, int);
int ion_port_bind_i64_array(ion_port_t, int64_t *, int);
int ion_port_bind_u1_array(ion_port_t, bool *, int);
int ion_port_bind_u8_array(ion_port_t, uint8_t *, int);
int ion_port_bind_u16_array(ion_port_t, uint16_t *, int);
int ion_port_bind_u32_array(ion_port_t, uint32_t *, int);
int ion_port_bind_u64_array(ion_port_t, uint64_t *, int);
int ion_port_bind_f32_array(ion_port_t, float *, int);
int ion_port_bind_f64_array(ion_port_t, double *, int);


int ion_port_bind_buffer(ion_port_t, ion_buffer_t);
int ion_port_bind_buffer_array(ion_port_t, ion_buffer_t *, int);

int ion_param_create(ion_param_t *, const char *, const char *);
int ion_param_destroy(ion_param_t);

int ion_node_create(ion_node_t *);
int ion_node_destroy(ion_node_t);
int ion_node_get_port(ion_node_t, const char *, ion_port_t *);
int ion_node_set_iports(ion_node_t, ion_port_t *, int);
int ion_node_set_params(ion_node_t, ion_param_t *, int);

int ion_builder_create(ion_builder_t *);
int ion_builder_destroy(ion_builder_t);
int ion_builder_set_target(ion_builder_t, const char *);
int ion_builder_with_bb_module(ion_builder_t, const char *);
int ion_builder_add_graph(ion_builder_t, const char *, ion_graph_t *);
int ion_builder_add_node(ion_builder_t, const char *, ion_node_t *);
int ion_builder_compile(ion_builder_t, const char *, ion_builder_compile_option_t option);
int ion_builder_save(ion_builder_t, const char *);
int ion_builder_load(ion_builder_t, const char *);
int ion_builder_bb_metadata(ion_builder_t, char *, int, int *);
int ion_builder_run(ion_builder_t);
int ion_builder_run_with_port_map(ion_builder_t, ion_port_map_t);

int ion_buffer_create(ion_buffer_t *, ion_type_t, int *, int);
int ion_buffer_create_with_data(ion_buffer_t *, ion_type_t, void *, int *, int);
int ion_buffer_destroy(ion_buffer_t);
int ion_buffer_write(ion_buffer_t, void *, int size);
int ion_buffer_read(ion_buffer_t, void *, int size);

int ion_graph_create(ion_graph_t *, ion_builder_t, const char *);
int ion_graph_add_node(ion_graph_t, const char *, ion_node_t *);
int ion_graph_destroy(ion_graph_t);
int ion_graph_run(ion_graph_t);
int ion_graph_create_with_multiple(ion_graph_t *ptr, ion_graph_t *objs, int size);

#if defined __cplusplus
}
#endif

#endif  // ION_C_ION_H
