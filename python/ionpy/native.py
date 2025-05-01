import ctypes
import os
import platform

pre_built_path = os.path.join(os.path.dirname(__file__), 'module')

if platform.system() == 'Windows':
    ion_core_module = os.path.join(pre_built_path, 'windows/ion-core.dll')
    ion_bb_module = os.path.join(pre_built_path, 'windows/ion-bb.dll')
    os.environ["PATH"] = '{};'.format(os.path.join(pre_built_path, 'windows')) + os.environ["PATH"]
    os.add_dll_directory(os.path.join(pre_built_path, 'windows'))
elif platform.system() == 'Darwin':
    ion_core_module = os.path.join(pre_built_path, 'macos/libion-core.dylib')
    ion_bb_module = os.path.join(pre_built_path, 'macos/libion-bb.dylib')
elif platform.system() == 'Linux':
    ion_core_module = os.path.join(pre_built_path, 'linux/libion-core.so')
    ion_bb_module = os.path.join(pre_built_path, 'linux/libion-bb.so')

ion_core = ctypes.cdll.LoadLibrary(ion_core_module)
ion_bb = ctypes.cdll.LoadLibrary(ion_bb_module)

class c_ion_type_t(ctypes.Structure):
    _fields_ = [
        ('code', ctypes.c_int), # ion_type_code_t (enum)
        ('bits', ctypes.c_uint8),
        ('lanes', ctypes.c_uint8),
    ]

class c_builder_compile_option_t(ctypes.Structure):
    _fields_ = [
        ('output_directory', ctypes.c_char_p),
    ]


c_ion_port_t = ctypes.POINTER(ctypes.c_int)
c_ion_param_t = ctypes.POINTER(ctypes.c_int)
c_ion_node_t = ctypes.POINTER(ctypes.c_int)

c_ion_builder_t = ctypes.POINTER(ctypes.c_int)
c_ion_graph_t = ctypes.POINTER(ctypes.c_int)
c_ion_buffer_t = ctypes.POINTER(ctypes.c_int)
c_ion_port_map_t = ctypes.POINTER(ctypes.c_int)

# int ion_port_create(ion_port_t *, const char *, ion_type_t, int);
ion_port_create = ion_core.ion_port_create
ion_port_create.restype = ctypes.c_int
ion_port_create.argtypes = [ ctypes.POINTER(c_ion_port_t), ctypes.c_char_p, c_ion_type_t, ctypes.c_int ]

# ion_port_create_with_index(ion_port_t*, ion_port_t, int);
ion_port_create_with_index = ion_core.ion_port_create_with_index
ion_port_create_with_index.restype = ctypes.c_int
ion_port_create_with_index.argtypes =[ctypes.POINTER(c_ion_port_t), c_ion_port_t, ctypes.c_int ]

# int ion_port_destroy(ion_port_t);
ion_port_destroy = ion_core.ion_port_destroy
ion_port_destroy.restype = ctypes.c_int
ion_port_destroy.argtypes = [ c_ion_port_t ]

# int ion_port_bind_i8(ion_port_t, int8_t*);
ion_port_bind_i8 = ion_core.ion_port_bind_i8
ion_port_bind_i8.restype = ctypes.c_int
ion_port_bind_i8.argtypes = [c_ion_port_t, ctypes.POINTER(ctypes.c_int8) ]

# int ion_port_bind_i16(ion_port_t, int16_t*);
ion_port_bind_i16 = ion_core.ion_port_bind_i16
ion_port_bind_i16.restype = ctypes.c_int
ion_port_bind_i16.argtypes = [c_ion_port_t, ctypes.POINTER(ctypes.c_int16) ]

# int ion_port_bind_i32(ion_port_t, int32_t*);
ion_port_bind_i32 = ion_core.ion_port_bind_i32
ion_port_bind_i32.restype = ctypes.c_int
ion_port_bind_i32.argtypes = [c_ion_port_t, ctypes.POINTER(ctypes.c_int32)]

# int ion_port_bind_i64(ion_port_t, int64_t*);
ion_port_bind_i64 = ion_core.ion_port_bind_i64
ion_port_bind_i64.restype = ctypes.c_int
ion_port_bind_i64.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_int64) ]

# int ion_port_map_set_u1(ion_port_t, bool*);
ion_port_bind_u1 = ion_core.ion_port_bind_u1
ion_port_bind_u1.restype = ctypes.c_int
ion_port_bind_u1.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_bool) ]

# int ion_port_bind_u8(ion_port_t, uint8_t*);
ion_port_bind_u8 = ion_core.ion_port_bind_u8
ion_port_bind_u8.restype = ctypes.c_int
ion_port_bind_u8.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint8) ]

# int ion_port_bind_u16(ion_port_t, uint16_t*);
ion_port_bind_u16 = ion_core.ion_port_bind_u16
ion_port_bind_u16.restype = ctypes.c_int
ion_port_bind_u16.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint16) ]

# int ion_port_bind_u32(ion_port_t, uint32_t*);
ion_port_bind_u32 = ion_core.ion_port_bind_u32
ion_port_bind_u32.restype = ctypes.c_int
ion_port_bind_u32.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint32) ]

# int ion_port_bind_u64(ion_port_t, uint64_t*);
ion_port_bind_u64 = ion_core.ion_port_bind_u64
ion_port_bind_u64.restype = ctypes.c_int
ion_port_bind_u64.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint64) ]

# int ion_port_bind_f32(ion_port_t, float*);
ion_port_bind_f32 = ion_core.ion_port_bind_f32
ion_port_bind_f32.restype = ctypes.c_int
ion_port_bind_f32.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_float) ]

# int ion_port_bind_f64(ion_port_t, double*;);
ion_port_bind_f64 = ion_core.ion_port_bind_f64
ion_port_bind_f64.restype = ctypes.c_int
ion_port_bind_f64.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_double) ]

# int ion_port_bind_i8_array(ion_port_t, int8_t*, int);
ion_port_bind_i8_array = ion_core.ion_port_bind_i8_array
ion_port_bind_i8_array.restype = ctypes.c_int
ion_port_bind_i8_array.argtypes = [c_ion_port_t, ctypes.POINTER(ctypes.c_int8), ctypes.c_int ]

# int ion_port_bind_i16_array(ion_port_t, int16_t*, int);
ion_port_bind_i16_array = ion_core.ion_port_bind_i16_array
ion_port_bind_i16_array.restype = ctypes.c_int
ion_port_bind_i16_array.argtypes = [c_ion_port_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_int ]

# int ion_port_bind_i32_array(ion_port_t, int32_t*, int);
ion_port_bind_i32_array = ion_core.ion_port_bind_i32_array
ion_port_bind_i32_array.restype = ctypes.c_int
ion_port_bind_i32_array.argtypes = [c_ion_port_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_int ]

# int ion_port_bind_i64_array(ion_port_t, int64_t*, int);
ion_port_bind_i64_array = ion_core.ion_port_bind_i64_array
ion_port_bind_i64_array.restype = ctypes.c_int
ion_port_bind_i64_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_int64), ctypes.c_int ]

# int ion_port_map_set_u1_array(ion_port_t, bool*, int);
ion_port_bind_u1_array = ion_core.ion_port_bind_u1_array
ion_port_bind_u1_array.restype = ctypes.c_int
ion_port_bind_u1_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_bool), ctypes.c_int ]

# int ion_port_bind_u8_array(ion_port_t, uint8_t*, int);
ion_port_bind_u8_array = ion_core.ion_port_bind_u8_array
ion_port_bind_u8_array.restype = ctypes.c_int
ion_port_bind_u8_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int ]

# int ion_port_bind_u16_array(ion_port_t, uint16_t*, int);
ion_port_bind_u16_array = ion_core.ion_port_bind_u16_array
ion_port_bind_u16_array.restype = ctypes.c_int
ion_port_bind_u16_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint16), ctypes.c_int ]

# int ion_port_bind_u32_array(ion_port_t, uint32_t*, int);
ion_port_bind_u32_array = ion_core.ion_port_bind_u32_array
ion_port_bind_u32_array.restype = ctypes.c_int
ion_port_bind_u32_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int ]

# int ion_port_bind_u64_array(ion_port_t, uint64_t*, int);
ion_port_bind_u64_array = ion_core.ion_port_bind_u64_array
ion_port_bind_u64_array.restype = ctypes.c_int
ion_port_bind_u64_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int ]

# int ion_port_bind_f32_array(ion_port_t, float*, int);
ion_port_bind_f32_array = ion_core.ion_port_bind_f32_array
ion_port_bind_f32_array.restype = ctypes.c_int
ion_port_bind_f32_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_float), ctypes.c_int ]

# int ion_port_bind_f64_array(ion_port_t, double*, int);
ion_port_bind_f64_array = ion_core.ion_port_bind_f64_array
ion_port_bind_f64_array.restype = ctypes.c_int
ion_port_bind_f64_array.argtypes = [ c_ion_port_t, ctypes.POINTER(ctypes.c_double), ctypes.c_int ]


# int ion_port_bind_buffer(ion_port_t, ion_buffer_t);
ion_port_bind_buffer = ion_core.ion_port_bind_buffer
ion_port_bind_buffer.restype = ctypes.c_int
ion_port_bind_buffer.argtypes = [c_ion_port_t, c_ion_buffer_t ]

# int ion_port_bind_buffer_array(ion_port_t obj, ion_buffer_t *bs, int n)
ion_port_bind_buffer_array = ion_core.ion_port_bind_buffer_array
ion_port_bind_buffer_array.restype = ctypes.c_int
ion_port_bind_buffer_array.argtypes = [c_ion_port_t, ctypes.POINTER(c_ion_buffer_t), ctypes.c_int]

# int ion_param_create(ion_param_t *, const char *, const char *);
ion_param_create = ion_core.ion_param_create
ion_param_create.restype = ctypes.c_int
ion_param_create.argtypes = [ ctypes.POINTER(c_ion_param_t), ctypes.c_char_p, ctypes.c_char_p ]

# int ion_param_destroy(ion_param_t);
ion_param_destroy = ion_core.ion_param_destroy
ion_param_destroy.restype = ctypes.c_int
ion_param_destroy.argtypes = [ c_ion_param_t ]


# int ion_node_create(ion_node_t *);
ion_node_create = ion_core.ion_node_create
ion_node_create.restype = ctypes.c_int
ion_node_create.argtypes = [ ctypes.POINTER(c_ion_node_t) ]

# int ion_node_destroy(ion_node_t);
ion_node_destroy = ion_core.ion_node_destroy
ion_node_destroy.restype = ctypes.c_int
ion_node_destroy.argtypes = [ c_ion_node_t ]


# int ion_node_get_port(ion_node_t, const char *, ion_port_t *);
ion_node_get_port = ion_core.ion_node_get_port
ion_node_get_port.restype = ctypes.c_int
ion_node_get_port.argtypes = [ c_ion_node_t, ctypes.c_char_p, ctypes.POINTER(c_ion_port_t) ]

# int ion_node_set_iports(ion_node_t, ion_port_t *, int);
ion_node_set_iports = ion_core.ion_node_set_iports
ion_node_set_iports.restype = ctypes.c_int
ion_node_set_iports.argtypes = [ c_ion_node_t, ctypes.POINTER(c_ion_port_t), ctypes.c_int ]

# int ion_node_set_params(ion_node_t, ion_param_t *, int);
ion_node_set_params = ion_core.ion_node_set_params
ion_node_set_params.restype = ctypes.c_int
ion_node_set_params.argtypes = [ c_ion_node_t, ctypes.POINTER(c_ion_param_t), ctypes.c_int ]


# int ion_builder_create(ion_builder_t *);
ion_builder_create = ion_core.ion_builder_create
ion_builder_create.restype = ctypes.c_int
ion_builder_create.argtypes = [ ctypes.POINTER(c_ion_builder_t) ]

# int ion_builder_destroy(ion_builder_t);
ion_builder_destroy = ion_core.ion_builder_destroy
ion_builder_destroy.restype = ctypes.c_int
ion_builder_destroy.argtypes = [ c_ion_builder_t ]


# int ion_builder_set_target(ion_builder_t, const char *);
# obj, target
ion_builder_set_target = ion_core.ion_builder_set_target
ion_builder_set_target.restype = ctypes.c_int
ion_builder_set_target.argtypes = [ c_ion_builder_t, ctypes.c_char_p ]

# int ion_builder_with_bb_module(ion_builder_t, const char *);
# obj, module_name
ion_builder_with_bb_module = ion_core.ion_builder_with_bb_module
ion_builder_with_bb_module.restype = ctypes.c_int
ion_builder_with_bb_module.argtypes = [ c_ion_builder_t, ctypes.c_char_p ]

# int ion_builder_add_node(ion_builder_t, const char *, ion_node_t *);
# obj, key, node_ptr
ion_builder_add_node = ion_core.ion_builder_add_node
ion_builder_add_node.restype = ctypes.c_int
ion_builder_add_node.argtypes = [ c_ion_builder_t, ctypes.c_char_p, ctypes.POINTER(c_ion_node_t) ]


# int ion_builder_compile(ion_builder_t, const char *, ion_builder_compile_option_t option);
# obj, function_name, option
ion_builder_compile = ion_core.ion_builder_compile
ion_builder_compile.restype = ctypes.c_int
ion_builder_compile.argtypes = [ c_ion_builder_t, ctypes.c_char_p, c_builder_compile_option_t ]


# int ion_builder_save(ion_builder_t, const char *);
ion_builder_save = ion_core.ion_builder_save
ion_builder_save.restype = ctypes.c_int
ion_builder_save.argtypes = [ c_ion_builder_t, ctypes.c_char_p ]

# int ion_builder_load(ion_builder_t, const char *);
ion_builder_load = ion_core.ion_builder_load
ion_builder_load.restype = ctypes.c_int
ion_builder_load.argtypes = [ c_ion_builder_t, ctypes.c_char_p ]


# int ion_builder_bb_metadata(ion_builder_t, char *, int, int *);
ion_builder_bb_metadata = ion_core.ion_builder_bb_metadata
ion_builder_bb_metadata.restype = ctypes.c_int
ion_builder_bb_metadata.argtypes = [ c_ion_builder_t, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int) ]

# int ion_builder_add_graph(ion_builder_t, const char , ion_graph_t *)
ion_builder_add_graph = ion_core.ion_builder_add_graph
ion_builder_add_graph.restype = ctypes.c_int
ion_builder_add_graph.argtypes = [ c_ion_builder_t, ctypes.c_char_p, ctypes.POINTER(c_ion_graph_t) ]


# int ion_builder_run(ion_builder_t, ion_port_map_t);
ion_builder_run = ion_core.ion_builder_run
ion_builder_run.restype = ctypes.c_int
ion_builder_run.argtypes = [ c_ion_builder_t ]

# int ion_buffer_create(ion_buffer_t *, ion_type_t, int *, int);
ion_buffer_create = ion_core.ion_buffer_create
ion_buffer_create.restype = ctypes.c_int
ion_buffer_create.argtypes = [ ctypes.POINTER(c_ion_buffer_t), c_ion_type_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int ]

# int ion_buffer_create_with_data(ion_buffer_t *, ion_type_t, void *, int *, int);
ion_buffer_create_with_data = ion_core.ion_buffer_create_with_data
ion_buffer_create_with_data.restype = ctypes.c_int
ion_buffer_create_with_data.argtypes = [ ctypes.POINTER(c_ion_buffer_t), c_ion_type_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int ]


# int ion_buffer_destroy(ion_buffer_t);
ion_buffer_destroy = ion_core.ion_buffer_destroy
ion_buffer_destroy.restype = ctypes.c_int
ion_buffer_destroy.argtypes = [ c_ion_buffer_t ]


# int ion_buffer_write(ion_buffer_t, void *, int size);
ion_buffer_write = ion_core.ion_buffer_write
ion_buffer_write.restype = ctypes.c_int
ion_buffer_write.argtypes = [ c_ion_buffer_t, ctypes.c_void_p, ctypes.c_int ]

# int ion_buffer_read(ion_buffer_t, void *, int size);
ion_buffer_read = ion_core.ion_buffer_read
ion_buffer_read.restype = ctypes.c_int
ion_buffer_read.argtypes = [ c_ion_buffer_t, ctypes.c_void_p, ctypes.c_int ]

# int ion_graph_create(ion_graph_t *, ion_builder_t, const char *)
ion_graph_create = ion_core.ion_graph_create
ion_graph_create.restype = ctypes.c_int
ion_graph_create.argtypes =[ ctypes.POINTER(c_ion_graph_t), c_ion_builder_t, ctypes.c_char_p ]

# int ion_graph_create_with_multiple(ion_graph_t*, ion_graph_t*, int size)
ion_graph_create_with_multiple = ion_core.ion_graph_create_with_multiple
ion_graph_create_with_multiple.restype = ctypes.c_int
ion_graph_create_with_multiple.argtypes = [  ctypes.POINTER(c_ion_graph_t), ctypes.POINTER(c_ion_graph_t), ctypes.c_int]

# int ion_graph_add_node(ion_graph_t, const char*, ion_node_t *)
ion_graph_add_node = ion_core.ion_graph_add_node
ion_graph_add_node.restype = ctypes.c_int
ion_graph_add_node.argtypes =[ c_ion_graph_t, ctypes.c_char_p, ctypes.POINTER(c_ion_node_t) ]

# int ion_graph_run(ion_graph_t)
ion_graph_run=ion_core.ion_graph_run
ion_graph_run.restype = ctypes.c_int
ion_graph_run.argtypes =[ c_ion_graph_t]

# ion_graph_destroy(ion_graph_t)
ion_graph_destroy=ion_core.ion_graph_destroy
ion_graph_destroy.restype = ctypes.c_int
ion_graph_destroy.argtypes =[ c_ion_graph_t]
