#
# ion-core Testing
#

# Test BB module
add_library(ion-bb-test SHARED ion-bb-test.cc)
target_include_directories(ion-bb-test PUBLIC ${PROJECT_SOURCE_DIR}/src)
if(UNIX)
    target_compile_options(ion-bb-test PUBLIC -fno-rtti)
endif()
target_link_libraries(ion-bb-test ion-core)

# C interface test
ion_jit_executable(c_api SRCS c_api.cc)

# Error case test
ion_jit_executable(error SRCS error.cc)

# Metadata test
ion_jit_executable(metadata SRCS metadata.cc)

# Multi-out test
ion_jit_executable(multi_out SRCS multi_out.cc)

# Array In/Out test
ion_jit_executable(array_inout SRCS array_inout.cc)

# Array Input test
ion_jit_executable(array_input SRCS array_input.cc)

# Array Output test
ion_jit_executable(array_output SRCS array_output.cc)

# Duplicate array names test
ion_jit_executable(array_dup_names SRCS array_dup_names.cc)

# Inverted dep test
ion_jit_executable(inverted_dep SRCS inverted_dep.cc)

# Duplicate port
ion_jit_executable(dup SRCS dup.cc)

# Port binding
ion_jit_executable(port-binding SRCS port-binding.cc)

# Port index access
if(${OpenCV_DIR})
    ion_jit_executable(port-access SRCS port-access.cc)
endif()

# Asigning port in arbitrary order
ion_jit_executable(port-assign SRCS port-assign.cc)

# zero copy i/o for extern functions
ion_jit_executable(direct-extern SRCS direct-extern.cc)

if (${CUDAToolkit_FOUND})
    ion_jit_executable(gpu-extern SRCS gpu-extern.cc gpu-extern-lib.cu)
    ion_jit_executable(cuda-interop SRCS cuda-interop.cc LIBS CUDA::cuda_driver CUDA::cudart)
endif()

# Duplicate name test
ion_jit_executable(dup-port-name SRCS dup-port-name.cc)

# BuildingBlock::configure
ion_jit_executable(configure SRCS configure.cc)

# Export test
ion_jit_executable(export SRCS export.cc)

# Validation test
ion_jit_executable(validation SRCS validation.cc)

# Multi
ion_jit_executable(multi_pipe SRCS multi_pipe.cc)

# Unbound Binding
ion_jit_executable(unbound_binding SRCS unbound_binding.cc)

# Graph
ion_jit_executable(graph SRCS graph.cc)

ion_aot_executable(simple_graph SRCS_COMPILE simple_graph_compile.cc SRCS_RUN simple_graph_run.cc LIBS ion-bb-test)
ion_jit_executable(simple_graph SRCS simple_graph_jit.cc)

ion_aot_executable(complex_graph SRCS_COMPILE complex_graph_compile.cc SRCS_RUN complex_graph_run.cc LIBS ion-bb-test)
ion_jit_executable(complex_graph SRCS complex_graph_jit.cc)
