#
# ion-core Testing
#

# Test BB module
add_library(ion-bb-test SHARED ion-bb-test.cc)
if(UNIX)
    target_compile_options(ion-bb-test PUBLIC -fno-rtti)
endif()
target_link_libraries(ion-bb-test ion-core)

# C interface test
ion_jit(c_test SRCS c_test.c)
add_dependencies(c_test ion-bb-test)

# Error case test
ion_jit(error SRCS error.cc)
add_dependencies(error ion-bb-test)

# Metadata test
ion_jit(metadata SRCS metadata.cc)
add_dependencies(metadata ion-bb-test)

# Multi-out test
ion_jit(multi_out SRCS multi_out.cc)
add_dependencies(multi_out ion-bb-test)

# Array In/Out test
ion_jit(array_inout SRCS array_inout.cc)
add_dependencies(array_inout ion-bb-test)

# Array Input test
ion_jit(array_input SRCS array_input.cc)
add_dependencies(array_input ion-bb-test)

# Array Output test
ion_jit(array_output SRCS array_output.cc)
add_dependencies(array_output ion-bb-test)

# Duplicate array names test
ion_jit(array_dup_names SRCS array_dup_names.cc)
add_dependencies(array_dup_names ion-bb-test)

# Inverted dep test
ion_jit(inverted_dep SRCS inverted_dep.cc)
add_dependencies(inverted_dep ion-bb-test)

# Duplicate port
ion_jit(dup SRCS dup.cc)
add_dependencies(dup ion-bb-test)

# Output error handling
ion_jit(port-binding SRCS port-binding.cc)
add_dependencies(port-binding ion-bb-test)

ion_jit(direct-extern SRCS direct-extern.cc)
add_dependencies(direct-extern ion-bb-test)

if (WITH_CUDA)
    ion_jit(gpu-extern SRCS gpu-extern.cc)
    cuda_add_library(gpu-extern-lib SHARED gpu-extern-lib.cu)
    add_dependencies(gpu-extern ion-bb-test gpu-extern-lib)
endif()

# Duplicate name test
ion_jit(dup-port-name SRCS dup-port-name.cc)

# Export test
ion_jit(export SRCS export.cc)

#
# Pipeline testing
#
ion_compile(simple_graph_compile SRCS simple_graph_compile.cc PIPELINE_NAME simple_graph)
ion_run(simple_graph_aot_test simple_graph_compile SRCS simple_graph_run.cc test-rt.h TARGET_STRING "host-profile")
ion_jit(simple_graph_jit SRCS simple_graph_jit.cc test-rt.h)
ion_register_test(simple_graph_jit_test simple_graph_jit TARGET_STRING "host-profile")

ion_compile(complex_graph_compile SRCS complex_graph_compile.cc PIPELINE_NAME complex_graph)
ion_run(complex_graph_aot_test complex_graph_compile SRCS complex_graph_run.cc test-rt.h TARGET_STRING "host-profile")
ion_jit(complex_graph_jit SRCS complex_graph_jit.cc test-rt.h)
ion_register_test(complex_graph_jit_test complex_graph_jit TARGET_STRING "host-profile")
