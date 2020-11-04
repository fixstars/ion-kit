set(HALIDE_ROOT $ENV{HALIDE_ROOT} CACHE PATH "Path to Halide")
if(HALIDE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set appropriate path to Halide")
endif()

# find_package(OpenCV 3 REQUIRED)
# add_compile_options(-Wno-format-security)

# set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT} CACHE PATH "Path to onnxruntime")
# if(ONNXRUNTIME_ROOT STREQUAL "")
#     message(FATAL_ERROR "Set appropriate path to onnxruntime")
# endif()

set(INCLUDE_DIRS
    ${HALIDE_ROOT}/include
    #${ONNXRUNTIME_ROOT}/include
    #${OpenCV_INCLUDE_DIRS}
    )

set(LINK_DIRS
    ${HALIDE_ROOT}/bin
    #${ONNXRUNTIME_ROOT}/lib
    #${OpenCV_DIR}/lib
    )

set(RUNTIME_ENVS
    LD_LIBRARY_PATH ${HALIDE_ROOT}/bin
    #LD_LIBRARY_PATH ${OpenCV_DIR}/lib
    #LD_LIBRARY_PATH ${ONNXRUNTIME_ROOT}/lib
    )

set(LIBRARIES
    rt
    dl
    pthread
    m
    z
    uuid
    #${OpenCV_LIBS}
    )
