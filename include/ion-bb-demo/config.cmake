set(HALIDE_ROOT $ENV{HALIDE_ROOT} CACHE PATH "Path to Halide")
if(HALIDE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set appropriate path to Halide")
endif()

find_package(OpenCV 3 REQUIRED)

set(INCLUDE_DIRS
    ${HALIDE_ROOT}/include
    ${OpenCV_INCLUDE_DIRS})

set(LINK_DIRS
    ${HALIDE_ROOT}/bin
    ${OpenCV_DIR}/lib)

set(RUNTIME_ENVS
    LD_LIBRARY_PATH ${HALIDE_ROOT}/bin
    LD_LIBRARY_PATH ${OpenCV_DIR}/lib)

set(LIBRARIES
    rt
    dl
    pthread
    m
    z
    ${OpenCV_LIBS})
