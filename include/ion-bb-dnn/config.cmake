set(HALIDE_ROOT $ENV{HALIDE_ROOT} CACHE PATH "Path to Halide")
if(HALIDE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set appropriate path to Halide")
endif()

# OpenCV
find_package(OpenCV 3 REQUIRED)
if (UNIX)
    add_compile_options(-Wno-format-security)
endif()

# onnxruntime
set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT} CACHE PATH "Path to onnxruntime")
if(ONNXRUNTIME_ROOT STREQUAL "")
    message(FATAL_ERROR "Set appropriate path to onnxruntime")
endif()


set(INCLUDE_DIRS
    ${HALIDE_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include)

set(LINK_DIRS
    ${HALIDE_ROOT}/bin
    ${OpenCV_DIR}/lib
    ${ONNXRUNTIME_ROOT}/lib)

set(RUNTIME_ENVS
    LD_LIBRARY_PATH ${HALIDE_ROOT}/bin
    LD_LIBRARY_PATH ${OpenCV_DIR}/lib
    LD_LIBRARY_PATH ${ONNXRUNTIME_ROOT}/lib)

if (UNIX)
    set(LIBRARIES
        rt
        dl
        pthread
        m
        z
        ${OpenCV_LIBS})
else()
    set(LIBRARIES
        ${OpenCV_LIBS})
endif()
