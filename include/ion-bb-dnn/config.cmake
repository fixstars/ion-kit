set(HALIDE_ROOT $ENV{HALIDE_ROOT} CACHE PATH "Path to Halide")
if(HALIDE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set HALIDE_ROOT")
endif()

# OpenCV
find_package(OpenCV 4 REQUIRED)
if (UNIX)
    add_compile_options(-Wno-format-security)
endif()

set(INCLUDE_DIRS
    ${HALIDE_ROOT}/include
    ${OpenCV_INCLUDE_DIRS})

set(LINK_DIRS
    ${HALIDE_ROOT}/bin
    ${OpenCV_DIR}/lib)

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
