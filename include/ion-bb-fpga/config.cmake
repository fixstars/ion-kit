set(HALIDE_ROOT $ENV{HALIDE_ROOT} CACHE PATH "Path to Halide")
if(HALIDE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set HALIDE_ROOT")
endif()

set(INCLUDE_DIRS
    ${HALIDE_ROOT}/include)

set(LINK_DIRS
    ${HALIDE_ROOT}/bin)

if (UNIX)
    set(LIBRARIES
        rt
        dl
        pthread)
endif()
