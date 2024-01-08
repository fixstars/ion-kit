if (${ION_ENABLE_HALIDE_FPGA_BACKEND})
    set(ION_BB_BUILD_fpga TRUE)

    if (APPLE)
        set(LIBRARIES
            dl
            pthread)
    elseif (UNIX)
        set(LIBRARIES
            rt
            dl
            pthread)
    endif()
else()
    set(ION_BB_BUILD_fpga FALSE)
endif()
