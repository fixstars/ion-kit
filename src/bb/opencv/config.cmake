find_package(OpenCV 4 QUIET)

if (${OpenCV_FOUND})
    set(ION_BB_BUILD_opencv TRUE)

    if (UNIX)
        add_compile_options(-Wno-format-security)
    endif()

    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    set(LINK_DIRS ${OpenCV_DIR}/lib)
    set(LIBRARIES ${OpenCV_LIBRARIES})

    if (APPLE)
        list(APPEND RUNTIME_ENVS DYLD_LIBRARY_PATH ${OpenCV_DIR}/lib)
    elseif (UNIX)
        list(APPEND RUNTIME_ENVS LD_LIBRARY_PATH ${OpenCV_DIR}/lib)
    else()
        list(APPEND RUNTIME_ENVS PATH ${OpenCV_DIR}/x64/vc15/bin)
    endif()
else()
    set(ION_BB_BUILD_opencv FALSE)
endif()
