find_package(OpenCV 4 QUIET)

if (${OpenCV_FOUND})
    set(ION_BB_BUILD_opencv TRUE)

    if (UNIX)
        add_compile_options(-Wno-format-security)
    endif()

    set(INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS})

    set(LINK_DIRS
        ${OpenCV_DIR}/lib)

    if (APPLE)
        set(LIBRARIES
            dl
            pthread
            m
            ${OpenCV_LIBRARIES})
        list(APPEND RUNTIME_ENVS LD_LIBRARY_PATH ${OpenCV_DIR}/lib)
    elseif (UNIX)
        set(LIBRARIES
            rt
            dl
            pthread
            m
            ${OpenCV_LIBRARIES})
        list(APPEND RUNTIME_ENVS LD_LIBRARY_PATH ${OpenCV_DIR}/lib)
    else()
        set(LIBRARIES
            ${OpenCV_LIBRARIES})
        list(APPEND RUNTIME_ENVS PATH ${OpenCV_DIR}/x64/vc15/bin)
    endif()
else()
    set(ION_BB_BUILD_opencv FALSE)
endif()
