find_package(JPEG QUIET)
find_package(PNG QUIET)

if(JPEG_FOUND AND PNG_FOUND)
    set(ION_BB_BUILD_image-io TRUE)

    set(INCLUDE_DIRS ${JPEG_INCLUDE_DIR ${PNG_INCLUDE_DIR} }

    find_package(OpenCV 4 QUIET)
    if (${OpenCV_FOUND})
        add_compile_definitions(HAS_OPENCV)

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
    endif()
else()
    set(ION_BB_BUILD_image-io FALSE)
endif()
