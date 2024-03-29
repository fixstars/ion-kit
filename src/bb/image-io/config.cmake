find_package(JPEG)
find_package(PNG)
find_package(ZLIB)

if(JPEG_FOUND AND PNG_FOUND AND ZLIB_FOUND)
    set(ION_BB_BUILD_image-io TRUE)

    set(INCLUDE_DIRS ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR})
    set(LIBRARIES ${JPEG_LIBRARY} ${PNG_LIBRARY} ${ZLIB_LIBRARY})

    find_package(OpenCV 4 QUIET)
    if (OpenCV_FOUND)
        add_compile_definitions(HAS_OPENCV)

        if (UNIX)
            add_compile_options(-Wno-format-security)
        endif()

        list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
        list(APPEND LINK_DIRS ${OpenCV_DIR}/lib)
        list(APPEND LIBRARIES ${OpenCV_LIBRARIES})

        if (APPLE)
            list(APPEND RUNTIME_ENVS DYLD_LIBRARY_PATH ${OpenCV_DIR}/lib)
        elseif (UNIX)
            list(APPEND RUNTIME_ENVS LD_LIBRARY_PATH ${OpenCV_DIR}/lib)
        else()
            list(APPEND RUNTIME_ENVS PATH ${OpenCV_DIR}/x64/vc15/bin)
        endif()
    endif()
else()
    set(ION_BB_BUILD_image-io FALSE)
endif()
