if (UNIX)
find_package(OpenCV REQUIRED)

set(INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS})

set(LINK_DIRS
    ${OpenCV_DIR}/lib)

if (APPLE)
    set(LIBRARIES
        dl
        pthread
        m
        z
        ${OpenCV_LIBRARIES})
else()
    set(LIBRARIES
        rt
        dl
        pthread
        m
        z
        ${OpenCV_LIBRARIES})
endif()
else()
    # NOTE: Tentatively OpenCV is not supported in Windows release
endif()
