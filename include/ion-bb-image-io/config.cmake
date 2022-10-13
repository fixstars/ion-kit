if (UNIX)
find_package(OpenCV REQUIRED)

set(INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS})

set(LINK_DIRS
    ${OpenCV_DIR}/lib)

set(LIBRARIES
    rt
    dl
    pthread
    m
    z
    ${OpenCV_LIBRARIES})
else()
    # NOTE: Tentatively OpenCV is not supported in Windows release
endif()
