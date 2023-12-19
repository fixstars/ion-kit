# OpenCV
find_package(OpenCV 4 REQUIRED)
if (UNIX)
    add_compile_options(-Wno-format-security)
endif()

set(INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS})

set(LINK_DIRS
    ${OpenCV_DIR}/lib)

if (UNIX)
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
    set(LIBRARIES ${OpenCV_LIBRARIES})
endif()
