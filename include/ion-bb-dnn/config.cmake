# OpenCV
find_package(OpenCV 4 REQUIRED)
if (UNIX)
    add_compile_options(-Wno-format-security)
endif()

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/3rdparty
    ${OpenCV_INCLUDE_DIRS})

set(LINK_DIRS
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
