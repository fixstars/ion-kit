if (UNIX)
if (APPLE)
    set(LIBRARIES
        dl
        pthread)
else()
    set(LIBRARIES
        rt
        dl
        pthread)
endif()
endif()
