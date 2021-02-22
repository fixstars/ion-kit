macro(ion_import_building_block)
    cmake_policy(SET CMP0057 NEW)

    if(NOT ION_BBS_TO_BUILD STREQUAL "")
        set(ION_BUILD_ALL_BB OFF)
    endif()

    set(ION_BBS)
    set(ION_BB_INCLUDE_DIRS)
    set(ION_BB_LINK_DIRS)
    set(ION_BB_LIBRARIES)

    file(GLOB childs ${CMAKE_CURRENT_SOURCE_DIR}/include/ion-bb-*)
    foreach(child ${childs})
        if(IS_DIRECTORY ${child})
            get_filename_component(BB_NAME ${child} NAME)
            set(IS_TARGET_BB FALSE)
            if(${BB_NAME} IN_LIST ION_BBS_TO_BUILD)
                set(IS_TARGET_BB TRUE)
            endif()

            if(${ION_BUILD_ALL_BB} OR ${IS_TARGET_BB})
                set(INCLUDE_DIRS)
                set(LINK_DIRS)
                set(LIBRARIES)
                include(${child}/config.cmake)
                list(APPEND ION_BBS ${BB_NAME})
                list(APPEND ION_BB_INCLUDE_DIRS ${INCLUDE_DIRS})
                list(APPEND ION_BB_LINK_DIRS    ${LINK_DIRS})
                list(APPEND ION_BB_LIBRARIES    ${LIBRARIES})

                install(DIRECTORY ${child}
                    DESTINATION ./include
                    PATTERN ".git" EXCLUDE)
            endif()
        endif()
    endforeach()

    message(STATUS "Found BBs: ${ION_BBS}")
    message(STATUS "  Include directories: ${ION_BB_INCLUDE_DIRS}")
    message(STATUS "  Link directories: ${ION_BB_LINK_DIRS}")
    message(STATUS "  Dependent Libaries: ${ION_BB_LIBRARIES}")
endmacro()
