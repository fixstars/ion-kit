cmake_policy(SET CMP0057 NEW)

set(ION_BB_INCLUDE_DIRS)
set(ION_BB_LINK_DIRS)
set(ION_BB_LIBRARIES)
file(GLOB childs ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(child ${childs})
    if(IS_DIRECTORY ${child})
        get_filename_component(BB_NAME ${child} NAME)
        set(IS_TARGET_BB FALSE)
        if(${BB_NAME} IN_LIST ION_TARGET_BB)
            set(IS_TARGET_BB TRUE)
        endif()

        if(${IS_TARGET_BB})
            set(INCLUDE_DIRS)
            set(LINK_DIRS)
            set(LIBRARIES)
            if(EXISTS ${child}/config.cmake)
                include(${child}/config.cmake)
            endif()
            list(APPEND ION_BB_INCLUDE_DIRS ${INCLUDE_DIRS})
            list(APPEND ION_BB_LINK_DIRS    ${LINK_DIRS})
            list(APPEND ION_BB_LIBRARIES    ${LIBRARIES})
            string(TOUPPER ${BB_NAME} POSTFIX)
            string(REPLACE "-" "_" POSTFIX ${POSTFIX})
            add_definitions("-DION_ENABLE_BB_${POSTFIX}")
        endif()
    endif()
endforeach()

add_library(ion-bb SHARED bb.cc)
target_include_directories(ion-bb PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR} ${ION_BB_INCLUDE_DIRS})
target_link_libraries(ion-bb PUBLIC ion-core ${ION_BB_LIBRARIES})
if(UNIX)
    target_compile_options(ion-bb
        PUBLIC -fno-rtti  # For Halide::Generator
        PUBLIC -rdynamic) # For JIT compiling
endif()


#
# Install
#
if (UNIX)
    install(TARGETS ion-bb DESTINATION lib)
else()
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-bb.dll DESTINATION bin)
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-bb.lib DESTINATION lib)
endif()
