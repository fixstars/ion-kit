#
# Utilities
#
if (UNIX)
    set(PLATFORM_LIBRARIES pthread dl)
endif()

function(ion_aot_executable NAME_PREFIX)
    set(options)
    set(oneValueArgs TARGET_STRING)
    set(multiValueArgs SRCS_COMPILE SRCS_RUN INCS LIBS RUNTIME_ARGS COMPILER_ARGS)
    cmake_parse_arguments(IAE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT "${IAE_TARGET_STRING}")
        set(IAE_TARGET_STRING "host")
    endif()

    # Build compile
    set(COMPILE_NAME ${NAME_PREFIX}_compile)
    add_executable(${COMPILE_NAME} ${IAE_SRCS_COMPILE})
    if(UNIX)
        target_compile_options(${COMPILE_NAME} PUBLIC -fno-rtti)  # For Halide::Generator
        if(NOT APPLE)
            target_link_options(${COMPILE_NAME} PUBLIC -Wl,--export-dynamic) # For JIT compiling
        endif()
    endif()
    target_include_directories(${COMPILE_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(${COMPILE_NAME} PRIVATE ion-core ${PLATFORM_LIBRARIES})
    set_target_properties(${COMPILE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY compile)
    add_dependencies(${COMPILE_NAME} ion-bb ion-bb-test)

    # Run compile
    set(OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/compile/${COMPILE_NAME}_out)
    set(HEADER ${OUTPUT_PATH}/${NAME_PREFIX}.h)
    set(STATIC_LIB ${OUTPUT_PATH}/${NAME_PREFIX}${CMAKE_STATIC_LIBRARY_SUFFIX})
    add_custom_command(OUTPUT ${OUTPUT_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_PATH}
    )
    if(APPLE)
        add_custom_command(OUTPUT ${HEADER} ${STATIC_LIB}
        COMMAND ${CMAKE_SOURCE_DIR}/script/invoke.sh $<TARGET_FILE:${COMPILE_NAME}>
            HL_TARGET ${IAE_TARGET_STRING}
            DYLD_LIBRARY_PATH ${Halide_DIR}/../../../bin
            DYLD_LIBRARY_PATH ${CMAKE_BINARY_DIR}
            DYLD_LIBRARY_PATH ${CMAKE_BINARY_DIR}/src/bb
        DEPENDS ${COMPILE_NAME} ${OUTPUT_PATH}
        WORKING_DIRECTORY ${OUTPUT_PATH})
    elseif(UNIX)
        add_custom_command(OUTPUT ${HEADER} ${STATIC_LIB}
        COMMAND ${CMAKE_SOURCE_DIR}/script/invoke.sh $<TARGET_FILE:${COMPILE_NAME}>
            HL_TARGET ${IAE_TARGET_STRING}
            LD_LIBRARY_PATH ${Halide_DIR}/../../../bin
            LD_LIBRARY_PATH ${CMAKE_BINARY_DIR}
            LD_LIBRARY_PATH ${CMAKE_BINARY_DIR}/src/bb
        DEPENDS ${COMPILE_NAME} ${OUTPUT_PATH}
        WORKING_DIRECTORY ${OUTPUT_PATH})
    else()
        add_custom_command(OUTPUT ${HEADER} ${STATIC_LIB}
            COMMAND ${CMAKE_SOURCE_DIR}/script/invoke.bat $<TARGET_FILE:${COMPILE_NAME}>
                HL_TARGET ${IAE_TARGET_STRING}
                PATH ${Halide_DIR}/../../../bin/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:Release>
                PATH ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:Release>
                PATH ${CMAKE_BINARY_DIR}/src/bb/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:Release>
            DEPENDS ${COMPILE_NAME} ${OUTPUT_PATH}
            WORKING_DIRECTORY ${OUTPUT_PATH})
    endif()

    # Build run
    set(NAME ${NAME_PREFIX}_aot)
    add_executable(${NAME} ${IAE_SRCS_RUN} ${HEADER})
    target_include_directories(${NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${OUTPUT_PATH} ${IAE_INCS})
    target_link_libraries(${NAME} PRIVATE ${STATIC_LIB} Halide::Halide Halide::Runtime ${PLATFORM_LIBRARIES} ${IAE_LIBS})

    # Test
    if(APPLE)
        set(RUNTIME_ENVS "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/bb:${CMAKE_BINARY_DIR}/test")
    elseif(UNIX)
        set(RUNTIME_ENVS "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/bb:${CMAKE_BINARY_DIR}/test")
    else()
        set(RUNTIME_ENVS "PATH=${CMAKE_BINARY_DIR}/Release\;${CMAKE_BINARY_DIR}/src/bb/Release\;${CMAKE_BINARY_DIR}/test/Release\;${Halide_DIR}/../../../bin/Release")
    endif()

    if (IJE_TARGET_STRING)
        list(APPEND RUNTIME_ENVS "HL_TARGET=${IJE_TARGET_STRING}")
    endif()

    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> ${IAE_RUNTIME_ARGS})
    set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "${RUNTIME_ENVS}")
endfunction()

function(ion_jit_executable NAME_PREFIX)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SRCS INCS LIBS RUNTIME_ARGS)
    cmake_parse_arguments(IJE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(NAME ${NAME_PREFIX}_jit)
    add_executable(${NAME} ${IJE_SRCS})
    if (UNIX AND NOT APPLE)
        target_link_options(${NAME} PUBLIC -Wl,--export-dynamic) # For JIT compiling
    endif()
    add_dependencies(${NAME} ion-bb ion-bb-test)
    target_include_directories(${NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${ION_BB_INCLUDE_DIRS} ${IJE_INCS})
    target_link_libraries(${NAME} PRIVATE ion-core ${ION_BB_LIBRARIES} ${PLATFORM_LIBRARIES} ${IJE_LIBS})
    set_target_properties(${NAME} PROPERTIES ENABLE_EXPORTS ON)

    # Test
    if(APPLE)
        set(RUNTIME_ENVS "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/bb:${CMAKE_BINARY_DIR}/test")
    elseif(UNIX)
        set(RUNTIME_ENVS "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/bb:${CMAKE_BINARY_DIR}/test")
    else()
        set(RUNTIME_ENVS "PATH=${CMAKE_BINARY_DIR}/Release\;${CMAKE_BINARY_DIR}/src/bb/Release\;${CMAKE_BINARY_DIR}/test/Release\;${Halide_DIR}/../../../bin/Release")
    endif()

    if (IJE_TARGET_STRING)
        list(APPEND RUNTIME_ENVS "HL_TARGET=${IJE_TARGET_STRING}")
    endif()

    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> ${IJE_RUNTIME_ARGS})
    set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "${RUNTIME_ENVS}")
endfunction()

function(ion_register_test TEST_NAME EXEC_NAME)
    set(options)
    set(oneValueArgs TARGET_STRING)
    set(multiValueArgs RUNTIME_ARGS)
    cmake_parse_arguments(IERT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(APPLE)
        set(IERT_RUNTIME_ENVS "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/bb:${CMAKE_BINARY_DIR}/test")
    elseif(UNIX)
        set(IERT_RUNTIME_ENVS "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/bb:${CMAKE_BINARY_DIR}/test")
    else()
        set(IERT_RUNTIME_ENVS "PATH=${CMAKE_BINARY_DIR}/Release\;${CMAKE_BINARY_DIR}/src/bb/Release\;${CMAKE_BINARY_DIR}/test/Release\;${Halide_DIR}/../../../bin/Release")
    endif()

    if (IERT_TARGET_STRING)
        list(APPEND IERT_RUNTIME_ENVS "HL_TARGET=${IERT_TARGET_STRING}")
    endif()

    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${EXEC_NAME}> ${IERT_RUNTIME_ARGS})
    set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "${IERT_RUNTIME_ENVS}")
endfunction()
