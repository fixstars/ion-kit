#
# Utilities
#
if (UNIX)
    set(PLATFORM_LIBRARIES pthread dl)
endif()

function(ion_compile NAME)
    set(options)
    set(oneValueArgs PIPELINE_NAME)
    set(multiValueArgs SRCS)
    cmake_parse_arguments(IEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IEC_PIPELINE_NAME)
        set(IEC_PIPELINE_NAME ${NAME})
    endif()

    # Build compile
    add_executable(${NAME} ${IEC_SRCS})
    if(UNIX)
        target_compile_options(${NAME}
            PUBLIC -fno-rtti  # For Halide::Generator
            PUBLIC -rdynamic) # For JIT compiling
    endif()
    target_include_directories(${NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include;${ION_BB_INCLUDE_DIRS};${HALIDE_INCLUDE_DIR}")
    target_link_libraries(${NAME} ion-core Halide::Halide ${ION_BB_LIBRARIES} ${PLATFORM_LIBRARIES})
    set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY compile PIPELINE_NAME ${IEC_PIPELINE_NAME})
endfunction()

function(ion_run NAME COMPILE_NAME)
    set(options)
    set(oneValueArgs TARGET_STRING)
    set(multiValueArgs SRCS RUNTIME_ENVS COMPILE_ARGS RUNTIME_ARGS)
    cmake_parse_arguments(IER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IER_TARGET_STRING)
        set(IER_TARGET_STRING host)
    endif()

    get_target_property(PIPELINE_NAME ${COMPILE_NAME} PIPELINE_NAME)

    # Run compile
    set(OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/compile/${IER_TARGET_STRING})
    set(HEADER ${OUTPUT_PATH}/${PIPELINE_NAME}.h)
    set(STATIC_LIB ${OUTPUT_PATH}/${PIPELINE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    add_custom_command(OUTPUT ${OUTPUT_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_PATH}
    )
    if (UNIX)
        add_custom_command(OUTPUT ${HEADER} ${STATIC_LIB}
            COMMAND ${CMAKE_SOURCE_DIR}/script/invoke.sh $<TARGET_FILE:${COMPILE_NAME}>
                HL_TARGET ${IER_TARGET_STRING}
                LD_LIBRARY_PATH ${HALIDE_ROOT}/bin
                LD_LIBRARY_PATH ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>
            DEPENDS ${COMPILE_NAME} ${OUTPUT_PATH}
            WORKING_DIRECTORY ${OUTPUT_PATH})
    else()
        add_custom_command(OUTPUT ${HEADER} ${STATIC_LIB}
            COMMAND ${CMAKE_SOURCE_DIR}/script/invoke.bat $<TARGET_FILE:${COMPILE_NAME}>
                HL_TARGET ${IER_TARGET_STRING}
                PATH ${HALIDE_ROOT}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>
                PATH ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>
            DEPENDS ${COMPILE_NAME} ${OUTPUT_PATH}
            WORKING_DIRECTORY ${OUTPUT_PATH})
    endif()

    # Build run
    add_executable(${NAME} ${IER_SRCS} ${HEADER})
    target_include_directories(${NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include;${ION_BB_INCLUDE_DIRS};${HALIDE_INCLUDE_DIR};${OUTPUT_PATH}")
    target_link_libraries(${NAME} ${STATIC_LIB} ${ION_BB_LIBRARIES} Halide::Halide ${PLATFORM_LIBRARIES})

    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> ${IER_RUNTIME_ARGS})
    set_tests_properties(${NAME} PROPERTIES ENVIRONMENT "${IER_RUNTIME_ENVS}")
endfunction()

# For multiple target, don't use this function.
function(ion_compile_and_run NAME)
    set(options)
    set(oneValueArgs TARGET_STRING)
    set(multiValueArgs COMPILE_SRCS RUN_SRCS RUNTIME_ENVS RUNTIME_ARGS)
    cmake_parse_arguments(IECR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    ion_compile(${NAME}_compile SRCS ${IECR_COMPILE_SRCS} PIPELINE_NAME ${NAME})
    ion_run(${NAME} ${NAME}_compile SRCS ${IECR_RUN_SRCS} RUNTIME_ENVS ${IECR_RUNTIME_ENVS} RUNTIME_ARGS ${IECR_RUNTIME_ARGS} TARGET_STRING ${IECR_TARGET_STRING})
endfunction()

function(ion_jit NAME)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SRCS)
    cmake_parse_arguments(IEJ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${NAME} ${IEJ_SRCS})
    if (UNIX)
        # For JIT compiling
        target_compile_options(${NAME} PUBLIC -rdynamic)
    endif()
    target_include_directories(${NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include;${ION_BB_INCLUDE_DIRS};${HALIDE_INCLUDE_DIR}")
    target_link_libraries(${NAME} ion-core Halide::Halide ${ION_BB_LIBRARIES} ${PLATFORM_LIBRARIES})
    set_target_properties(${NAME} PROPERTIES ENABLE_EXPORTS ON)
endfunction()

function(ion_register_test TEST_NAME EXEC_NAME)
    set(options)
    set(oneValueArgs TARGET_STRING)
    set(multiValueArgs RUNTIME_ENVS RUNTIME_ARGS)
    cmake_parse_arguments(IERT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (IERT_TARGET_STRING)
        list(APPEND IERT_RUNTIME_ENVS "HL_TARGET=${IERT_TARGET_STRING}")
    endif()

    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${EXEC_NAME}> ${IERT_RUNTIME_ARGS})
    set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "${IERT_RUNTIME_ENVS}")
endfunction()
