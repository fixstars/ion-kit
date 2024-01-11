cmake_minimum_required(VERSION 3.12)
project(ion-kit LANGUAGES C CXX)

#
# Options
#
option(ION_BUILD_DOC "Build documents." OFF)
option(ION_BUILD_TEST "Enable to build tests" ON)
option(ION_BUILD_EXAMPLE "Enable to build examples" ON)
option(ION_BUNDLE_HALIDE "Enable to bundle Halide binary into package" OFF)
option(ION_ENABLE_HALIDE_FPGA_BACKEND "Enable to Halide FPGA backend" OFF)

#
# CMake common settings
#
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # This is required to export symbols on windows platform
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


#
# Dependent libraries
#
# NOTE: WIP: It's for static linking
# set(ZLIB_USE_STATIC_LIBS "ON")
# find_package(ZLIB REQUIRED)
# find_package(LLVM REQUIRED)
# find_package(Halide REQUIRED COMPONENTS static)
find_package(Halide REQUIRED COMPONENTS shared)
find_package(CUDA)

#
# Version
#
execute_process(COMMAND git describe --tags
    OUTPUT_VARIABLE GIT_DESCRIBE_RESULT
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
string(STRIP "${GIT_DESCRIBE_RESULT}" ION_KIT_VERSION_S)
message(STATUS "Version: ${ION_KIT_VERSION_S}")
add_definitions("-DION_KIT_VERSION=\"${ION_KIT_VERSION_S}\"")
string(REPLACE "v" "" ION_KIT_VERSION "${ION_KIT_VERSION_S}")
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/VERSION CONTENT "${ION_KIT_VERSION_S}")

#
# ion utilities
#
include(IonUtil)

#
# Enable FPGA feature
#
if (ION_ENABLE_HALIDE_FPGA_BACKEND)
    add_compile_definitions(HALIDE_FOR_FPGA)
endif()

#
# Build main library
#
file(GLOB ION_CORE_SRC LIST_DIRECTORIES false ${PROJECT_SOURCE_DIR}/src/*)
list(REMOVE_ITEM ION_CORE_SRC "${PROJECT_SOURCE_DIR}/src/generator.cc")
add_library(ion-core SHARED ${ION_CORE_SRC})
target_include_directories(ion-core PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
if (UNIX)
  target_link_libraries(ion-core PUBLIC Halide::Halide Halide::Runtime dl pthread z m stdc++)
else()
  target_link_libraries(ion-core PUBLIC Halide::Halide Halide::Runtime)
endif()

#
# ion-bb
#
add_subdirectory(src/bb)

#
# Test
#
if (ION_BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

#
# Examples
#
if(ION_BUILD_EXAMPLE)
    enable_testing()
    add_subdirectory(example)
endif()

#
# Documentation
#
if (ION_BUILD_DOC)
    find_package(Doxygen)
    find_package(Sphinx)
    if (Doxygen_FOUND AND Sphinx_FOUND)
        file(GLOB_RECURSE FILES ${CMAKE_SOURCE_DIR}/include/*.h)
        list(APPEND PUBLIC_HEADERS ${FILES})
        set(DOXYGEN_INDEX_FILE ${CMAKE_BINARY_DIR}/xml/index.xml)
        add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/doc/Doxyfile
            DEPENDS ${PUBLIC_HEADERS}
            MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/doc/Doxyfile
            COMMENT "Generating docs")
        add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

        set(SPHINX_INDEX_FILE ${CMAKE_BINARY_DIR}/doc/index.html)
        add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
            COMMAND ${SPHINX_EXECUTABLE} -b html
            -Dbreathe_projects.ion-core=${CMAKE_BINARY_DIR}/xml
            ${CMAKE_SOURCE_DIR}/doc ${CMAKE_BINARY_DIR}/doc
            DEPENDS ${DOXYGEN_INDEX_FILE}
            MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/doc/index.rst)
        add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

        install(DIRECTORY ${CMAKE_BINARY_DIR}/doc DESTINATION .)
    endif()
endif()

#
# Install
#
if (UNIX)
    install(TARGETS ion-core DESTINATION lib)
else()
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-core.dll DESTINATION bin)
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-core.lib DESTINATION lib)

    # These lines is for backward compatibility.
    # TODO: Deprecate old name of dll/lib.
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-core.dll DESTINATION bin RENAME ion.dll)
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-core.lib DESTINATION lib RENAME ion.lib)
endif()

install(DIRECTORY include DESTINATION .)
install(FILES
    ${CMAKE_BINARY_DIR}/VERSION
    DESTINATION .)
install(FILES
    ${CMAKE_SOURCE_DIR}/LICENSE
    DESTINATION license)
install(FILES
    ${CMAKE_SOURCE_DIR}/thirdparty_notice.txt
    DESTINATION license)

# Halide
if (ION_BUNDLE_HALIDE)
    set(HALIDE_ROOT ${Halide_DIR}/../../..)

    file(GLOB HALIDE_HEADERS ${HALIDE_ROOT}/include/Halide*.h ${HALIDE_ROOT}/include/wasm*.h)
    install(FILES ${HALIDE_HEADERS} DESTINATION include)

    file(GLOB HALIDE_BINS
        ${HALIDE_ROOT}/bin/featurization_to_sample*
        ${HALIDE_ROOT}/bin/get_host_target*
        ${HALIDE_ROOT}/bin/retrain_cost_model*
        ${HALIDE_ROOT}/bin/weightsdir_to_weightsfile*)
    install(FILES ${HALIDE_BINS} DESTINATION bin)

    install(DIRECTORY ${HALIDE_ROOT}/lib/cmake/Halide DESTINATION lib/cmake/)
    if (EXISTS ${HALIDE_ROOT}/lib/cmake/HalideHelpers)
        install(DIRECTORY ${HALIDE_ROOT}/lib/cmake/HalideHelpers DESTINATION lib/cmake/)
    endif()
    if (UNIX)
        file(GLOB HALIDE_LIBRARIES ${HALIDE_ROOT}/lib/libHalide*)
        install(FILES ${HALIDE_LIBRARIES} DESTINATION lib)
        file(GLOB HALIDE_LIBRARIES ${HALIDE_ROOT}/lib/libautoschedule*)
        install(FILES ${HALIDE_LIBRARIES} DESTINATION lib)
    else()
        install(FILES ${HALIDE_ROOT}/bin/Release/Halide.dll DESTINATION bin)
        install(FILES ${HALIDE_ROOT}/lib/Release/Halide.lib DESTINATION lib)
    endif()
endif()

#
# Packaging
#
if (UNIX)
    set(CPACK_GENERATOR "TGZ")
else ()
    set(CPACK_GENERATOR "ZIP")
endif ()
set(CPACK_PACKAGE_VERSION ${ION_KIT_VERSION})
include(CPack)
