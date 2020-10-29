project(ion-kit LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#
# Options
#
set(HALIDE_ROOT "" CACHE PATH "Path to Halide")
option(ION_BUILD_ALL_BB "Enable to build all building blocks" ON)
set(ION_BBS_TO_BUILD "" CACHE STRING "A List of building blocks to build (This overrides ION_BUILD_ALL_BB)")
option(ION_BUILD_DOC "Build documents." OFF)
option(ION_BUILD_TEST "Enable to build tests" ON)
option(ION_BUILD_EXAMPLE "Enable to build examples" ON)
option(ION_BUNDLE_HALIDE "Enable to bundle halide binary into package" OFF)
option(WITH_CUDA "Enable CUDA with buliding examples." ON)

#
# Dependent libraries
#
find_package(Halide REQUIRED)
if (WITH_CUDA)
    find_package(CUDA REQUIRED)
endif()

#
# Version
#
execute_process(COMMAND git rev-parse --short HEAD OUTPUT_VARIABLE GIT_REV ERROR_QUIET)
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/VERSION CONTENT "${GIT_REV}")

#
# ion utilities
#
include(IonUtil)

#
# Build main library
#
file(GLOB ION_SRC src/*)
add_library(ion-core SHARED ${ION_SRC})
target_include_directories(ion-core PUBLIC "${PROJECT_SOURCE_DIR}/include;${HALIDE_INCLUDE_DIR}")
target_link_libraries(ion-core ${HALIDE_LIBRARY} dl pthread z)

#
# ion-bb
#
include(ImportIonBB)
ion_import_building_block()

#
# Link directories
#
link_directories(${ION_BB_LINK_DIRS})

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
install(FILES cmake/ion-bb-config.cmake DESTINATION share/cmake/)

if (UNIX)
    install(TARGETS ion-core ion-bb-test DESTINATION lib)
else()
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-core.dll DESTINATION bin)
    install(FILES ${CMAKE_BINARY_DIR}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/ion-core.lib DESTINATION lib)
endif()

install(DIRECTORY include DESTINATION .)
install(FILES
    ${CMAKE_BINARY_DIR}/VERSION
    DESTINATION .)
install(FILES
    ${CMAKE_SOURCE_DIR}/LICENSE
    DESTINATION license)

# Halide
if (ION_BUNDLE_HALIDE)
    install(DIRECTORY ${HALIDE_INCLUDE_DIR} DESTINATION .)
    #TODO: Handle for Windows
    if (UNIX)
        install(FILES ${HALIDE_LIBRARY} DESTINATION lib)
    else()
        install(FILES ${HALIDE_ROOT}/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/Halide.dll DESTINATION bin)
        install(FILES ${HALIDE_ROOT}/lib/Halide.lib DESTINATION lib)
    endif()
endif()

#
# Packaging
#
if (UNIX)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Fixstars Solutions Inc.")
    execute_process(COMMAND git describe --tags
        OUTPUT_VARIABLE GIT_DESCRIBE_RESULT
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    string(STRIP ${GIT_DESCRIBE_RESULT} ION_CORE_VERSION_S)
    string(REPLACE "v" "" ION_CORE_VERSION ${ION_CORE_VERSION_S})
    set(CPACK_PACKAGE_VERSION ${ION_CORE_VERSION})
    include(CPack)
endif()
