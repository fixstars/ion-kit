cmake_minimum_required(VERSION 3.25)
project(ion-kit LANGUAGES C CXX)

#
# Options
#
option(ION_BUILD_DOC "Build documents." OFF)
option(ION_BUILD_TEST "Enable to build tests" ON)
option(ION_BUILD_EXAMPLE "Enable to build examples" ON)
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

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()


# Dependent libraries
#
find_package(Threads REQUIRED)
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
add_library(ion-core SHARED ${ION_CORE_SRC})
target_include_directories(ion-core PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ion-core PUBLIC Halide::Halide Threads::Threads ${CMAKE_DL_LIBS} )

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
file(GLOB_RECURSE HALIDE_HEADERS "${Halide_DIR}/../../../Halide*.h")
install(FILES ${HALIDE_HEADERS} DESTINATION include)
install(IMPORTED_RUNTIME_ARTIFACTS Halide::Halide)

#
# Packaging
#
if(UNIX)
    set(CPACK_GENERATOR "TGZ")
else()
    set(CPACK_GENERATOR "ZIP")
endif ()
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${ION_KIT_VERSION})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-unknown)
if(APPLE)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-x86-64-macos)
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-arm-64-macos)
    endif()
elseif(UNIX)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-x86-64-linux)
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-arm-64-linux)
    endif()
elseif(WIN32)
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-x86-64-windows)
    endif()
endif()

include(CPack)
