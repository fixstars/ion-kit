# OpenCV
find_package(OpenCV 4 REQUIRED)
if (UNIX)
    add_compile_options(-Wno-format-security)
endif()

#
# tvm
#
if(NOT WITH_TVM STREQUAL "OFF")
  message(STATUS "will use libtvm_runtime.so")

  # confirm CMake can find the libtvm_runtime.so & set rpath if required
  if (WITH_TVM STREQUAL "ON")
      find_library(TVM_RUNTIME_LIB libtvm_runtime.so REQUIRED)
  else()
      find_library(TVM_RUNTIME_LIB NAMES libtvm_runtime.so PATHS ${WITH_TVM} REQUIRED)
      get_filename_component(RPATH_DIR ${TVM_RUNTIME_LIB} DIRECTORY)
      set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath ${RPATH_DIR}")
  endif()
endif()

set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include/ion-bb-dnn/3rdparty
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
