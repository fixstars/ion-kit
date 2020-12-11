if(HALIDE_ROOT STREQUAL "")
    message(STATUS ${HALIDE_ROOT})
    message(FATAL_ERROR "Set appropriate path to Halide")
endif()

# Find paths
find_path(HALIDE_INCLUDE_DIR NAMES Halide.h PATHS ${HALIDE_ROOT}/include)
if(UNIX)
  find_library(HALIDE_LIBRARY NAMES Halide PATHS ${HALIDE_ROOT}/bin ${HALIDE_ROOT}/lib)
elseif(WIN32)
  find_library(HALIDE_LIBRARY NAMES Halide.dll PATHS ${HALIDE_ROOT}/Release)
  find_library(HALIDE_LIBRARY_DEBUG NAMES Halide.dll PATHS ${HALIDE_ROOT}/Debug)
else()
  message(FATAL_ERROR "Unsupported platform")
endif()
find_path(HALIDE_TOOLS_DIR NAMES tools PATHS ${HALIDE_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Halide
  DEFAULT_MSG
  HALIDE_INCLUDE_DIR
  HALIDE_LIBRARY
  HALIDE_TOOLS_DIR
)

if(HALIDE_FOUND AND NOT TARGET Halide::Halide)
  add_library(Halide::Halide SHARED IMPORTED)
  set_target_properties(Halide::Halide PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "C;CXX"
    IMPORTED_LOCATION "${HALIDE_LIBRARY}"
    IMPORTED_LOCATION_DEBUG "${HALIDE_LIBRARY_DEBUG}"
    IMPORTED_IMPLIB "${HALIDE_LIBRARY}"
    IMPORTED_IMPLIB_DEBUG "${HALIDE_LIBRARY_DEBUG}"
    INTERFACE_INCLUDE_DIRECTORIES "${HALIDE_INCLUDE_DIR}"
  )
endif()
