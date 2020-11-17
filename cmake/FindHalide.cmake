if(HALIDE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set appropriate path to Halide")
endif()

# Find paths
find_path(HALIDE_INCLUDE_DIR NAMES Halide.h PATHS ${HALIDE_ROOT}/include)
if (WIN32)
  find_library(HALIDE_LIBRARY_DEBUG NAMES Halide PATHS ${HALIDE_ROOT}/Debug)
  find_library(HALIDE_LIBRARY_RELEASE NAMES Halide PATHS ${HALIDE_ROOT}/Release)
else()
  find_library(HALIDE_LIBRARY NAMES Halide PATHS ${HALIDE_ROOT}/bin ${HALIDE_ROOT}/lib)
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
    IMPORTED_LOCATION_DEBUG "${HALIDE_LIBRARY_DEBUG}"
    IMPORTED_LOCATION_RELEASE "${HALIDE_LIBRARY_RELEASE}"
    INTERFACE_INCLUDE_DIRECTORIES "${HALIDE_INCLUDE_DIR}"
  )
endif()
