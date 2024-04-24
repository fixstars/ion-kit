# Build and install llama.cpp with following command
# cmake -D CMAKE_INSTALL_PREFIX=<prefix> -D BUILD_SHARED_LIBS=on -D LLAMA_STATIC=off ..
find_package(Llama QUIET)
if (${Llama_FOUND})
    set(ION_BB_BUILD_llm TRUE)
    set(INCLUDE_DIRS ${LLAMA_INCLUDE_DIR})
    set(LINK_DIRS ${LLAMA_LIB_DIR})
    set(LIBRARIES llama llava_shared)
    set(EXTRA_SRCS
        bb.cc
        common.cpp
        sampling.cpp
        json-schema-to-grammar.cpp
        grammar-parser.cpp)
else()
    set(ION_BB_BUILD_llm FALSE)
endif()
