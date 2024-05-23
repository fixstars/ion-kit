# 1. Build and install llama.cpp (tag:b2972) with following command
#
# cmake -D CMAKE_INSTALL_PREFIX=<path-to-llama.cpp-install> -D CMAKE_POSITION_INDEPENDENT_CODE=on -D LLAMA_CUDA=on ..

# 2. (Optional) Maintain sources originated from llama.cpp to use different version of llama.cpp
#
# export LLAMA_DIR=<path-to-llama.cpp>
# cp ${LLAMA_DIR}/common/{base64,json}.hpp .
# cp ${LLAMA_DIR}/common/build-info.cpp .
# cp ${LLAMA_DIR}/common/stb_image.h .
# cp ${LLAMA_DIR}/common/{common,sampling,grammar-parser,json-schema-to-grammar}.{h,cpp} .
# cp ${LLAMA_DIR}/examples/llava/{llava,clip}.{h,cpp} .
# sed -e "s/#pragma once/#pragma once\n#define LOG(...)/g" --in-place sampling.h
# sed -e "s/#pragma once/#pragma once\n#define LOG_TEE(...)/g" --in-place sampling.h
# sed -e "s/#pragma once/#pragma once\n#define LOG_DISABLE_LOGS/g" --in-place common.h
# sed -e "s/#define CLIP_H/#define CLIP_H\n#define LOG_TEE(...)/g" --in-place clip.h

# 3. Build ion-kit with Llama_DIR path to enable llm BB
#
# cmake -D Llama_DIR=<path-to-llama.cpp-install> ..

find_package(Llama QUIET)
if (${Llama_FOUND})
    set(ION_BB_BUILD_llm TRUE)
    set(INCLUDE_DIRS ${LLAMA_INCLUDE_DIR})
    set(LINK_DIRS ${LLAMA_LIB_DIR})
    set(LIBRARIES llama)
    set(EXTRA_SRCS
        bb.cc
        build-info.cpp
        clip.cpp
        common.cpp
        grammar-parser.cpp
        json-schema-to-grammar.cpp
        llava.cpp
        sampling.cpp)
else()
    set(ION_BB_BUILD_llm FALSE)
endif()
