FROM ubuntu:18.04

# Basic package
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        cmake \
        g++ \
        gcc \
        git \
        libssl-dev \
        make \
        ninja-build \
        python3 \
        python3-pip \
        zlib1g-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Extra tools
RUN pip3 install scikit-build
RUN pip3 install cmake

# LLVM 12
RUN git clone https://github.com/llvm/llvm-project.git -b release/12.x --depth=1
RUN mkdir llvm-project/build && cd llvm-project/build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
        -DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;Hexagon;PowerPC;AMDGPU;RISCV" \
        -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_EH=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_BUILD_32_BITS=OFF \
        ../llvm && \
    cmake --build . --target install

# Halide 12
ARG HALIDE_GIT_URL=invalid
ARG HALIDE_GIT_BRANCH=invalid
RUN git clone ${HALIDE_GIT_URL} -b ${HALIDE_GIT_BRANCH} --depth=1
RUN mkdir -p Halide/build && cd Halide/build && \
    cmake -GNinja \
        -DHALIDE_ENABLE_RTTI=ON \
        -DWITH_APPS=OFF \
        .. && \
    cmake --build . --target install
