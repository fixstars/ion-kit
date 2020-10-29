# 1. Builder stage
FROM ubuntu:18.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        curl \
        doxygen \
        g++ \
        gcc \
        git \
        git-lfs \
        graphviz \
        make \
        ninja-build \
        python3 \
        python3-pip \
        zlib1g-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN curl -L https://ion-archives.s3-us-west-2.amazonaws.com/Halide-8c7129af4.tar.gz | tar zx -C /usr/local/
ENV HALIDE_ROOT=/usr/local/Halide

RUN pip3 install sphinx_rtd_theme breathe


# 2.1. Build for ion-core stage
FROM builder AS ion-core-build

COPY . ion-kit
WORKDIR ion-kit
RUN git checkout `git describe --tags --abbrev=0`
RUN cmake -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D HALIDE_ROOT=/usr/local/Halide \
        -D ION_BUILD_ALL_BB=OFF \
        -D ION_BUILD_DOC=ON \
        -D ION_BUILD_EXAMPLE=OFF \
        -D ION_BUNDLE_HALIDE=ON \
        -D WITH_CUDA=OFF .
RUN cmake --build .
RUN cmake --build . --target package
RUN mv ion-kit-`git describe --tags --abbrev=0 | sed -e "s/v\([0-9]\)/\1/"`-Linux.deb ion-core.deb


# 2.3. Runtime stage
FROM ubuntu:18.04 AS ion-core

COPY --from=ion-core-build /ion-core/ion-core.deb /ion-core.deb
RUN dpkg -i ion-core.deb && rm -f ion-core.deb
ENV ION_CORE_ROOT=/usr

# 2.4. Release stage
FROM scratch AS ion-core-release

COPY --from=ion-core-build ion-core.deb /opt/
