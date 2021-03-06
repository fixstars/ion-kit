#
# 1. Builder stage for ion-core
# docker build -t --target builder -t .
#

FROM ubuntu:18.04 AS ion-core-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
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

RUN curl -L https://github.com/halide/Halide/releases/download/v10.0.0/Halide-10.0.0-x86-64-linux-db901f7f7084025abc3cbb9d17b0f2d3f1745900.tar.gz | tar zx -C /usr/local/

RUN pip3 install sphinx_rtd_theme breathe cmake==3.18.4.post1


#
# 2.1. Build stage for ion-core
# docker build --target ion-core-build -t docker.pkg.github.com/fixstars/ion-kit/ion-core:latest-build-ubuntu18.04 .
#

FROM ion-core-builder AS ion-core-build

COPY . ion-kit
WORKDIR ion-kit
RUN mkdir build
WORKDIR build
RUN cmake -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=ion-kit-install \
        -D Halide_DIR=/usr/local/halide/lib/cmake/Halide \
        -D HALIDE_ROOT=/usr/local/halide \
        -D ION_BUILD_ALL_BB=OFF \
        -D ION_BUILD_DOC=ON \
        -D ION_BUILD_TEST=ON \
        -D ION_BUILD_EXAMPLE=OFF \
        -D ION_BUNDLE_HALIDE=ON \
        -D WITH_CUDA=OFF ../
RUN cmake --build . --target install
RUN cmake --build . --target package
RUN find ./ -maxdepth 1 -name "ion-kit_*.deb" -exec cp {} ion-core.deb \;


#
# 2.2. Runtime stage for ion-core
# docker build --target ion-core-runtime -t docker.pkg.github.com/fixstars/ion-kit/ion-core:latest-runtime-ubuntu18.04 .
#

FROM ubuntu:18.04 AS ion-core-runtime

COPY --from=ion-core-build /ion-kit/build/ion-core.deb /ion-core.deb
RUN dpkg -i ion-core.deb && rm -f ion-core.deb


#
# 2.3. Release stage for ion-core
# docker build --target ion-core-release -t docker.pkg.github.com/fixstars/ion-kit/ion-core:latest-release-ubuntu18.04 .
#

FROM scratch AS ion-core-release

COPY --from=ion-core-build /ion-kit/build/ion-core.deb /opt/


#
# 3.1 Builder stage for ion-kit
#
#

FROM ion-core-builder AS ion-kit-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        libgtk2.0-0 \
        libopenexr22 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# OpenCV
RUN curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-x86_64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh

# ONNXRuntime
RUN curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/onnxruntime-linux-x64-gcc75-1.6.0.tar.gz | tar xz -C /usr --strip-components 1

#
# 3.2. Build stage for ion-kit
# docker build --target ion-kit-build -t docker.pkg.github.com/fixstars/ion-kit/ion-kit:latest-build-ubuntu18.04 .
#

FROM ion-kit-builder AS ion-kit-build

COPY . ion-kit
WORKDIR ion-kit
RUN mkdir build
WORKDIR build
RUN cmake -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=ion-kit-install \
        -D Halide_DIR=/usr/local/halide/lib/cmake/Halide \
        -D HALIDE_ROOT=/usr/local/halide \
        -D ION_BUILD_ALL_BB=ON \
        -D ION_BUILD_DOC=ON \
        -D ION_BUILD_TEST=ON \
        -D ION_BUILD_EXAMPLE=ON \
        -D ION_BUNDLE_HALIDE=ON \
        -D WITH_CUDA=OFF ../
RUN cmake --build . --target install
RUN cmake --build . --target package
RUN find ./ -maxdepth 1 -name "ion-kit_*.deb" -exec cp {} ion-kit.deb \;


#
# 3.3. Runtime stage for ion-kit
# docker build --target ion-kit-runtime -t docker.pkg.github.com/fixstars/ion-kit/ion-kit:latest-runtime-ubuntu18.04 .
#

FROM ubuntu:18.04 AS ion-kit-runtime

COPY --from=ion-kit-build /ion-kit/build/ion-kit.deb /ion-kit.deb
RUN dpkg -i ion-kit.deb && rm -f ion-kit.deb


#
# 3.4. Release stage for ion-kit
# docker build --target ion-kit-release -t docker.pkg.github.com/fixstars/ion-kit/ion-kit:latest-release-ubuntu18.04 .
#

FROM scratch AS ion-kit-release

COPY --from=ion-kit-build /ion-kit/build/ion-kit.deb /opt/
