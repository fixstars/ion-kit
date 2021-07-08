#
# 1. Builder stage for ion-core
# docker build -t --target builder -t .
#
FROM ubuntu:20.04 AS ion-kit-builder

# Basic package
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

# Extra tools
RUN pip3 install sphinx_rtd_theme breathe cmake==3.18.4.post1

# Halide
RUN mkdir /usr/local/halide
RUN curl -L https://github.com/halide/Halide/releases/download/v12.0.1/Halide-12.0.1-x86-64-linux-5dabcaa9effca1067f907f6c8ea212f3d2b1d99a.tar.gz | tar zx -C /usr/local/halide --strip-components 1

# OpenCV
RUN curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.2-x86_64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh

#
# 2. Build stage for ion-kit
# docker build --target ion-kit-build -t docker.pkg.github.com/fixstars/ion-kit/ion-kit:latest-build-ubuntu18.04 .
#
FROM ion-kit-builder AS ion-kit-build

COPY . ion-kit
WORKDIR ion-kit
RUN mkdir build
WORKDIR build
RUN cmake -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_PREFIX_PATH=/usr/local/halide \
        -D CMAKE_INSTALL_PREFIX=./ion-kit-install \
        -D ION_BUILD_ALL_BB=ON \
        -D ION_BUILD_DOC=ON \
        -D ION_BUILD_TEST=OFF \
        -D ION_BUILD_EXAMPLE=OFF \
        -D ION_BUNDLE_HALIDE=ON \
        -D WITH_CUDA=OFF ..

# To avoid to fail cpack when copying unicode filename certs
RUN rm -rf /usr/share/ca-certificates/mozilla/NetLock*

RUN cmake --build . --target install
RUN cmake --build . --target package

#
# 3. Runtime stage for ion-kit
# docker build --target ion-kit-runtime -t docker.pkg.github.com/fixstars/ion-kit/ion-kit:latest-runtime-ubuntu18.04 .
#
FROM ubuntu:20.04 AS ion-kit-runtime

COPY --from=ion-kit-build /ion-kit/build/ion-kit*.deb /ion-kit.deb
RUN dpkg -i ion-kit.deb && rm -f ion-kit.deb
