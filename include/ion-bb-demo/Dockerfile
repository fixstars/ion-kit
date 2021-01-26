# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-demo .

FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive

ARG TARGETPLATFORM

#
# Basic package
#
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        curl \
        g++ \
        gcc \
        make

#
# Halide
#
RUN mkdir -p /usr/local/halide/include
RUN cd /usr/local/halide/include && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-8c7129af4/HalideRuntime.h -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-8c7129af4/HalideBuffer.h
ENV HALIDE_ROOT=/usr/local/halide

#
# OpenCV
#
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-x86_64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-arm64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi

#
# For realsense
#
RUN apt-get update && apt-get install -y \
        software-properties-common
RUN apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE
RUN add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
RUN apt-get update && apt-get install -y \
        librealsense2-utils \
        librealsense2-dev
