# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-demo .

FROM ubuntu:18.04
ARG TARGETPLATFORM

#
# Basic package
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        curl \
        zlib1g-dev \
        libgtk2.0-0 \
        libopenexr22

#
# Halide
#
RUN cd /usr/include \
        && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-7bf2c2d7f/HalideRuntime.h  \
                      -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-7bf2c2d7f/HalideBuffer.h
#
# OpenCV
#
# Deps: libgtk2.0-0, libopenexr22
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-x86_64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-aarch64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi

#
# RealSense
#
RUN apt-get update && apt-get install -y \
        software-properties-common
RUN apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE
RUN add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
RUN apt-get update && apt-get install -y \
        librealsense2-utils \
        librealsense2-dev
