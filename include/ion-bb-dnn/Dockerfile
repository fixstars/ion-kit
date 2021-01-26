# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-dnn .

FROM ubuntu:18.04
ARG TARGETPLATFORM

#
# Basic package
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        curl \
        g++ \
        gcc \
        make \
        libgtk2.0-0 \
        libopenexr22

#
# Halide
#
RUN mkdir -p /usr/local/halide/include
RUN cd /usr/local/halide/include && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-8c7129af4/HalideRuntime.h -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-8c7129af4/HalideBuffer.h
ENV HALIDE_ROOT=/usr/local/halide

#
# OpenCV
#
# Deps: libgtk2.0-0, libopenexr22
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-x86_64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/OpenCV-4.5.1-arm64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi

#
# ONNXRuntime
#
RUN mkdir /usr/local/onnxruntime
# NOTE: This dockerfile is used for Building and Running Preview, then we need to utilize version which is not depends on CUDA ATM.
# For a performance measurement, installed onnxruntime version will be used.
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/onnxruntime-linux-x64-gcc75-1.6.0.tgz | tar xz -C /usr/local/onnxruntime --strip-components 1 ;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/onnxruntime-linux-arm64-gcc75-1.6.0.tgz  | tar xz -C /usr/local/onnxruntime --strip-components 1 ;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi
ENV ONNXRUNTIME_ROOT=/usr/local/onnxruntime
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib
