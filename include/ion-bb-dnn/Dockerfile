# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-dnn .

FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        curl \
        g++ \
        gcc \
        libopencv-dev \
        make \
        uuid-dev

ARG TARGETPLATFORM
RUN mkdir -p /usr/local/halide/include
RUN cd /usr/local/halide/include && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-8c7129af4/HalideRuntime.h -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-8c7129af4/HalideBuffer.h
ENV HALIDE_ROOT=/usr/local/halide


ARG TARGETPLATFORM
RUN mkdir /usr/local/onnxruntime
# NOTE: This dockerfile is used for Building and Running Preview, then we need to utilize version which is not depends on CUDA ATM.
# For a performance measurement, installed onnxruntime version will be used.
# TODO: Build 1.5.1 for arm64

# RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
#         curl -L https://ion-archives.s3-us-west-2.amazonaws.com/onnxruntime-linux-x64-gpu-tensorrt-1.5.0.tar.gz | tar xz -C /usr/local/onnxruntime --strip-components 1 ;\
#     elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
#         curl -L https://ion-archives.s3-us-west-2.amazonaws.com/onnxruntime-linux-arm64-gpu-tensorrt-1.4.0.tar.gz | tar xz -C /usr/local/onnxruntime --strip-components 1 ;\
#     else \
#         echo "Unsupported platform" && exit -1 ;\
#     fi
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/onnxruntime-linux-x64-1.5.1.tgz | tar xz -C /usr/local/onnxruntime --strip-components 1 ;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://ion-archives.s3-us-west-2.amazonaws.com/onnxruntime-linux-arm64-1.3.0.tgz  | tar xz -C /usr/local/onnxruntime --strip-components 1 ;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi
ENV ONNXRUNTIME_ROOT=/usr/local/onnxruntime
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib

ARG TARGETPLATFORM
RUN mkdir /usr/local/ion-bb-dnn-models
RUN curl -L https://ion-archives.s3-us-west-2.amazonaws.com/ion-bb-dnn-models-73504d0.tgz | tar xz -C /usr/local/ion-bb-dnn-models --strip-components 1
ENV ION_BB_DNN_MODELS_ROOT=/usr/local/ion-bb-dnn-models
