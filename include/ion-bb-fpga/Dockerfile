# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-genesis-cloud .

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
        make

#
# Halide
#
RUN mkdir -p /usr/local/halide/include
RUN cd /usr/local/halide/include \
        && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-7bf2c2d7f/HalideRuntime.h  \
                      -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-7bf2c2d7f/HalideBuffer.h
ENV HALIDE_ROOT=/usr/local/halide
