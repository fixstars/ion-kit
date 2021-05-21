# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-demo .

FROM ubuntu:18.04
ARG TARGETPLATFORM

#
# Basic package
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        curl

#
# Halide
#
RUN cd /usr/include \
        && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-0936c8bd3/HalideRuntime.h  \
                      -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-0936c8bd3/HalideBuffer.h
