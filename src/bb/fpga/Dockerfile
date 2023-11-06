# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-genesis-cloud .

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
        && curl -fsSL -O https://ion-kit.s3.us-west-2.amazonaws.com/dependencies/Halide-Headers-v12.0.1-9a99bee06/HalideRuntime.h  \
                      -O https://ion-kit.s3.us-west-2.amazonaws.com/dependencies/Halide-Headers-v12.0.1-9a99bee06/HalideBuffer.h
