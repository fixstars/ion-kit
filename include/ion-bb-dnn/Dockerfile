# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-dnn .

FROM ubuntu:18.04
ARG TARGETPLATFORM

#
# Basic package
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        curl \
        zlib1g-dev \
        libgtk-3-0

#
# Halide
#
RUN cd /usr/include \
        && curl -fsSL -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-v12.0.1-9a99bee06/HalideRuntime.h  \
                      -O https://ion-archives.s3-us-west-2.amazonaws.com/Halide-Headers-v12.0.1-9a99bee06/HalideBuffer.h

#
# genesis-runtime
#
# Deps: curl, libgtk-3-0
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://genesis-runtime.s3-us-west-2.amazonaws.com/genesis-runtime-generic-amd64-v0.0.7.tar.gz | tar xz -C /usr --strip-components 1 ;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://genesis-runtime.s3-us-west-2.amazonaws.com/genesis-runtime-generic-arm64-v0.0.7.tar.gz | tar xz -C /usr --strip-components 1 ;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi
