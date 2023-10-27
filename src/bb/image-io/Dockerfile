# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-demo .

FROM ubuntu:18.04
ARG TARGETPLATFORM

#
# Basic package
#
ENV DEBIAN_FRONTEND=noninteractive

# NOTE: Installing humanity-icon-theme separately because this package is offtenly corrupted when installing with libgtk-3-0 at the same time
RUN apt-get update && apt-get install -y \
        humanity-icon-theme
RUN apt-get install -y \
        curl \
        zlib1g-dev \
        libgtk-3-0

#
# Halide
#
RUN cd /usr/include \
        && curl -fsSL -O https://ion-kit.s3.us-west-2.amazonaws.com/dependencies/Halide-Headers-v12.0.1-9a99bee06/HalideRuntime.h  \
                      -O https://ion-kit.s3.us-west-2.amazonaws.com/dependencies/Halide-Headers-v12.0.1-9a99bee06/HalideBuffer.h
#
# OpenCV
#
# Deps: curl, libgtk-3-0
RUN if [ "x$TARGETPLATFORM" = "xlinux/amd64" ]; then \
        curl -L https://ion-kit.s3.us-west-2.amazonaws.com/dependencies/genesis-runtime/OpenCV-4.5.2-x86_64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    elif [ "x$TARGETPLATFORM" = "xlinux/arm64" ]; then \
        curl -L https://ion-kit.s3.us-west-2.amazonaws.com/dependencies/genesis-runtime/OpenCV-4.5.2-aarch64-gcc75.sh -o x.sh && sh x.sh --skip-license --prefix=/usr && rm x.sh;\
    else \
        echo "Unsupported platform" && exit -1 ;\
    fi
