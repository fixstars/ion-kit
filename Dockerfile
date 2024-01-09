FROM ubuntu:22.04

# Basic package
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        curl \
        cmake \
        build-essential \
        zlib1g-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://github.com/halide/Halide/releases/download/v16.0.0/Halide-16.0.0-x86-64-linux-1e963ff817ef0968cc25d811a25a7350c8953ee6.tar.gz | tar zx --strip-components=1 -C /usr/
