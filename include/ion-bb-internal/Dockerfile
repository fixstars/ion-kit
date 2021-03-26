# Build command:
# $ docker buildx build --platform linux/amd64,linux/arm64 --push -t localhost:5000/ion-bb-internal .

FROM ubuntu:18.04

#
# Basic package
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        g++ \
        gcc \
        make
