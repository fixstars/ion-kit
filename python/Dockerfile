# FROM python:3.6.9

# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvidia/cuda:10.2-devel-ubuntu18.04


ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.6.9
ARG PYTHON_ROOT=/usr/local
ARG PYENV_ROOT=/tmp/.pyenv
ARG PYBUILD_ROOT=/tmp/python-build

RUN apt update && apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python-openssl \
    git \
  && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
  && PREFIX=$PYBUILD_ROOT $PYENV_ROOT/plugins/python-build/install.sh \
  && $PYBUILD_ROOT/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
  && rm -rf $PYBUILD_ROOT $PYENV_ROOT


ADD ./install_ion.sh /tmp/
RUN bash /tmp/install_ion.sh && \
    echo "/root/local/halide/bin" >> /etc/ld.so.conf.d/halide.conf && \
    echo "/root/local/onnxruntime/lib" >> /etc/ld.so.conf.d/onnxruntime.conf && \
    echo "/root/local/ion-kit/lib" >> /etc/ld.so.conf.d/ion-kit.conf && \
    ldconfig && \
    rm /tmp/install_ion.sh


ADD . /ionpy

RUN pip3 install /ionpy
