# Build and run environment: Ubuntu 24.04, GCC 13, CMake, Ninja, Python for model export and the
# parity check, valgrind for callgrind profiles. Used locally (docker run -v $PWD:/src) and as the
# reference environment for every number in results/.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ cmake ninja-build git curl ca-certificates xz-utils \
      python3 python3-pip python3-venv valgrind linux-tools-common procps time \
  && rm -rf /var/lib/apt/lists/*

# Python side: ONNX Runtime (same version as the C++ library) for the parity check and the load
# generator, and the exporter for the small transformer.
RUN python3 -m venv /opt/py && /opt/py/bin/pip install --no-cache-dir \
      onnxruntime==1.20.1 "onnx<1.18" "numpy<2" pillow requests tokenizers
ENV PATH=/opt/py/bin:$PATH

WORKDIR /src
