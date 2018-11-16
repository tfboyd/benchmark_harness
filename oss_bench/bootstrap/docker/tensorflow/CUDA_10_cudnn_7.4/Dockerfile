
# Start from tensorflow nightly GPU.
FROM nvidia/cuda:10.0-base-ubuntu16.04

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
      git \
      software-properties-common \
      python \
      python-dev \
      python-pip \
      python-setuptools \
      cuda-command-line-tools-10-0 \
      cuda-cublas-10-0 \
      cuda-cudart-10-0 \
      cuda-cufft-10-0 \
      cuda-curand-10-0 \
      cuda-cusolver-10-0 \
      cuda-cusparse-10-0 \
      libcudnn7=7.4.1.5-1+cuda10.0 \
      libnccl2=2.3.5-2+cuda10.0 \
      ca-certificates \
      wget

RUN pip install --upgrade pip

RUN pip install --upgrade --force-reinstall https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud google-cloud-bigquery

# Needed by tf_models code (may already be meet by other needs above)
# tf_models also wants google-cloud-bigquery but resulted in needless warnings.
# RUN pip install --upgrade psutil py-cpuinfo
