
# Start from tensorflow nightly GPU.
FROM nvidia/cuda:10.0-base-ubuntu16.04

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
      curl \
      build-essential \
      git \
      software-properties-common \
      python-software-properties \
      cuda-command-line-tools-10-0 \
      cuda-cublas-dev-10-0 \
      cuda-cudart-dev-10-0 \
      cuda-cufft-dev-10-0 \
      cuda-curand-dev-10-0 \
      cuda-cusolver-dev-10-0 \
      cuda-cusparse-dev-10-0 \
      libcudnn7=7.3.1.20-1+cuda10.0 \
      libcudnn7-dev=7.3.1.20-1+cuda10.0 \
      libnccl2=2.3.5-2+cuda10.0 \
      libnccl-dev=2.3.5-2+cuda10.0 \
      python \
      python-pip \
      python-setuptools \
      ca-certificates \
      virtualenv \
      htop \
      wget

RUN pip install --upgrade pip

RUN pip install --upgrade --force-reinstall https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0rc0.1a6dea3.AVX2-cp27-cp27mu-linux_x86_64.whl

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud google-cloud-bigquery

# Needed by tf_models code (may already be meet by other needs above)
# tf_models also wants google-cloud-bigquery but resulted in needless warnings.
RUN pip install --upgrade psutil py-cpuinfo
