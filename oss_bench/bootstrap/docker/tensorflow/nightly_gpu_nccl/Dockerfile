
# Start from nvidia/cuda which has nccl 2.x
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install --upgrade pip
RUN pip install --upgrade --force-reinstall https://s3-us-west-2.amazonaws.com/tf-benchmark/tf_binary/tensorflow-1.8.0.f528eba.CUDA_9_2.NCCL_2-cp27-cp27mu-linux_x86_64.whl

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
# Needed by tf_models code (may already be meet by other needs above)
# tf_models also wants google-cloud-bigquery but resulted in needless warnings.
RUN pip install --upgrade psutil py-cpuinfo
