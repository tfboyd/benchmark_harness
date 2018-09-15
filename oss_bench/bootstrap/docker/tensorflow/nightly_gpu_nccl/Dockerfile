
# Start from nvidia/cuda which has nccl 2.x
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update
RUN sudo apt-get install -y cuda9.2 cuda-cublas-9-2 cuda-cufft-9-2 cuda-curand-9-2 \
  cuda-cusolver-9-2 cuda-cusparse-9-2 libcudnn7=7.2.1.38-1+cuda9.2 \
   libnccl2=2.2.13-1+cuda9.2 cuda-command-line-tools-9-2

RUN pip install --upgrade --force-reinstall https://storage.googleapis.com/tf-performance/public/tensorflow-1.11.0rc0.tag.1e43819.AVX2.CUDA9_2-cp27-cp27mu-linux_x86_64.whl

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
# Needed by tf_models code (may already be meet by other needs above)
# tf_models also wants google-cloud-bigquery but resulted in needless warnings.
RUN pip install --upgrade psutil py-cpuinfo
