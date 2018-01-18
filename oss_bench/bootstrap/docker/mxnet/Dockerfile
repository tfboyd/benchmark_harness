
# Start from tensorflow nightly GPU.
FROM mxnet/python:gpu_1.0.0_cuda9

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Install everything needed via apt-get
# MXNet Docker comes with git.
# RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
