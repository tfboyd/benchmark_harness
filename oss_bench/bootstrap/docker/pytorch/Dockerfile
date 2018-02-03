
# PyTorch Docker.  Starts with tensorflow due to it having CUDA and cuDNN.
# Maybe even other libraries but TODO(tobyboyd@): Start Docker from the base
#nvidia-docker for more control over the CUDA and cuDNN version.

FROM tensorflow/tensorflow:nightly-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl torchvision
