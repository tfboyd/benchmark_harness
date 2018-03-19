
# Start from tensorflow nightly CPU.
FROM tensorflow/tensorflow:nightly

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Install custom MKL build
RUN pip install --upgrade --force-reinstall https://s3-us-west-2.amazonaws.com/tf-benchmark/tf_binary/tensorflow-1.5.0.926fc13_MKL-cp27-cp27mu-linux_x86_64.whl

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
