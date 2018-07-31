
# Start from tensorflow latest.  At some point nightly
FROM tensorflow/tensorflow:1.9.0-rc0-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Fix cuDNN version
# RUN apt-get -y install --allow-downgrades libcudnn7=7.1.3.16-1+cuda9.0

# Install everything needed via apt-get
RUN apt-get install -y git

# Install the python basics
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud google-cloud-bigquery

