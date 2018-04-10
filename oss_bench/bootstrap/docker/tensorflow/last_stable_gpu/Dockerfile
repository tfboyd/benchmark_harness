
# Start from tensorflow latest.  At some point nightly
FROM tensorflow/tensorflow:1.7.0-rc1-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Install everything needed via apt-get
RUN apt-get install -y git

# Install the python basics
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud

