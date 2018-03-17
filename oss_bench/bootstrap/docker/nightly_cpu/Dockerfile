
# Start from tensorflow nightly CPU.
FROM tensorflow/tensorflow:nightly

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
