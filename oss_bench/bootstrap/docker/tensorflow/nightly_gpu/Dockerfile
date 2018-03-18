
# Start from tensorflow nightly GPU.
FROM tensorflow/tensorflow:nightly-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Needed until eng_prod fixed docker nightly to include actual nightly
# added 15-MAR-2018 should not be needed.
RUN pip install --upgrade --force-reinstall tf-nightly-gpu

RUN apt-get -y install --allow-downgrades libcudnn7=7.0.5.15-1+cuda9.0

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
