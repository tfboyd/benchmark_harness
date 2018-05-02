
# Start from tensorflow nightly GPU.
FROM tensorflow/tensorflow:nightly-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

RUN pip install --upgrade pip
#RUN apt-get -y install --allow-downgrades libcudnn7=7.0.5.15-1+cuda9.0

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud
# Needed by tf_models code (may already be meet by other needs above)
RUN pip install --upgrade psutil py-cpuinfo google-cloud-bigquery
