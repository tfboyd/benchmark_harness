
# Start from tensorflow nightly GPU.
FROM tensorflow/tensorflow:nightly-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

RUN pip install --upgrade pip
# RUN apt-get -y install libcudnn7=7.2.1.38-1+cuda9.0
RUN pip install --upgrade --force-reinstall tf-nightly-gpu

# Install everything needed via apt-get
RUN apt-get install -y git
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud google-cloud-bigquery
# Needed by tf_models code (may already be meet by other needs above)
# tf_models also wants google-cloud-bigquery but resulted in needless warnings.
RUN pip install --upgrade psutil py-cpuinfo
