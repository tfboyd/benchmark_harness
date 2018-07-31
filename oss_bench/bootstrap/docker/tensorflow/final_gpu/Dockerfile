# Start from tensorflow latest.  At some point nightly
FROM tensorflow/tensorflow:1.8.0-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update
RUN pip install --upgrade pip

# Install everything needed via apt-get
RUN apt-get install -y git

# Install the python basics
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud google-cloud-bigquery

# Needed by tf_models code (may already be meet by other needs above)
# tf_models also wants google-cloud-bigquery but resulted in needless warnings.
RUN pip install --upgrade psutil py-cpuinfo
