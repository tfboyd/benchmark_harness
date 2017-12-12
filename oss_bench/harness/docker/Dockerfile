
# Start from tensorflow latest.  At some point nightly
FROM tensorflow/tensorflow:latest-gpu

# File Author / Maintainer
MAINTAINER Toby Boyd

# Update apt-get so latest apps can get installed
RUN apt-get update

# Install everything needed via apt-get
RUN apt-get install -y git

# Install the python basics
#RUN apt-get install -y python python-dev python-distribute python-pip

# pip install paramiko
# pip install pyyaml
RUN pip install --upgrade google-api-python-client pyyaml paramiko google-cloud

# likely not needed if you have google-cloud
#pip install google-auth

#pip install --upgrade google-cloud