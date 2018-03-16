
# Start from mxnet official base with nccl. which should be the same as
# with GPU but for some reason I see a very small regression for
# kv-store=device compared with the normal GPU build.  Makes zero sense based
# on the code I skimmed as nccl should never get touched without kv-store=nccl.
FROM mxnet/python:1.1.0_nccl

# File Author / Maintainer
MAINTAINER Toby Boyd

RUN pip install --upgrade google-api-python-client pyyaml google-cloud
