# RogueZero: Benchmark Harness
RogueZero is a benchmark harness that is a pure means to an end solution to benchmark
TensorFlow as well as other platforms.  It can be run against code in Docker or standalone with
reporting to console (for testing) or BigQuery or reporting and alerts.


```bash

# Install needed software
# TODO: make a requirements.txt file
pip install --upgrade google-api-python-client pyyaml paramiko google-cloud google-cloud-bigquery


# Running tests without Docker

# Pick a folder as the workspace
mkdir git
cd git
git clone https://github.com/tfboyd/benchmark_harness.git 

cd benchmark_harness/oss_bench
# Runs a test config (configs/dev/default.yaml') targeting TensorFlow that will run on most GPUs.  
python -m harness.controller --workspace=<path to workspace with /git/benchmark_harness>


```

## How to add a new test runner


## Development basics

### Unit tests

```bash
# From oss_bench directory
python -m unittest discover -p '*_test.py'
```


