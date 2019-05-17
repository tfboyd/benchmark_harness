"""Generates and uploads test results for mxnet based tests."""
from __future__ import print_function
import os

from test_runners.common import util
import yaml


def process_folder(folder_path, report_config=None):
  """Process one or more results of a single test found in the folder path.

  Args:
    folder_path: Folder to recursively search for results files, e.g.
      worker_0_stdout.log
    report_config: dict based config information normally passed down from a
      higher level harness with high level system information.
  """
  report_config = {} if report_config is None else report_config
  results = _collect_results(folder_path)
  agg_result = util.report_aggregate_results(results)

  util.upload_results(
      report_config, agg_result, framework='pytorch', test_harness='pytorch')


def _collect_results(folder_path):
  """Walks folder path looking for and parsing results files."""
  results = []
  for r, _, files in os.walk(folder_path):
    for f in files:
      if f in ('worker_0_stdout.log', 'worker_0_stdout.txt'):
        result_file = os.path.join(r, f)
        results.append(parse_result_file(result_file))
  return results


def parse_result_file(result_file_path):
  """Parses a result file.

  Note: Pytorch prints Time for the specific step printed. These can be added
  together and then divided to get an average then used with batch_size to get
  images per second.  This is similar (maybe the same) as tf_cnn_bencharks.

  Args:
    result_file_path: Path to file to parse

  Returns:
    `dict` representing the results.
  """
  result = {}
  result_file = open(result_file_path, 'r')
  samples = 0
  total_time = 0

  # Get the config
  result_dir = os.path.dirname(result_file_path)
  config = get_config(result_dir)
  result['config'] = config
  result['result_dir'] = result_dir
  result['test_id'] = config['test_id']
  result['data_type'] = 'real'

  # Number of gpus = number of servers * number of gpus
  if 'gpus' in config:
    result['gpu'] = int(config['gpus'])

  # Processes results file and aggregates the results of one run.
  for line in result_file:
    # Identifies lines with performance data to aggregate
    if line.find('Epoch') > -1:
      #  Example: Epoch: [0][  0/40037] Time 12.788 (12.788)  Data 9.518 (9.518)
      parts = line.split()
      batch = int(parts[2].split('/')[0])
      # Ignores first 10 batches as a warm up, tf_benchmarks does the same.
      if batch > 20:
        total_time += float(parts[4].rstrip())
        samples += 1

      # After 100 batches are found, calculate average and break.
      if batch > 200:
        break
  total_batch_size = config['batch_size'] * config['gpus']
  result['imgs_sec'] = (1 / (total_time / samples)) * total_batch_size
  result['batches_sampled'] = samples
  return result


def get_config(result_dir):
  config_file = os.path.join(result_dir, 'config.yaml')
  with open(config_file) as f:
    config = yaml.safe_load(f)
  return config
