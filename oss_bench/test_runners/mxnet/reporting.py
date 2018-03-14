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
      report_config,
      agg_result,
      framework='mxnet',
      test_harness='mxnet_benchmark')


def _collect_results(folder_path):
  """Walks folder path looking for and parsing results files."""
  results = []
  for r, _, files in os.walk(folder_path):
    for f in files:
      if f == 'worker_0_stdout.log':
        result_file = os.path.join(r, f)
        results.append(parse_result_file(result_file))
  return results


def parse_result_file(result_file_path):
  """Parses a result file.

  Note: MXNet prints samples/sec every so often to the console and then code
  below averages a number of the rows together.  This is mathematically
  incorrect as speed cannot be averaged this way and should be total time /
  total_items_processes. Getting a more correct number would require changing
  how the MXNet code works, which can be done but was not needed for the initial
  number.

  Args:
    result_file_path: Path to file to parse

  Returns:
    `dict` representing the results.
  """
  result = {}
  result_file = open(result_file_path, 'r')
  samples = 0
  sum_speed = 0

  # Get the config
  result_dir = os.path.dirname(result_file_path)
  config = get_config(result_dir)
  result['config'] = config
  result['result_dir'] = result_dir
  result['test_id'] = config['test_id']

  if 'data-train' in config['args']:
    result['data_type'] = 'real'
  else:
    result['data_type'] = 'synth'

  # Number of gpus = number of servers * number of gpus
  if 'gpus' in config:
    result['gpu'] = int(config['gpus'])

  # Processes results file and aggregates the results of one run.
  for line in result_file:
    # Rows with 'Epoch[' and 'samples/sec' contain result data to aggregate.
    if line.find('Epoch[') > 0 and line.find('samples/sec') > 0:
      parts = line.split()
      batch = int(parts[2].replace(']', '').replace('[', ''))
      # Ignores first 10 batches as a warm up, tf_benchmarks does the same.
      if batch > 10:
        sum_speed += float(parts[4].rstrip())
        samples += 1

      # After 100 batches are found, calculate average and break.
      if batch > 100:
        break
  result['imgs_sec'] = sum_speed / samples
  result['batches_sampled'] = samples
  return result


def get_config(result_dir):
  config_file = os.path.join(result_dir, 'config.yaml')
  with open(config_file) as f:
    config = yaml.safe_load(f)
  return config
