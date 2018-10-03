"""Generates and uploads test results for keras tf models based tests."""
from __future__ import print_function
from ast import literal_eval
import os
import yaml
from test_runners.common import util


def process_folder(folder_path, report_config=None, test_config=None):
  """Process one or more results of a single test found in the folder path.

  Args:
    folder_path: Folder to recursively search for results files, e.g.
      worker_0_stdout.log
    report_config: dict based config information normally passed down from a
      higher level harness with high level system information.
  """
  report_config = {} if report_config is None else report_config
  results = _collect_results(folder_path, test_config)
  agg_result = util.report_aggregate_results(results)

  util.upload_results(
      report_config,
      agg_result,
      framework='tensorflow',
      test_harness='keras_tf_models')


def _collect_results(folder_path, test_config=None):
  """Walks folder path looking for and parsing results files."""
  results = []
  for r, _, files in os.walk(folder_path):
    for f in files:
      if f == 'config.yaml':
        result = {}
        process_base_result_files(result, os.path.join(r, f))
        result_file = os.path.join(r, 'worker_0_stdout.log')
        parse_result_file(result, result_file, test_config)
        results.append(result)
  return results


def process_base_result_files(result, config_file_path):
  """Process config.yaml file and extra_results.yaml."""

   # Get the config
  f = open(config_file_path, 'r')
  config = yaml.safe_load(f)
  result['config'] = config

  # Number of gpus = number of servers * number of gpus
  if 'gpus' in config:
    result['gpu'] = int(config['gpus'])
  # Avoids files that might have multiple total lines in them.
  # First line found wins.

  result['test_id'] = config['test_id']

  if 'use_synthetic_data' in config['args']:
    result['data_type'] = 'synth'
  else:
    result['data_type'] = 'real'

  result_dir = os.path.dirname(config_file_path)
  result['result_dir'] = result_dir

  # Load extra results
  extra_results_file = os.path.join(result_dir, 'extra_results.yaml')
  try:
    f = open(extra_results_file, 'r')
    extra_results = yaml.safe_load(f)
    result['raw_extra_results'] = extra_results
  except IOError:
    extra_results = None
    print('{}  not found.'.format(extra_results_file))


def parse_result_file(result, result_file_path, test_config=None):
  """Parses a result file."""
  try:
    result_file = open(result_file_path, 'r')
  except IOError:
    print('{}  not found.'.format(result_file_path))
    return
  samples = 0
  sum_speed = 0

  # number of samples in 100 batches = num_gpus * batch_size * 100
  num_samples = test_config['gpus'] * test_config['batch_size'] * 100

  # Processes results file and aggregates the results of one run.
  for line in result_file:
    # For eg: "BenchmarkMetric: {'num_batches': 100, 'time_taken': 55.004655}"
    if (line.find('BenchmarkMetric') != -1 and
        line.find('num_batches') != -1):
      start_index = line.find('{')
      end_index = line.find('}')
      line_parsed = line[start_index:end_index] + '}'
      metric_dict = literal_eval(line_parsed)
      # metric dict will be of the following format:
      # {'num_batches':100, 'time_taken': 54.434641}
      # Ignores first 100 batches as a warm up
      if metric_dict['num_batches'] > 100:
        current_imgs_sec = num_samples / metric_dict['time_taken']
        sum_speed += current_imgs_sec
        samples +=1

  result['imgs_sec'] = sum_speed / samples
  result['batches_sampled'] = samples
  return result


def get_config(result_dir):
  config_file = os.path.join(result_dir, 'config.yaml')
  with open(config_file) as f:
    config = yaml.safe_load(f)
  return config
