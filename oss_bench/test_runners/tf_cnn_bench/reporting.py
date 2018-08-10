"""Generates and updates test results from stored log files."""
from __future__ import print_function
import os

from test_runners.common import util
from upload import result_info
import yaml


def process_folder(folder_path, report_config=None):
  """Process and print aggregated results found in folder.

  Args:
    folder_path: Folder to recursively search for results files, e.g.
      worker_0_stdout.log
    report_config: dict config information normally passed down from a
      higher level harness with high level system information.
  """
  report_config = {} if report_config is None else report_config
  results = _collect_results(folder_path)
  agg_result = util.report_aggregate_results(results)

  util.upload_results(
      report_config,
      agg_result,
      framework='tensorflow',
      test_harness='tf_cnn_benchmark')


def _collect_results(folder_path):
  """Walks folder path looking for and parsing results files."""
  results = []
  for r, _, files in os.walk(folder_path):
    for f in files:
      if f == 'config.yaml':
        result = {}
        process_base_result_files(result, os.path.join(r, f))
        result_file = os.path.join(r, 'worker_0_stdout.log')
        parse_result_file(result, result_file)
        eval_file = os.path.join(r, 'eval_0_stdout.log')
        parse_eval_result_file(result, eval_file)
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

  if 'data_dir' in config:
    result['data_type'] = 'real'
  else:
    result['data_type'] = 'synth'

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


def parse_result_file(result, result_file_path):
  """Parses a result file."""
  try:
    result_file = open(result_file_path, 'r')
  except IOError:
    print('{}  not found.'.format(result_file_path))
    return

  for line in result_file:
    # Summary line starts with images/sec
    if line.find('total images/sec') == 0:
      parts = line.split(' ')
      result['imgs_sec'] = float(parts[2].rstrip())
      # Avoids files that might have multiple total lines in them.
      break


def parse_eval_result_file(result, result_file_path):
  """Parses a eval result file."""
  results = []
  try:
    result_file = open(result_file_path, 'r')
  except IOError:
    print('{}  not found.'.format(result_file_path))
    return

  exp_per_sec = 0
  for line in result_file:
    # Summary line starts with images/sec
    if line.find('Accuracy @') == 0:
      results = []
      parts = line.split(' ')
      top_1 = float(parts[4].rstrip())
      result_info.build_result_info(results,
                                    top_1,
                                    'top_1',
                                    result_units='accuracy')
      top_5 = float(parts[9].rstrip())
      result_info.build_result_info(results,
                                    top_5,
                                    'top_5',
                                    result_units='accuracy')
    elif line.find('total images/sec') == 0:
      parts = line.split(' ')
      exp_per_sec = float(parts[2].rstrip())

  if exp_per_sec:
    result_info.build_result_info(results,
                                  exp_per_sec,
                                  'eval_exp_per_sec',
                                  result_units='exp_per_sec')

  # if examples per second is not already set this may be an eval only test.
  if exp_per_sec and 'imgs_sec' not in result:
    result['imgs_sec'] = exp_per_sec

  if 'raw_extra_results' in result:
    result['raw_extra_results'].extend(results)
  else:
    result['raw_extra_results'] = results


def check_oom(result_file_path):
  result_file = open(result_file_path, 'r')
  for line in result_file:
    if line.find('OOM when allocating tensor') > -1:
      return True
  return False


def oom_batch_size_search(low, high, current, current_oom):
  next_val = None
  if current_oom:
    high = current
    next_val = high - ((high - low) / 2)
  else:
    low = current
    next_val = low + ((high - low) / 2)
  # Indicates there is nothing left to test.
  if next_val == current:
    return low, high, -1
  else:
    return low, high, next_val
