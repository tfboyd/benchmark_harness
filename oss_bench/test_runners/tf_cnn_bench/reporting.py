"""Generates and updates test results from stored log files."""
import os

from test_runners.common import util
from upload import result_info
from upload import result_upload
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

  # Sets defaults on report_config if any required values are missing.
  report_config = util.report_config_defaults(
      report_config, test_harness='tf_cnn_benchmark')

  # Main result config
  test_result, results = result_info.build_test_result(
      agg_result['config']['test_id'],
      agg_result['mean'],
      result_type='exp_per_sec',
      test_harness=report_config['test_harness'],
      test_environment=report_config['test_environment'])

  test_info = result_info.build_test_info(
      framework_version=report_config.get('framework_version'),
      framework_describe=report_config.get('framework_describe'),
      batch_size=agg_result['config']['batch_size'],
      model=agg_result['config']['model'],
      accel_cnt=agg_result['gpu'],
      cmd=agg_result['config']['cmd'])
  # Stores info on each of the repos used in testing.
  if 'git_repo_info' in report_config:
    test_info['git_info'] = report_config['git_repo_info']

  system_info = result_info.build_system_info(
      platform=report_config['platform'],
      platform_type=report_config['platform_type'],
      accel_type=report_config['accel_type'])

  print 'Uploading test results...'
  result_upload.upload_result(
      test_result,
      results,
      report_config['report_project'],
      dataset=report_config['report_dataset'],
      table=report_config['report_table'],
      test_info=test_info,
      system_info=system_info,
      extras=agg_result)


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

  Parse a results.txt file into an object.  Example of one line:
  repeatResNet50WinoSendRecv/20170310_204841_tensorflow_resnet50_32/8.txt:
  images/sec:391.0 +/- 1.6 (jitter = 8.0)

  Args:
    result_file_path: Path to file to parse

  Returns:
    `dict` representing the results.
  """
  result = {}
  result_file = open(result_file_path, 'r')
  for line in result_file:
    # Summary line starts with images/sec
    if line.find('total images/sec') == 0:
      result = {}
      parts = line.split(' ')
      result['imgs_sec'] = float(parts[2].rstrip())

      # Get the config
      result_dir = os.path.dirname(result_file_path)
      config_file = os.path.join(result_dir, 'config.yaml')
      f = open(config_file)
      config = yaml.safe_load(f)
      result['config'] = config
      result['result_dir'] = result_file_path
      result['test_id'] = config['test_id']

      if 'data_dir' in config:
        result['data_type'] = 'real'
      else:
        result['data_type'] = 'synth'

      # Number of gpus = number of servers * number of gpus
      if 'gpus' in config:
        result['gpu'] = int(config['gpus'])
      # Avoids files that might have multiple total lines in them.
      # First line found wins.
      break

  return result


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
