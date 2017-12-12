import csv
import datetime
import json
from operator import itemgetter
import numpy
import os
import yaml
import StringIO
from upload import result_upload
from upload import result_info

# Local modules
import command_builder


def process_results_folder(folder_path, report_config={}):
  """Process and print aggregated results found in folder

  Args:
    folder_path: Folder to recursively search for results files, e.g.
    worker_0_stdout.log
  """
  results = []
  for r, d, f in os.walk(folder_path):
    for file in f:
      if file == 'worker_0_stdout.log':
        result_file = os.path.join(r, file)
        #print(result_file)
        results = results + parse_result_files(result_file)

  agg_result = aggregate_results(results)

  # Reporting project and dataset defaults are to the dev instance.
  report_project = report_config.get('report_project', 'google.com:tensorflow-performance')
  report_dataset = report_config.get('report_dataset', 'benchmark_results_dev')
  report_table =  report_config.get('report_table', 'result')
  # Details about where the test was run
  test_harness = report_config.get('test_harness', 'tf_cnn_benchmark')
  test_environment = report_config.get('test_environment', 'unknown')
  platform = report_config.get('platform', 'unknown')
  platform_type = report_config.get('platform_type', 'unknown')
  accel_type = report_config.get('accel_type','unknown')

  # Main result config
  test_result, results = result_info.build_test_result(agg_result['config']['test_id'],
                              agg_result['mean'],
                              result_type='exp_per_sec',
                              test_harness=test_harness,
                              test_environment=test_environment)
  
  test_info = result_info.build_test_info(batch_size=agg_result['config']['batch_size'],
                                   model=agg_result['config']['model'],
                                   accel_cnt=agg_result['gpu'])

  system_info = result_info.build_system_info(platform=platform,
                                       platform_type=platform_type,
                                       accel_type=accel_type)

  print 'Uploading test results.'
  result_upload.upload_result(test_result, results, report_project,
    dataset=report_dataset, table=report_table, test_info=test_info,
    system_info=system_info, extras=agg_result)


def parse_result_files(result_file_path):
  """Parses a result file


  Parse a results.txt file into an object.  Example of one line:
  repeatResNet50WinoSendRecv/20170310_204841_tensorflow_resnet50_32/8.txt:images/sec:
  391.0 +/- 1.6 (jitter = 8.0)

  Args:
    result_file_path: Path to file to parse

  """

  results = []

  result_file = open(result_file_path, 'r')
  for line in result_file:
    # Summary line starts with images/sec
    if line.find('total images/sec') == 0:
      result = {}
      parts = line.split(' ')
      result['imgs_sec'] = parts[2].rstrip()

      # Get the config
      result_dir = os.path.dirname(result_file_path)
      config_file = os.path.join(result_dir, 'config.yaml')
      f = open(config_file)
      config = yaml.safe_load(f)
      result['config'] = config
      result['result_dir'] = result_file_path

      if 'ps_servers' in config:
        total = len(command_builder.WorkerUtil(config['ps_servers']).split(','))
        result['ps_servers'] = str(total)
      else:
        result['ps_servers'] = 'N/A'

      if 'data_dir' in config:
        result['data_type'] = 'real'
      else:
        result['data_type'] = 'synth'

      # Number of servers = number of workers
      servers = len(command_builder.WorkerUtil(config['workers']).split(','))
      result['servers'] = servers
      # Number of gpus = number of servers * number of gpus
      if 'gpus' in config:
        result['gpu'] = servers * int(config['gpus'])

      results.append(result)
      # Avoids files that might have multiple total lines in them.
      # First line found wins.
      break

  return results


def aggregate_results(results_list):
  """Aggregates the results via a methodology to get a final answer

  Aggregates results but make sure it is not grouping different results
  together.
  Supports grouping models and gpus.  Just group things in similar logs folders
  to make aggregation work.

  Note:  If 10 or more, drop highest and lowest then average.  If less than 10
  just the
  average.

  Does not support:
     - aggregate models using different batch sizes

  Args:
    results_list: List of results that can be a mix of models and gpus
    report_type: Type of report which determines sorting and grouping

  """

  # test_id.results[result,result]
  # test_id.aggs[1,2.3,1.25]
  # test_id.mean
  # test_id.std
  # test_id.max

  # TODO(tobyboyd@): Adapted from code the handled aggregating multiple
  # tests.  In this usage only one test is being aggreagted, finding two
  # different test_ids in the results would be an error.

  agg_stage = {}

  # Groups the results to then be aggregated
  for result in results_list:
    model_gpu_result = agg_stage.setdefault(result['config']['test_id'], {})
    # Add the raw result to an array with other results from the test_id
    model_gpu_raw_results = model_gpu_result.setdefault('results', [])
    model_gpu_raw_results.append(result)

    # Adds the total to list to aggregate later
    model_gpu_agg = model_gpu_result.setdefault('aggs', [])
    model_gpu_agg.append(result['imgs_sec'])

  agg_results = []
  for key, test_id_result in agg_stage.iteritems():
    # Gets details of first result, assumes all details are the same
    agg_result = test_id_result['results'][0].copy()
    result_list = [float(i) for i in test_id_result['aggs']]
    result_list.sort()
    agg_result['samples'] = len(result_list)
    agg_result['mean'] = numpy.mean(result_list)
    agg_result['std'] = numpy.std(result_list)
    agg_result['max'] = result_list[-1]
    agg_result['min'] = result_list[0]

    agg_results.append(agg_result)

  # See TODO on only processing one result, for now throwing error if two 
  # results are found.

  if len(agg_results) > 1:
    raise Exception('Only one result (test_id) should exists after aggregation')
  return agg_results[0]


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

