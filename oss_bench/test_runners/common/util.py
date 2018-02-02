"""Utils for the test_runners."""
from __future__ import print_function
import numpy
from upload import result_info
from upload import result_upload


def report_config_defaults(report_config, test_harness=None):
  """Creates copy of report_config and sets defaults for missing entries."""
  config = report_config.copy()

  config['report_project'] = report_config.get(
      'report_project', 'google.com:tensorflow-performance')
  config['report_dataset'] = report_config.get('report_dataset',
                                               'benchmark_results_dev')
  config['report_table'] = report_config.get('report_table', 'result')
  # Details about where the test was run
  if test_harness:
    config['test_harness'] = test_harness
  else:
    config['test_harness'] = report_config.get('test_harness', 'unknown')
  config['test_environment'] = report_config.get('test_environment', 'unknown')
  config['platform'] = report_config.get('platform', 'unknown')
  config['platform_type'] = report_config.get('platform_type', 'unknown')
  config['accel_type'] = report_config.get('accel_type', 'unknown')
  config['accel_type'] = report_config.get('accel_type', 'unknown')
  config['framework_describe'] = report_config.get('framework_describe',
                                                   'unknown')
  return config


def upload_results(report_config, agg_result, framework=None,
                   test_harness=None):
  """Upload results of the test.

  Args:

    report_config: config for the tests
    agg_result: results of the test
    framework: Name of the framework being tested
    test_harness: Name of the test harness
  """
  assert framework
  assert test_harness

  report_config = report_config_defaults(
      report_config, test_harness=test_harness)

  # Main result config
  test_result, results = result_info.build_test_result(
      agg_result['config']['test_id'],
      agg_result['mean'],
      result_type='exp_per_sec',
      test_harness=report_config['test_harness'],
      test_environment=report_config['test_environment'])

  test_info = result_info.build_test_info(
      framework=framework,
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

  print('Uploading test results...')
  result_upload.upload_result(
      test_result,
      results,
      report_config['report_project'],
      dataset=report_config['report_dataset'],
      table=report_config['report_table'],
      test_info=test_info,
      system_info=system_info,
      extras=agg_result)


def report_aggregate_results(results_list):
  """Aggregates the results in the list.

  Args:
    results_list: List of results to aggregate.

  Returns:
    dict summarizing the results in the list.
  """

  agg_result = {}
  # Assumes first entry has same test_id as the rest of the results.
  agg_result['test_id'] = results_list[0]['test_id']
  agg_result = results_list[0].copy()

  # Groups the results to then be aggregated
  results = []
  for result in results_list:
    # Adds the total to list to aggregate later
    results.append(result['imgs_sec'])

  results.sort()
  agg_result['samples'] = len(results)
  agg_result['mean'] = numpy.mean(results)
  agg_result['std'] = numpy.std(results)
  agg_result['max'] = results[-1]
  agg_result['min'] = results[0]

  return agg_result
