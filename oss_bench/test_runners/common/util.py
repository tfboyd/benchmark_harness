"""Utils for the test_runners."""

import numpy


def report_config_defaults(report_config, test_harness='unknown'):
  """Creates copy of report_config and sets defaults for missing entries."""
  config = report_config.copy()

  config['report_project'] = report_config.get(
      'report_project', 'google.com:tensorflow-performance')
  config['report_dataset'] = report_config.get('report_dataset',
                                               'benchmark_results_dev')
  config['report_table'] = report_config.get('report_table', 'result')
  # Details about where the test was run
  config['test_harness'] = report_config.get('test_harness', test_harness)
  config['test_environment'] = report_config.get('test_environment', 'unknown')
  config['platform'] = report_config.get('platform', 'unknown')
  config['platform_type'] = report_config.get('platform_type', 'unknown')
  config['accel_type'] = report_config.get('accel_type', 'unknown')

  return config


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
