"""Tests reporting module."""
from __future__ import print_function

import unittest

from mock import patch
import reporting


class TestReporting(unittest.TestCase):
  """Tests for reporting module."""

  @patch('test_runners.tf_cnn_bench.reporting._collect_results')
  @patch('test_runners.tf_cnn_bench.reporting.aggregate_results')
  @patch('upload.result_upload.upload_result')
  def test_process_folder(self, mock_upload, mock_aggregate_results, _):
    """Tests process folder and verifies args passed to upload_result."""
    agg_result = self._agg_result_example()
    # Adds extra field that should get piped through.
    extra_field = {}
    extra_field['foo'] = 485
    extra_field['bar'] = ['a', 'b']
    agg_result['test_extra_field'] = extra_field

    mock_aggregate_results.return_value = agg_result
    report_config = self._report_config_example()

    reporting.process_folder('/foo/folder', report_config=report_config)

    test_result = mock_upload.call_args[0][0]

    # Spot checks test_result.
    self.assertEqual(test_result['test_harness'], 'tf_cnn_benchmark')
    self.assertEqual(test_result['test_environment'],
                     report_config['test_environment'])
    self.assertEqual(test_result['test_id'], agg_result['config']['test_id'])

    # Spot checks results and GCE project info used for reporting.
    results = mock_upload.call_args[0][1]
    arg_report_project = mock_upload.call_args[0][2]
    arg_dataset = mock_upload.call_args[1]['dataset']
    arg_table = mock_upload.call_args[1]['table']
    self.assertEqual(results[0]['result'], agg_result['mean'])
    self.assertEqual(arg_report_project, 'google.com:tensorflow-performance')
    self.assertEqual(arg_dataset, 'benchmark_results_dev')
    self.assertEqual(arg_table, 'result')

    # Spot checks test_info.
    arg_test_info = mock_upload.call_args[1]['test_info']
    self.assertEqual(arg_test_info['framework_version'],
                     report_config['framework_version'])
    self.assertEqual(arg_test_info['framework_describe'],
                     report_config['framework_describe'])

    self.assertEqual(arg_test_info['git_info']['benchmarks']['describe'],
                     'a2384503f')

    # Spot checks system_info.
    arg_system_info = mock_upload.call_args[1]['system_info']
    self.assertEqual(arg_system_info['accel_type'], report_config['accel_type'])

    # Very spotty check of extras to confirm a random field is passed.
    arg_extras = mock_upload.call_args[1]['extras']
    self.assertEqual(arg_extras['test_extra_field'], extra_field)

  def _agg_result_example(self):
    """Returns a mocked up example aggregated result."""
    agg_result = {}
    agg_result['config'] = {}
    agg_result['config']['test_id'] = 'fake_test_id'
    agg_result['config']['model'] = 'resnet165'
    agg_result['config']['batch_size'] = 32
    agg_result['gpu'] = 5
    agg_result['mean'] = 58.3

    return agg_result

  def _report_config_example(self):
    """Returns a mocked up expected report_config with some values left out."""
    report_config = {}
    report_config['test_environment'] = 'unit_test_env'
    report_config['accel_type'] = 'GTX 940'
    report_config['platform'] = 'test_platform_name'
    report_config['framework_version'] = 'v1.5RC0-dev20171027'
    report_config['framework_describe'] = 'v1.3.0-rc1-2884-g2d5b76169'
    # git repo info with expected repo info.
    report_config['git_repo_info'] = {}
    # benchmark repo info
    git_info_dict = {}
    git_info_dict['describe'] = 'a2384503f'
    git_info_dict['last_commit_id'] = '267d7e81977f23998078f39afd48e9a97c3acf5a'
    report_config['git_repo_info']['benchmarks'] = git_info_dict

    return report_config
