"""Tests mxnet reporting module."""
from __future__ import print_function

import unittest

from mock import patch
import reporting


class TestReporting(unittest.TestCase):
  """Tests for mxnet reporting module."""

  @patch('test_runners.mxnet.reporting._collect_results')
  @patch('upload.result_upload.upload_result')
  def test_process_folder(self, mock_upload, mock_collect_results):
    """Tests process folder and verifies args passed to upload_result."""
    # Results to process
    test_id_0 = 'made.up.test_id'
    results_list = []
    results_list.append(self._mock_result(test_id_0, 10.5))
    results_list.append(self._mock_result(test_id_0, 20))
    results_list.append(self._mock_result(test_id_0, .44444))
    # Sets results_list to be returned from mock method.
    mock_collect_results.return_value = results_list

    report_config = self._report_config_example()

    reporting.process_folder('/foo/folder', report_config=report_config)

    test_result = mock_upload.call_args[0][0]

    # Spot checks test_result.
    self.assertEqual(test_result['test_harness'], 'mxnet')
    self.assertEqual(test_result['test_environment'],
                     report_config['test_environment'])
    self.assertEqual(test_result['test_id'], test_id_0)

    # Spot checks results and GCE project info used for reporting.
    results = mock_upload.call_args[0][1]
    arg_report_project = mock_upload.call_args[0][2]
    arg_dataset = mock_upload.call_args[1]['dataset']
    arg_table = mock_upload.call_args[1]['table']
    self.assertEqual(results[0]['result'], 10.314813333333333)
    self.assertEqual(arg_report_project, 'google.com:tensorflow-performance')
    self.assertEqual(arg_dataset, 'benchmark_results_dev')
    self.assertEqual(arg_table, 'result')

    # Spot checks test_info.
    arg_test_info = mock_upload.call_args[1]['test_info']
    self.assertEqual(arg_test_info['framework_version'],
                     report_config['framework_version'])
    self.assertEqual(arg_test_info['accel_cnt'], 2)
    self.assertEqual(arg_test_info['cmd'],
                     'python some_script.py --arg0=foo --arg1=bar')
    # Spot checks system_info.
    arg_system_info = mock_upload.call_args[1]['system_info']
    self.assertEqual(arg_system_info['accel_type'], report_config['accel_type'])

    # Very spotty check of extras to confirm a random field is passed.
    arg_extras = mock_upload.call_args[1]['extras']
    # Checks that the config is saved in the extras field.
    self.assertIn('config', arg_extras)
    self.assertIn('batches_sampled', arg_extras)

  def test_parse_result_file(self):
    """Tests parsing one results file."""
    result = reporting.parse_result_file(
        'test_runners/mxnet/unittest_files/basic_synth/test_result.txt')

    self.assertEqual(result['imgs_sec'], 178.55300000000003)
    self.assertEqual(result['batches_sampled'], 10)
    self.assertEqual(result['test_id'], 'resnet50.gpu_1.32.real')
    self.assertEqual(result['gpu'], 2)
    self.assertEqual(result['data_type'], 'synth')
    self.assertIn('config', result)

  def test_parse_result_file_real(self):
    """Tests parsing one results file for real data."""
    result = reporting.parse_result_file(
        'test_runners/mxnet/unittest_files/basic_real/test_result.txt')

    self.assertEqual(result['imgs_sec'], 178.55300000000003)
    self.assertEqual(result['batches_sampled'], 10)
    self.assertEqual(result['test_id'], 'resnet50.gpu_1.32.real')
    self.assertEqual(result['gpu'], 2)
    self.assertEqual(result['data_type'], 'real')
    self.assertIn('config', result)

  def _mock_config(self, test_id):
    config = {}
    config['test_id'] = test_id
    config['batch_size'] = 128
    config['gpus'] = 2
    config['model'] = 'resnet50'
    config['cmd'] = 'python some_script.py --arg0=foo --arg1=bar'
    return config

  def _mock_result(self, test_id, imgs_sec):
    result = {}
    result['config'] = self._mock_config(test_id)
    result['imgs_sec'] = imgs_sec
    result['test_id'] = test_id
    result['gpu'] = 2
    result['batches_sampled'] = 100
    return result

  def _report_config_example(self):
    """Returns a mocked up expected report_config with some values left out."""
    report_config = {}
    report_config['test_environment'] = 'unit_test_env'
    report_config['accel_type'] = 'GTX 940'
    report_config['platform'] = 'test_platform_name'
    report_config['framework_version'] = 'v1.5RC0-dev20171027'
    return report_config
