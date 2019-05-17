"""Tests pytorch reporting module."""
from __future__ import print_function

import unittest

from mock import patch

import test_runners.pytorch.reporting as reporting


class TestReporting(unittest.TestCase):
  """Tests for pytorch reporting module."""

  @patch('upload.result_upload.upload_result')
  def test_process_folder(self, mock_upload):
    """Tests process folder and verifies args passed to upload_result."""
    report_config = self._report_config_example()
    reporting.process_folder(
        'test_runners/pytorch/unittest_files/results/basic',
        report_config=report_config)

    test_result = mock_upload.call_args[0][0]

    # Spot checks test_result.
    self.assertEqual(test_result['test_harness'], 'pytorch')
    self.assertEqual(test_result['test_environment'],
                     report_config['test_environment'])
    self.assertEqual(test_result['test_id'], 'resnet50.gpu_1.32.real')

    # Spot checks results and GCE project info used for reporting.
    results = mock_upload.call_args[0][1]
    self.assertEqual(results[0]['result'], 177.07878258336976)

    # Spot checks test_info.
    arg_test_info = mock_upload.call_args[1]['test_info']
    self.assertEqual(arg_test_info['accel_cnt'], 2)
    self.assertEqual(arg_test_info['cmd'], 'python blahblah.py -arg foo')

    # Very spotty check of extras to confirm a random field is passed.
    arg_extras = mock_upload.call_args[1]['extras']
    # Checks that the config is saved in the extras field.
    self.assertIn('config', arg_extras)
    self.assertIn('batches_sampled', arg_extras)

  def test_parse_result_file(self):
    """Tests parsing one results file."""
    result = reporting.parse_result_file(
        'test_runners/pytorch/unittest_files/results/basic/'
        'worker_0_stdout.txt')

    self.assertEqual(result['imgs_sec'], 177.07878258336976)
    self.assertEqual(result['batches_sampled'], 19)
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
    cpu_info = {}
    cpu_info['model_name'] = 'Intel XEON 2600E 2.8Ghz'
    cpu_info['core_count'] = 36
    cpu_info['socket_count'] = 1
    report_config['cpu_info'] = cpu_info
    return report_config
