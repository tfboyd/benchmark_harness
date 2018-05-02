"""Tests runner.py module."""
from __future__ import print_function

import unittest

from mock import patch
import runner


class TestRunBenchmark(unittest.TestCase):
  """Tests for runner.py module."""

  @patch('test_runners.tf_models.runner.TestRunner.run_test_suite')
  @patch('test_runners.tf_models.runner.TestRunner._make_log_dir')
  def test_resnet50_128_gpu_8_fp16(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    run.resnet50_128_gpu_8_fp16()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 8)
    self.assertEqual(test_config['batch_size'], 128)
    self.assertEqual(test_config['args']['batch_size'], 1024)
    self.assertEqual(test_config['args']['dtype'], 'fp16')
    self.assertEqual(test_config['args']['use_synthetic_data'], '')
    self.assertEqual(test_config['test_id'], 'garden.resnet50.gpu_8.128.fp16')
    self.assertIn('model', test_config)

  @patch('test_runners.tf_models.runner.TestRunner.run_test_suite')
  @patch('test_runners.tf_models.runner.TestRunner._make_log_dir')
  def test_resnet50v2_64_gpu_1_real(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    run.resnet50v2_64_gpu_1_real()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 1)
    self.assertEqual(test_config['batch_size'], 64)
    self.assertEqual(test_config['args']['data_dir'], '/data/imagenet')
    self.assertEqual(test_config['args']['batch_size'], 64)
    self.assertEqual(test_config['args']['version'], 2)
    self.assertEqual(test_config['args']['dtype'], 'fp32')
    self.assertEqual(test_config['test_id'], 'garden.resnet50v2.gpu_1.64.real')
    self.assertIn('model', test_config)
