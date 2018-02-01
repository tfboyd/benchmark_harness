"""Tests runner.py module."""
from __future__ import print_function

import unittest

from mock import patch
import runner


class TestRunBenchmark(unittest.TestCase):
  """Tests for runner.py module."""

  @patch('test_runners.tf_models.runner.TestRunner.run_test_suite')
  @patch('test_runners.tf_models.runner.TestRunner._make_log_dir')
  def test_1gpu_fp32(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    run.renset50v2_32_gpu_1()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 1)
    self.assertEqual(test_config['batch_size'], 32)
    self.assertEqual(test_config['args']['batch_size'], 32)
    self.assertEqual(test_config['test_id'], 'garden.resnet50v2.gpu_1.32')
    self.assertIn('model', test_config)
