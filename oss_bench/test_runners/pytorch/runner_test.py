"""Tests runner.py module."""
from __future__ import print_function

import unittest

from mock import patch
import runner


class TestRunBenchmark(unittest.TestCase):
  """Tests for runner.py module."""

  @patch('test_runners.pytorch.runner.TestRunner.run_test_suite')
  @patch('test_runners.pytorch.runner.TestRunner._make_log_dir')
  def test_resnet_1gpu_fp32(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    run.renset50_32_gpu_1_real()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 1)
    self.assertEqual(test_config['batch_size'], 32)
    self.assertEqual(test_config['args']['batch-size'], 32)
    self.assertEqual(test_config['test_id'], 'resnet50.gpu_1.32.real')
    self.assertIn('model', test_config)
    print(test_config['pycmd'])

  @patch('test_runners.pytorch.runner.TestRunner.run_test_suite')
  @patch('test_runners.pytorch.runner.TestRunner._make_log_dir')
  def test_resnet_1gpu_fp32_cmd_builder(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    run.renset50_32_gpu_1_real()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 1)
    self.assertEqual(test_config['batch_size'], 32)
    self.assertEqual(test_config['args']['batch-size'], 32)
    self.assertEqual(test_config['test_id'], 'resnet50.gpu_1.32.real')
    self.assertIn('model', test_config)
    cmd = run._cmd_builder(test_config)
    # Verifies the full command.
    self.assertEqual(cmd, 'CUDA_VISIBLE_DEVICES=0 python main.py '
                     '--arch resnet50 --batch-size 32 --epochs 1 '
                     '--print-freq 1 --workers 5  /data/pytorch/imagenet')

  @patch('test_runners.pytorch.runner.TestRunner.run_test_suite')
  @patch('test_runners.pytorch.runner.TestRunner._make_log_dir')
  def test_resnet_8gpu_fp32_cmd_builder(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    run.renset50_64_gpu_8_real()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 8)
    self.assertEqual(test_config['batch_size'], 64)
    self.assertEqual(test_config['args']['batch-size'], 8 * 64)
    self.assertEqual(test_config['test_id'], 'resnet50.gpu_8.64.real')
    self.assertIn('model', test_config)
    cmd = run._cmd_builder(test_config)
    # Verifies the full command.
    self.assertEqual(cmd, 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py '
                     '--arch resnet50 --batch-size 512 --epochs 1 '
                     '--print-freq 1 --workers 5  /data/pytorch/imagenet')
