"""Tests runner.py module."""
from __future__ import print_function

import unittest

from mock import patch
import test_runners.mxnet.runner as runner


class TestRunBenchmark(unittest.TestCase):
  """Tests for runner.py module."""

  @patch('test_runners.mxnet.runner.TestRunner.run_test_suite')
  @patch('test_runners.mxnet.runner.TestRunner._make_log_dir')
  def test_1gpu_fp32(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu test."""
    run = runner.TestRunner('/workspace', '/workspace/git/mxnet_git')
    run.renset50v1_32_gpu_1()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 1)
    self.assertEqual(test_config['batch_size'], 32)
    self.assertEqual(test_config['args']['batch-size'], 32)
    self.assertEqual(test_config['args']['dtype'], 'float32')
    self.assertEqual(test_config['args']['gpus'], '0')
    self.assertEqual(test_config['test_id'], 'resnet50v1.gpu_1.32')
    self.assertIn('model', test_config)

  @patch('test_runners.mxnet.runner.TestRunner.run_test_suite')
  @patch('test_runners.mxnet.runner.TestRunner._make_log_dir')
  def test_8gpu_fp16_batch128(self, _, run_test_suite_mock):
    """Tests running a multi-gpu test and overriding the dtype."""
    run = runner.TestRunner('/workspace', '/workspace/git/mxnet_git')
    run.renset50v1_128_gpu_8_fp16()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 8)
    self.assertEqual(test_config['batch_size'], 128)
    self.assertEqual(test_config['args']['dtype'], 'float16')
    self.assertEqual(test_config['args']['gpus'], '0,1,2,3,4,5,6,7')
    self.assertEqual(test_config['args']['batch-size'], 128 * 8)
    self.assertEqual(test_config['test_id'], 'resnet50v1.gpu_8.128.fp16')

  @patch('test_runners.mxnet.runner.TestRunner.run_test_suite')
  @patch('test_runners.mxnet.runner.TestRunner._make_log_dir')
  def test_1gpu_fp32_real(self, _, run_test_suite_mock):
    """Tests init TestRunner and running a mocked single gpu real data test."""
    run = runner.TestRunner(
        '/workspace',
        '/workspace/git/mxnet_git',
        imagenet_dir='/data/mxnet/imagenet')
    run.renset50v1_32_gpu_1_real()
    test_config = run_test_suite_mock.call_args[0][0]

    # check GPU args and count
    self.assertEqual(test_config['gpus'], 1)
    self.assertEqual(test_config['batch_size'], 32)
    self.assertEqual(test_config['args']['batch-size'], 32)
    self.assertEqual(test_config['args']['data-train'], '/data/mxnet/imagenet')
    self.assertEqual(test_config['args']['dtype'], 'float32')
    self.assertEqual(test_config['args']['gpus'], '0')
    self.assertEqual(test_config['test_id'], 'resnet50v1.gpu_1.32.real')
    self.assertIn('model', test_config)

  @patch('test_runners.mxnet.runner.TestRunner.run_test_suite')
  @patch('test_runners.mxnet.runner.TestRunner._make_log_dir')
  def test_run_test_2(self, _, run_test_suite_mock):
    """Tests running list of tests by name and counts how many are run."""
    run = runner.TestRunner('/workspace', '/workspace/git/mxnet_git')
    tests = ['renset50v1_128_gpu_8_fp16', 'renset50v1_128_gpu_8']
    run.run_tests(tests)
    tests_run = run_test_suite_mock.call_count
    self.assertEqual(tests_run, len(tests))

  @patch('test_runners.mxnet.runner.TestRunner._make_log_dir')
  def test_base_imagenet_args(self, _):
    """Tests base imagenet_args are generated as expected."""
    test_config = {}
    test_config['data_threads'] = 16

    run = runner.TestRunner(
        '/workspace',
        '/workspace/git/mxnet_git',
        auto_test_config=test_config,
        imagenet_dir='/data/mxnet/imagenet/data')

    args = run._base_imagenet_args()

    self.assertEqual(args['data-train'], '/data/mxnet/imagenet/data')
    self.assertEqual(args['data-nthreads'], 16)

  @patch('test_runners.mxnet.runner.TestRunner._make_log_dir')
  def test_base_imagenet_args_no_data(self, _):
    """Tests base imagenet_args throws an assertion error."""
    run = runner.TestRunner(
        '/workspace', '/workspace/git/mxnet_git', imagenet_dir=None)
    self.assertRaises(AssertionError, run._base_imagenet_args)
