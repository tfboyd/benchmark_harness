"""Tests run_benchmark module."""
from __future__ import print_function

import unittest

from mock import patch
import run_benchmark
import yaml


class TestRunBenchmark(unittest.TestCase):
  """Tests for run_benchmark module."""

  @patch('test_runners.tf_cnn_bench.run_benchmark.TestRunner.run_test_suite')
  @patch('test_runners.tf_cnn_bench.run_benchmark.TestRunner._make_log_dir')
  def test_run_tests(self, make_log_dir_mock, run_test_suite):
    """Tests init TestRunner and run_tests with most methods mocked."""
    expected_file = ('test_runners/tf_cnn_bench/test_configs/'
                     'expected_full_config.yaml')
    f = open(expected_file)
    expected_full_config = yaml.safe_load(f)
    config = 'test_runners/tf_cnn_bench/test_configs/basic_run_config.yaml'
    test_runner = run_benchmark.TestRunner(config, '/workspace', 'bench_home')
    test_runner.run_tests()
    make_log_dir_mock.assert_called_with('/workspace/logs')
    # Verifies the config file is processed and passed to run_test_suite.
    run_test_suite.assert_called_with(expected_full_config)

  @patch('test_runners.tf_cnn_bench.run_benchmark.TestRunner._make_log_dir')
  @patch('test_runners.tf_cnn_bench.run_benchmark.reporting.process_folder')
  @patch('test_runners.tf_cnn_bench.run_benchmark.TestRunner.run_benchmark')
  def test_run_test_suite(self, run_benchmark_mock, reporting_mock, _):
    """Tests running run_test_suite with run_benchmark mocked."""
    config_file = (
        'test_runners/tf_cnn_bench/test_configs/expected_full_config.yaml')
    f = open(config_file)
    full_config = yaml.safe_load(f)
    test_runner = run_benchmark.TestRunner(None, '/workspace', 'bench_home')
    test_runner.run_test_suite(full_config)
    # Verifies run_benchmark called three times and reporting once. The
    # full_config contains one test run 3 times.
    self.assertEqual(run_benchmark_mock.call_count, 3)
    self.assertEqual(reporting_mock.call_count, 1)

    # Spot checks the args passed to run_benchmark the last time it is called.
    last_run_benchmark_arg0 = run_benchmark_mock.call_args[0][0]
    last_run_benchmark_arg1 = run_benchmark_mock.call_args[0][1]
    self.assertEqual(last_run_benchmark_arg0['copy'], 2)
    self.assertIn('test_suite_start_time', last_run_benchmark_arg0)
    self.assertTrue(last_run_benchmark_arg0['model'], 'resnet50')
    self.assertTrue(last_run_benchmark_arg0['batch_size'], 64)
    self.assertIsInstance(last_run_benchmark_arg1,
                          run_benchmark.cluster_local.LocalInstance)
