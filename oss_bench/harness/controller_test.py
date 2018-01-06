"""Tests test_controller module."""
from __future__ import print_function

import os
import sys
import unittest

import controller
from mock import patch
import yaml


class TestRunBenchmark(unittest.TestCase):
  """Tests for run_benchmark module."""

  @patch('tools.tf_version.get_tf_full_version')
  @patch('tools.nvidia.get_gpu_info')
  @patch('harness.controller.BenchmarkRunner._make_logs_dir')
  @patch('harness.controller.BenchmarkRunner._tf_cnn_bench')
  @patch('harness.controller.BenchmarkRunner._store_repo_info')
  @patch('harness.controller.BenchmarkRunner._clone_repos')
  def test_run_tests(self, clone_repos_mock, store_repo_info, tf_cnn_bench,
                     make_logs, gpu_info, get_tf_full_version):
    """Tests run_tests with most methods mocked."""
    benchmark_runner = controller.BenchmarkRunner(
        '/workspace', 'test_configs/basic_test_config.yaml')
    gpu_info.return_value = [387.11, 'GTX 970']

    expected_version = ['1.5RC0-dev20171001', 'v1.3.0-rc1-2884-g2d5b76169']
    get_tf_full_version.return_value = expected_version

    benchmark_runner.run_tests()

    clone_repos_mock.assert_called()
    make_logs.assert_called()
    store_repo_info.assert_called()
    self.assertEqual(os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
                     '/auth_tokens/tensorflow_performance_upload_tb.json')
    self.assertIn('/workspace/git/benchmark_harness/oss_bench', sys.path)

    arg0 = tf_cnn_bench.call_args[0][0]
    self.assertEqual(arg0['accel_type'], 'GTX 970')
    self.assertEqual(arg0['gpu_driver'], 387.11)
    self.assertEqual(arg0['framework_version'], expected_version[0])
    self.assertEqual(arg0['framework_describe'], expected_version[1])

  @patch('test_runners.tf_cnn_bench.run_benchmark.TestRunner')
  def test_tf_cnn_bench(self, test_runner_mock):
    """Tests call to run tf_cnn_bench tests."""
    # Config to pass directly to _tf_cnn_bench.
    config_file = ('harness/test_configs/basic_test_config.yaml')
    f = open(config_file)
    auto_config = yaml.safe_load(f)

    # Initializes class. test_config is not used in the test and set to None.
    benchmark_runner = controller.BenchmarkRunner('/workspace', None)

    # Verifies calls made to initialize TestRunner
    benchmark_runner._tf_cnn_bench(auto_config)
    call_args = test_runner_mock.call_args
    arg0 = call_args[0]
    self.assertEqual('{}{}'.format(
        '/workspace/git/', auto_config['tf_cnn_bench_configs'][0]), arg0[0])
    self.assertEqual('/workspace/logs/tf_cnn_workspace', arg0[1])
    self.assertEqual('/workspace/git/benchmarks/scripts/tf_cnn_benchmarks',
                     arg0[2])
    self.assertEqual(call_args[1]['auto_test_config'], auto_config)

  @patch('tools.git_info.git_repo_last_commit_id')
  @patch('tools.git_info.git_repo_describe')
  def test_store_repo_info(self, git_describe, git_last_commit_id):
    """Tests that repo info, e.g. last_commit_id, is stored in the config."""
    git_describe.return_value = 'bfd2f67'
    git_last_commit_id.return_value = 'bfd2f676fe5d21c66f30732648e58c4ca3f3d5b5'

    # Empty dict works fine as only values are added to the dict.
    config = {}

    # Initializes class. test_config is not used in the test and set to None.
    benchmark_runner = controller.BenchmarkRunner('/workspace', None)

    # Verifies calls made to initialize TestRunner
    benchmark_runner._store_repo_info(config)
    benchmarks_repo = config['git_repo_info']['benchmarks']

    self.assertEqual('bfd2f67', benchmarks_repo['describe'])
    self.assertEqual('bfd2f676fe5d21c66f30732648e58c4ca3f3d5b5',
                     benchmarks_repo['last_commit_id'])
