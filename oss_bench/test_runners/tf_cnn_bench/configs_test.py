"""Tests the configs to reduce issues during startup."""

from __future__ import print_function

import os
import unittest

from mock import patch
import run_benchmark


class TestRunConfigs(unittest.TestCase):
  """Tests for syntax and paths of config files."""

  @patch('test_runners.tf_cnn_bench.run_benchmark.TestRunner._make_log_dir')
  def test_all_configs(self, _):
    """Tests that configs and sub_configs point to files that exist."""
    base_dir = 'test_runners/tf_cnn_bench/configs'

    relative_configs = [
        'dgx_v100.yaml', 'dgx_v100_real.yaml', 'dgx_p100.yaml',
        'dgx_p100_real.yaml', 'dgx_io_test.yaml', 'dgx_io_test_real.yaml',
        'io_warmup_real.yaml', 'v100_training_resnetv1.yaml',
        'v100_training_resnetv1.yaml', 'tf_org_benchmark/aws_k80.yaml',
        'tf_org_benchmark/dgx_p100.yaml', 'tf_org_benchmark/dgx_v100.yaml',
        'tf_org_benchmark/aws_k80_real.yaml',
        'tf_org_benchmark/dgx_p100_real.yaml',
        'tf_org_benchmark/dgx_v100_real.yaml'
    ]

    for relative_config in relative_configs:
      config = os.path.join(base_dir, relative_config)
      print('Testing config {}.'.format(config))
      self.check_config_paths(config)

  def check_config_paths(self, config):
    """Throws an exception if main config or sub config paths are incorrect."""
    test_runner = run_benchmark.TestRunner(None, '/workspace', 'bench_home')
    config_obj = test_runner.load_yaml_configs([config])[0]
    base_dir = os.path.dirname(config_obj['config_path'])
    test_runner.load_yaml_configs(config_obj['sub_configs'], base_dir=base_dir)
