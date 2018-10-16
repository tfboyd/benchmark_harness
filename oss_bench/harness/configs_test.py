"""Tests that configs are correct."""
from __future__ import print_function

import unittest
from mock import patch
import test_runners.tf_models.runner as runner
import yaml


class TestHarnessConfigs(unittest.TestCase):
  """Tests for syntax and paths of config files."""

  @patch('test_runners.tf_models.runner.TestRunner._make_log_dir')
  def test_tf_models_methods(self, _):
    """Checks that the tests (methods) exist."""
    configs = [
        'harness/configs/prod/dgx2_v100.yaml',
        'harness/configs/dev/default.yaml',
        'harness/configs/prod/dgx_v100.yaml',
        'harness/configs/prod/dgx_v100.yaml',
        'harness/configs/prod/dgx_p100.yaml',
        'harness/configs/prod/dgx_p100_RC.yaml',
        'harness/configs/prod/dgx_p100_FINAL.yaml',
        'harness/configs/prod/dgx_v100_RC.yaml',
        'harness/configs/prod/dgx_v100_FINAL.yaml',
        'harness/configs/prod/aws_v100.yaml',
        'harness/configs/prod/gce_v100.yaml',
        'harness/configs/prod/gce_v100_xla.yaml'

    ]
    for config in configs:
      self.check_config_paths(config)

  def check_config_paths(self, config):
    """Asserts that the methods in the file for tf_models exist."""
    f = open(config)
    auto_config = yaml.safe_load(f)
    run = runner.TestRunner('/workspace', '/workspace/git/tf_models')
    if 'tf_models_tests' in auto_config:
      for test_method in auto_config['tf_models_tests']:
        ret = getattr(run, test_method, None)
        self.assertIsNotNone(
            ret,
            msg='Method "{}" not found from file "{}"'.format(
                test_method, config))
