"""Tests command_builder module."""
from __future__ import print_function

import unittest
import yaml

import command_builder


class TestRCommandBuilder(unittest.TestCase):
  """Tests for reporting module."""

  def test_load_yaml_run_config_real_data(self):
    """Tests building configs with data_dir and test_id_hacks."""
    config_file = (
        'test_runners/tf_cnn_bench/test_configs/expected_full_config.yaml')
    with open(config_file) as f:
      full_config = yaml.safe_load(f)
    full_config['data_dir'] = '/data/imagenet'
    full_config['test_id_hacks'] = True
    test_suite = command_builder.build_test_config_suite(full_config, 0)

    self.assertEqual(len(test_suite[0]), 3)
    actual_first_config = test_suite[0][0]
    actual_second_config = test_suite[0][1]
    actual_first_config.pop('test_id_hacks')
    actual_second_config.pop('test_id_hacks')
    expected_file = ('test_runners/tf_cnn_bench/test_configs/'
                     'expected_post_config_builder.yaml')
    with open(expected_file) as f:
      expected_config = yaml.safe_load(f)
    self.assertEqual(actual_first_config, expected_config)
    expected_config_2 = expected_config.copy()
    expected_config_2['copy'] = 1
    self.assertEqual(actual_second_config, expected_config_2)

  def test_load_yaml_run_config(self):
    """Tests building a config without test_id_hacks."""
    config_file = (
        'test_runners/tf_cnn_bench/test_configs/expected_full_config.yaml')
    with open(config_file) as f:
      full_config = yaml.safe_load(f)
    test_suite = command_builder.build_test_config_suite(full_config, 0)

    self.assertEqual(len(test_suite[0]), 3)
    actual_first_config = test_suite[0][0]
    expected_file = ('test_runners/tf_cnn_bench/test_configs/'
                     'expected_post_config_builder.yaml')
    with open(expected_file) as f:
      expected_config = yaml.safe_load(f)
    expected_config['test_id'] = 'resnet50.1_gpu.64.ps_cpu'
    expected_config.pop('data_dir')
    self.assertEqual(actual_first_config, expected_config)
