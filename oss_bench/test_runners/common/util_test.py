"""Tests util.py"""
from __future__ import print_function

import unittest

import util


class TestReporting(unittest.TestCase):
  """Tests for mxnet reporting module."""

  def test_aggregate_results(self):
    """Tests aggregating multiple results file from the same test."""
    test_id_0 = 'made.up.test_id'
    results_list = []
    results_list.append(self._mock_result(test_id_0, 10.5))
    results_list.append(self._mock_result(test_id_0, 20))
    results_list.append(self._mock_result(test_id_0, .44444))

    agg_result = util.report_aggregate_results(results_list)
    self.assertEqual(agg_result['test_id'], test_id_0)
    self.assertEqual(agg_result['mean'], 10.314813333333333)
    self.assertEqual(agg_result['max'], 20)
    self.assertEqual(agg_result['min'], 0.44444)
    self.assertEqual(agg_result['std'], 7.9845977692276744)
    self.assertEqual(agg_result['samples'], 3)
    self.assertIn('config', agg_result)

  def _mock_config(self, test_id):
    config = {}
    config['test_id'] = test_id
    config['batch_size'] = 128
    config['gpus'] = 2
    config['model'] = 'resnet50'
    return config

  def _mock_result(self, test_id, imgs_sec):
    result = {}
    result['config'] = self._mock_config(test_id)
    result['imgs_sec'] = imgs_sec
    result['test_id'] = test_id
    result['gpu'] = 2
    result['batches_sampled'] = 100
    return result
