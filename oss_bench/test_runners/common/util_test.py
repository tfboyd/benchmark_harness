"""Tests util.py."""
from __future__ import print_function

import unittest
from mock import patch
import util


class TestReporting(unittest.TestCase):
  """Tests for reporting module."""

  def test_aggregate_results(self):
    """Tests aggregating multiple results file from the same test."""
    test_id_0 = 'made.up.test_id'
    results_list = []
    extra_results = []
    extra_results.append(self._mock_extra_result(250, 'total_time', 'ms'))
    extra_results.append(self._mock_extra_result(250, 'total_time', 'ms'))
    extra_results_2 = []
    extra_results_2.append(self._mock_extra_result(125, 'total_time', 'ms'))
    extra_results_2.append(self._mock_extra_result(125, 'total_time', 'ms'))
    results_list.append(self._mock_result(test_id_0,
                                          10.5,
                                          extra_results=extra_results))
    results_list.append(self._mock_result(test_id_0,
                                          20,
                                          extra_results=extra_results))
    results_list.append(self._mock_result(test_id_0,
                                          .44444,
                                          extra_results=extra_results_2))

    agg_result = util.report_aggregate_results(results_list)
    self.assertEqual(agg_result['test_id'], test_id_0)
    self.assertEqual(agg_result['mean'], 10.314813333333333)
    self.assertEqual(agg_result['max'], 20)
    self.assertEqual(agg_result['min'], 0.44444)
    self.assertEqual(agg_result['std'], 7.9845977692276744)
    self.assertEqual(agg_result['samples'], 3)
    self.assertIn('config', agg_result)
    # Check for extra results
    self.assertEqual(agg_result['extra_results'][0]['mean'], 208.33333333333334)

  def test_aggregate_results_only_1(self):
    """Tests aggregating 1 result."""
    test_id_0 = 'made.up.test_id'
    results_list = []
    results_list.append(self._mock_result(test_id_0, 10.5))

    agg_result = util.report_aggregate_results(results_list)
    self.assertEqual(agg_result['test_id'], test_id_0)
    self.assertEqual(agg_result['mean'], 10.5)
    self.assertEqual(agg_result['max'], 10.5)
    self.assertEqual(agg_result['min'], 10.5)
    self.assertEqual(agg_result['std'], 0)
    self.assertEqual(agg_result['samples'], 1)
    self.assertIn('config', agg_result)

  def test_aggregate_results_no_imgs_sec(self):
    """Tests aggregating results that are empty."""
    test_id_0 = 'made.up.test_id'
    results_list = []
    results_list.append(self._mock_result(test_id_0, None))
    results_list.append(self._mock_result(test_id_0, None))
    results_list.append(self._mock_result(test_id_0, None))

    agg_result = util.report_aggregate_results(results_list)
    self.assertEqual(agg_result['test_id'], test_id_0)
    self.assertEqual(agg_result['mean'], 0)
    self.assertEqual(agg_result['max'], 0)
    self.assertEqual(agg_result['min'], 0)
    self.assertEqual(agg_result['std'], 0)
    self.assertEqual(agg_result['samples'], 0)
    self.assertIn('config', agg_result)

  @patch('upload.result_upload.upload_result')
  def test_upload_results(self, mock_upload_results):
    """Tests upload results with spot checking."""

    report_config = {}
    cpu_info = {}
    cpu_info['model_name'] = 'Intel XEON 2600E 2.8Ghz'
    cpu_info['core_count'] = 36
    cpu_info['socket_count'] = 1
    report_config['cpu_info'] = cpu_info
    report_config['channel'] = 'RC'
    report_config['build_type'] = 'OTB-GPU'
    report_config['framework_version'] = '1.4.0_20170124'

    agg_result = {}
    agg_result['mean'] = 99.25
    agg_result['gpu'] = 4
    agg_result['config'] = {}
    agg_result['config']['test_id'] = 'fake_test_id'
    agg_result['config']['batch_size'] = 32
    agg_result['config']['model'] = 'resnet50'
    agg_result['config']['cmd'] = 'python tf_cnn_benchmark foo=0'

    util.report_config_defaults(report_config)
    util.upload_results(
        report_config,
        agg_result=agg_result,
        framework='tensorflow',
        test_harness='tf_model')

    # Spot checks test_info.
    arg_test_info = mock_upload_results.call_args[1]['test_info']
    self.assertEqual(arg_test_info['framework'], 'tensorflow')
    self.assertEqual(arg_test_info['channel'], report_config['channel'])
    self.assertEqual(arg_test_info['build_type'], report_config['build_type'])
    self.assertEqual(arg_test_info['batch_size'],
                     agg_result['config']['batch_size'])
    self.assertEqual(arg_test_info['accel_cnt'], agg_result['gpu'])
    self.assertEqual(arg_test_info['cmd'], agg_result['config']['cmd'])

  def test_aggregate_extra_results(self):
    """Tests aggregating extra results."""
    total_times = []
    top_1s = []
    total_times.append(self._mock_extra_result(250, 'total_time', 'ms'))
    total_times.append(self._mock_extra_result(100, 'total_time', 'ms'))
    top_1s.append(self._mock_extra_result(.55, 'top_1', 'accuracy'))
    top_1s.append(self._mock_extra_result(.77, 'top_1', 'accuracy'))

    result_dict = {}
    result_dict['total_time'] = total_times
    result_dict['top_1'] = top_1s
    agg_result = util.aggregate_extra_results(result_dict)
    print('agg_result:{}'.format(agg_result))
    self.assertEqual(agg_result[0]['mean'], 175)
    self.assertEqual(agg_result[0]['result_type'], 'total_time')
    self.assertEqual(len(agg_result), 2)

  def _mock_extra_result(self, result, result_type, metric):
    result_dict = {}
    result_dict['result_units'] = metric
    result_dict['result_type'] = result_type
    result_dict['result'] = result
    return result_dict

  def _mock_config(self, test_id):
    config = {}
    config['test_id'] = test_id
    config['batch_size'] = 128
    config['gpus'] = 2
    config['model'] = 'resnet50'
    return config

  def _mock_result(self, test_id, imgs_sec, extra_results=None):
    result = {}
    result['config'] = self._mock_config(test_id)
    if imgs_sec:
      result['imgs_sec'] = imgs_sec
    result['test_id'] = test_id
    result['gpu'] = 2
    result['batches_sampled'] = 100
    if extra_results:
      result['raw_extra_results'] = extra_results
    return result
