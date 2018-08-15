"""Tests for google3.experimental.tf_benchmark_util.cluster_gce."""
from __future__ import print_function

import os
import pickle
import unittest

import gce_instance


class TestGceInstance(unittest.TestCase):
  """Tests the GCE Instance object."""

  def setUp(self):
    """Set path to test files."""
    self.base_dir = 'clusters/test_data'

  def test_return_create_instance_config_gpu(self):
    """Test creating a config to create an instance with a GPU accelerator.

    Tests that the config created to build a GCE instance with a GPU
    accelerator. The golden config is stored as a pickle that can be recreated
    with "print(pickle.dumps(config))".
    """
    project = 'google.com:test_project'
    zone = 'us-east1-d'
    image_id = 'test_image'
    instance_name = 'test_instance'
    instance_type = 'n1-standard-8'
    tag = 'tag'
    accelerator = 'k80-foo'
    accelerator_count = '5'
    config = gce_instance.return_create_instance_config(
        project,
        zone,
        image_id,
        instance_type,
        instance_name,
        tag,
        accelerator=accelerator,
        accelerator_count=accelerator_count,
        service_account='test-service-account@foobar.com')

    golden_file = os.path.join(self.base_dir, 'gce_config.pkl')
    config_expected = pickle.load(open(golden_file))

    self.assertEqual(config, config_expected)

  def test_return_create_instance_config_cpu_only(self):
    """Test creating a config to create an instance with a GPU accelerator.

    Tests that the config created to build a GCE instance without a GPU. The
    golden config is stored as a pickle that can be recreated with
    "print(pickle.dumps(config))".
    """
    project = 'google.com:test_project'
    zone = 'west-1a'
    image_id = 'test_image'
    instance_name = 'test_instance'
    instance_type = 'n1-standard-64'
    tag = 'tag'
    config = gce_instance.return_create_instance_config(
        project, zone, image_id, instance_type, instance_name, tag,
        service_account='test-service-account@foobar.com')

    golden_file = os.path.join(self.base_dir, 'gce_config_cpu_only.pkl')
    config_expected = pickle.load(open(golden_file))

    self.assertEqual(config, config_expected)

  def test_parse_instance_type_gpu(self):
    """Parse an instance string with a GPU."""
    instance_expected = 'n1-standard-7'
    accelerator_expected = 'k80-foo'
    count_expected = '4'
    instance, accelerator, count = gce_instance.parse_instance_type(
        '|'.join([instance_expected, accelerator_expected, count_expected]))
    self.assertEqual(instance_expected, instance)
    self.assertEqual(accelerator_expected, accelerator)
    self.assertEqual(count_expected, count)

  def test_parse_instance_gpu_no_count(self):
    """Parse an instance string with a GPU but no count."""
    instance_expected = 'n1-standard-7'
    accelerator_expected = 'k80-foo'
    try:
      gce_instance.parse_instance_type(
          '|'.join([instance_expected, accelerator_expected]))
      self.fail('Exception expected with accelerator count missing.')
    except ValueError as e:
      print(dir(e))
      self.assertIn('Instance type in unknown format', e.message)

  def test_parse_instance_type_cpu_only(self):
    """Parse a CPU only instance string."""
    instance_expected = 'n1-standard-7'
    instance, accelerator, count = gce_instance.parse_instance_type(
        '|'.join([instance_expected]))
    self.assertEqual(instance_expected, instance)
