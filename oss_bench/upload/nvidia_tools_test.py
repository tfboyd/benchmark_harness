"""Tests nvidia_tools module."""
import unittest

from mock import patch
import nvidia_tools


class TestNvidiaTools(unittest.TestCase):

  @patch('nvidia_tools._run_local_command')
  def test_build_row(self, run_local_command_mock):
    run_local_command_mock.return_value = [0, 'blah blah\n381.99, GTX 1080 \n']
    driver, gpu_info = nvidia_tools.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)
