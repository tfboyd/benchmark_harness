"""Tests nvidia_tools module."""
import unittest

from mock import patch
import nvidia_tools


class TestNvidiaTools(unittest.TestCase):

  @patch('nvidia_tools._run_local_command')
  def test_get_gpu_info(self, run_local_command_mock):
    run_local_command_mock.return_value = [0, 'blah blah\n381.99, GTX 1080 \n']
    driver, gpu_info = nvidia_tools.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)


  @patch('nvidia_tools._run_local_command')
  def test_get_gpu_info_quadro(self, run_local_command_mock):
    """Tests Quadro as the first card found."""
    run_local_command_mock.return_value = [0, 'blah blah\n200.99, Quadro K900 \n381.99, GTX 1080\n']
    driver, gpu_info = nvidia_tools.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)   
