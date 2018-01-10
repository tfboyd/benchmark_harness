"""Tests nvidia_tools module."""
import unittest

from mock import patch
import nvidia


class TestNvidiaTools(unittest.TestCase):

  @patch('tools.local_command.run_local_command')
  def test_get_gpu_info(self, run_local_command_mock):
    """Tests get gpu info parses expected value into expected components."""
    run_local_command_mock.return_value = [0, 'blah blah\n381.99, GTX 1080 \n']
    driver, gpu_info = nvidia.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)

  @patch('tools.local_command.run_local_command')
  def test_get_gpu_info_quadro(self, run_local_command_mock):
    """Tests gpu info returns second entry if first entry is a Quadro."""
    run_local_command_mock.return_value = [
        0, 'blah\n200.99, Quadro K900 \n381.99, GTX 1080\n'
    ]
    driver, gpu_info = nvidia.get_gpu_info()
    self.assertEqual('381.99', driver)
    self.assertEqual('GTX 1080', gpu_info)

  @patch('tools.local_command.run_local_command')
  def test_is_ok_to_run_false(self, run_local_command_mock):
    """"""
    smi_test = 'tools/test_files/example_nvidia-smi_processes.txt'
    f = open(smi_test)
    run_local_command_mock.return_value = [0, f.read()]
    ok_to_run = nvidia.is_ok_to_run()
    self.assertFalse(ok_to_run)

  @patch('tools.local_command.run_local_command')
  def test_is_ok_to_run(self, run_local_command_mock):
    """"""
    smi_test = 'tools/test_files/example_nvidia-smi_no_processes.txt'
    f = open(smi_test)
    run_local_command_mock.return_value = [0, f.read()]
    ok_to_run = nvidia.is_ok_to_run()
    self.assertTrue(ok_to_run)
