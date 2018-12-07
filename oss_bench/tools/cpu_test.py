"""Tests nvidia_tools module."""
import unittest

import tools.cpu as cpu
from mock import patch


class TestCpu(unittest.TestCase):

  @patch('tools.local_command.run_local_command')
  def test_get_model_name(self, run_local_command_mock):
    """Tests extract the cpu model name."""
    run_local_command_mock.return_value = [
        0, 'model name  : Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz\n'
    ]
    model_name = cpu._model_name()
    self.assertEqual('Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz', model_name)

  @patch('tools.local_command.run_local_command')
  def test_get_socket_count(self, run_local_command_mock):
    """Tests get socket count."""
    run_local_command_mock.return_value = [0, '2\n']
    socket_count = cpu._socket_count()
    self.assertEqual(2, socket_count)

  @patch('tools.cpu._socket_count')
  @patch('tools.local_command.run_local_command')
  def test_get_core_count(self, run_local_command_mock, mock_socket):
    """Tests get number of cores."""
    run_local_command_mock.return_value = [0, 'cpu cores  : 6\n']
    mock_socket.return_value = 2
    core_count = cpu._core_count()
    self.assertEqual(12, core_count)

  @patch('tools.local_command.run_local_command')
  def test_get_cpuinfo(self, run_local_command_mock):
    """Tests git cpuinfo echos back."""
    run_local_command_mock.return_value = [0, 'foo I show whatever shows up\n']
    cpuinfo = cpu._cpu_info()
    self.assertEqual('foo I show whatever shows up\n', cpuinfo)
