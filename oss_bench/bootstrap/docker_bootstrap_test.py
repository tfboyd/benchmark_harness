"""Tests docker_bootstrap module."""
from __future__ import print_function

import unittest
from mock import patch
from bootstrap import docker_bootstrap


class TestDockerBootstrap(unittest.TestCase):
  """Tests for docker_bootstrap module."""

  @patch('bootstrap.docker_bootstrap.Bootstrap.git_clone')
  @patch('bootstrap.docker_bootstrap.Bootstrap.existing_process_check')
  @patch('bootstrap.docker_bootstrap.Bootstrap._run_local_command')
  def test_run_test_no_boot_config(self, run_command_mock, process_check_mock,
                                   _):
    """Tests run_test with out a boot config and no process check."""
    bootstrap = docker_bootstrap.Bootstrap(
        '/docker_folder',
        '/workspace',
        'test_config.yaml',
        docker_tag='tf_test/framework',
        auth_token_dir='/test/auth_token',
        gpu_process_check=False)

    bootstrap.run_tests()
    # Assumes last call was to kick off the docker image.
    arg0 = run_command_mock.call_args[0][0]
    self.assertEqual(
        arg0, 'nvidia-docker run  --rm  '
        '-v /test/auth_token:/auth_tokens -v /workspace:/workspace '
        'tf_test/framework python '
        '/workspace/git/benchmark_harness/oss_bench/harness/controller.py'
        ' --workspace=/workspace --test-config=test_config.yaml'
        ' --framework=tensorflow')
    process_check_mock.assert_not_called()

  @patch('bootstrap.docker_bootstrap.Bootstrap.git_clone')
  @patch('bootstrap.docker_bootstrap.Bootstrap.existing_process_check')
  @patch('bootstrap.docker_bootstrap.Bootstrap._run_local_command')
  def test_build_docker_cmd_pure(self, run_command_mock, process_check_mock, _):
    """Tests rim_test with docker only and no boot config."""
    bootstrap = docker_bootstrap.Bootstrap(
        '/docker_folder',
        '/workspace',
        'test_config.yaml',
        docker_tag='tf_test/framework',
        auth_token_dir='/test/auth_token',
        pure_docker=True)

    bootstrap.run_tests()
    # Assumes last call was to kick off the docker image.
    arg0 = run_command_mock.call_args[0][0]
    self.assertEqual(
        arg0, 'docker run  --rm  '
        '-v /test/auth_token:/auth_tokens -v /workspace:/workspace '
        'tf_test/framework python '
        '/workspace/git/benchmark_harness/oss_bench/harness/controller.py'
        ' --workspace=/workspace --test-config=test_config.yaml'
        ' --framework=tensorflow')
    process_check_mock.assert_called()

  @patch('bootstrap.docker_bootstrap.Bootstrap.git_clone')
  @patch('bootstrap.docker_bootstrap.Bootstrap.existing_process_check')
  @patch('bootstrap.docker_bootstrap.Bootstrap._run_local_command')
  def test_build_docker_cmd_boot_config(self, run_command_mock,
                                        process_check_mock, _):
    """Tests run_test with a bootconfig."""
    bootstrap = docker_bootstrap.Bootstrap(
        '/docker_folder',
        '/workspace',
        'test_config.yaml',
        bootstrap_config='bootstrap/test_config/config_multi_mount_points.yaml',
        docker_tag='tf_test/framework',
        auth_token_dir='/test/auth_token')

    bootstrap.run_tests()
    # Assumes last call was to kick off the docker image.
    arg0 = run_command_mock.call_args[0][0]

    expected_docker_cmd = (
        'nvidia-docker run  --rm  '
        '-v /home/user/data/imagenet:/data/imagenet '
        '-v /home/user/data/cifar-10:/data/cifar-10 '
        '-v /test/auth_token:/auth_tokens '
        '-v /workspace:/workspace tf_test/framework python '
        '/workspace/git/benchmark_harness/oss_bench/harness/controller.py '
        '--workspace=/workspace --test-config=test_config.yaml '
        '--framework=tensorflow')

    self.assertEqual(arg0, expected_docker_cmd)
    process_check_mock.assert_called()

  @patch('bootstrap.docker_bootstrap.Bootstrap.git_clone')
  @patch('bootstrap.docker_bootstrap.Bootstrap.existing_process_check')
  @patch('bootstrap.docker_bootstrap.Bootstrap._run_local_command')
  def test_build_docker_cmd_mxnet(self, run_command_mock, process_check_mock,
                                  _):
    """Tests run_test with mxnet as the framework."""
    bootstrap = docker_bootstrap.Bootstrap(
        '/docker_folder',
        '/workspace',
        'test_config.yaml',
        framework='mxnet',
        docker_tag='tf_test/framework',
        auth_token_dir='/test/auth_token')

    bootstrap.run_tests()
    # Assumes last call was to kick off the docker image.
    arg0 = run_command_mock.call_args[0][0]
    self.assertEqual(
        arg0, 'nvidia-docker run  --rm  '
        '-v /test/auth_token:/auth_tokens -v /workspace:/workspace '
        'tf_test/framework python '
        '/workspace/git/benchmark_harness/oss_bench/harness/controller.py'
        ' --workspace=/workspace --test-config=test_config.yaml'
        ' --framework=mxnet')
    process_check_mock.assert_called()

  @patch('bootstrap.docker_bootstrap.Bootstrap.git_clone')
  @patch('bootstrap.docker_bootstrap.Bootstrap.existing_process_check')
  @patch('bootstrap.docker_bootstrap.Bootstrap._run_local_command')
  def test_build_docker_cmd_pytorch(self, run_command_mock, process_check_mock,
                                    _):
    """Tests run_test with pytorch as the framework."""
    bootstrap = docker_bootstrap.Bootstrap(
        '/docker_folder',
        '/workspace',
        'test_config.yaml',
        framework='pytorch',
        docker_tag='tf_test/framework',
        auth_token_dir='/test/auth_token')

    bootstrap.run_tests()
    # Assumes last call was to kick off the docker image.
    arg0 = run_command_mock.call_args[0][0]
    self.assertEqual(
        arg0, 'nvidia-docker run --ipc=host --rm  '
        '-v /test/auth_token:/auth_tokens -v /workspace:/workspace '
        'tf_test/framework python '
        '/workspace/git/benchmark_harness/oss_bench/harness/controller.py'
        ' --workspace=/workspace --test-config=test_config.yaml'
        ' --framework=pytorch')
    process_check_mock.assert_called()
