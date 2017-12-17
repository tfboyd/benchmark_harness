"""Sets up the environment and runs the test suites."""
from __future__ import print_function
import argparse
import os
import subprocess
import sys
import yaml


class BenchmarkRunner(object):
  """Setup environment an execute suite of tests.

  Args:
    workspace (str): workspace to download git code and store logs in. Path is
      either absolute to the host or mounted path on docker if using docker.
  """

  def __init__(self, workspace, test_config):
    """Initalize the BenchmarkRunner with values."""
    self.workspace = workspace
    self.git_repo_base = os.path.join(self.workspace, 'git')
    self.logs_dir = os.path.join(self.workspace, 'logs')
    self.test_config = test_config

  def run_local_command(self, cmd, stdout=None):
    """Run a command in a subprocess and log result.

    Args:
      cmd (str): Command to run.
      stdout (str, optional): File to write standard out.
    """
    if stdout is None:
      stdout = os.path.join(self.logs_dir, 'log.txt')
    print(cmd)
    f = None
    if stdout:
      f = open(stdout, 'a')
      f.write(cmd + '\n')
    for line in self._run_local_command(cmd):
      if line.strip('\n'):
        print(line.strip('\n'))
        if f:
          f.write(line.strip('\n') + '\n')

  def _run_local_command(self, cmd):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    while True:
      retcode = p.poll()
      line = p.stdout.readline()
      yield line
      if retcode is not None:
        break

  def _git_clone(self, git_repo, local_folder, branch=None, sha_hash=None):
    """Clone, update, or synce a repo.

    If the clone already exists the repo will be updated via a pull.

    Args:
      git_repo (str): Command to
      local_folder (str): Where to clone repo into.
      branch (str, optional): Branch to checkout.
      sha_hash (str, optional): Hash to sync to.
    """
    if os.path.isdir(local_folder):
      git_clone_or_pull = 'git -C {} pull'.format(local_folder)
    else:
      git_clone_or_pull = 'git clone {} {}'.format(git_repo, local_folder)
    self.run_local_command(git_clone_or_pull)

    if branch is not None:
      branch_cmd = 'git -C {} checkout {}'.format(local_folder, branch)
      self.run_local_command(branch_cmd)

    if sha_hash is not None:
      sync_to_hash_cmd = 'git -C {} reset --hard {}'.format(
          local_folder, sha_hash)
      self.run_local_command(sync_to_hash_cmd)

  def _call_tf_benchmarks_tests(self, auto_config):
    """Runs tests based on tf_cnn_benchmarks.

    Args:
      auto_config: Parent test config with information useful to child test
        framework.
    """
    # Module is loaded by this module.
    # pylint: disable=C6204
    import test_runners.tf_cnn_bench.run_benchmark as run_benchmark

    rel_config_paths = auto_config['tf_cnn_bench_configs']
    config_paths = []
    for rel_config_path in rel_config_paths:
      config_paths.append(os.path.join(self.git_repo_base, rel_config_path))

    config = ','.join(config_paths)

    tf_cnn_bench_path = os.path.join(self.git_repo_base,
                                     'benchmarks/scripts/tf_cnn_benchmarks')

    test_runner = run_benchmark.TestRunner(
        config,
        os.path.join(self.logs_dir, 'tf_cnn_workspace'),
        tf_cnn_bench_path,
        auto_test_config=auto_config)
    test_runner.run_tests()

  def _load_config(self):
    """Returns auto_run config for the environment."""
    config_path = None
    if self.test_config.startswith('/'):
      config_path = self.test_config
    else:
      config_path = os.path.join(
            os.path.dirname(__file__), self.test_config)
    f = open(config_path)
    return yaml.safe_load(f)

  def _clone_repos(self):
    """Clone repos with modules containing tests or utility modules."""
    self._git_clone(
        'https://github.com/tensorflow/benchmarks.git',
        os.path.join(self.git_repo_base, 'benchmarks'),
        sha_hash='267d7e81977f23998078f39afd48e9a97c3acf5a')
    self._git_clone('https://github.com/tfboyd/benchmark_harness.git',
                    os.path.join(self.git_repo_base, 'benchmark_harness'))

  def run_tests(self):
    """Runs the benchmark suite."""
    try:
      os.makedirs(self.logs_dir)
    except OSError:
      if not os.path.isdir(self.logs_dir):
        raise
    test_config = self._load_config()

    auth_token_path = os.path.join('/auth_tokens/', test_config['report_auth'])
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_token_path

    # pick a directory, download tfboyd for auto_run and then tf_cnn_benchmarks
    # then kick off some tests via auto_run.
    self._clone_repos()

    # Modify the python path for the libraries for the tests to run and then
    # import them.
    git_python_lib_paths = ['benchmark_harness/oss_bench']
    for lib_path in git_python_lib_paths:
      sys.path.append(os.path.join(self.git_repo_base, lib_path))

    # Module is loaded by this module.
    # pylint: disable=C6204
    import upload.nvidia_tools as nvidia_tools
    # Sets system GPU info on test_config for child modules to consume.
    test_config['gpu_driver'], test_config[
        'accel_type'] = nvidia_tools.get_gpu_info()
    self._call_tf_benchmarks_tests(test_config)


def main():
  runner = BenchmarkRunner(FLAGS.workspace, FLAGS.test_config)
  runner.run_tests()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test-config',
      type=str,
      default='configs/default.yaml',
      help='Path to the test_config or default to run default config')
  parser.add_argument(
      '--workspace',
      type=str,
      default='/workspace',
      help='Path to the workspace')

  FLAGS, unparsed = parser.parse_known_args()

  main()
