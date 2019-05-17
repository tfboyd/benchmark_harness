"""Sets up the environment and runs benchmarks."""
from __future__ import print_function
import argparse
import os
import subprocess
import sys
import yaml

# Set after module is dynamically loaded.
tracker = None


class BenchmarkRunner(object):
  """Setup environment an execute tests.

  Example:
    Run from command line with:
      python -m harness.controller --workspace=/workspace

    Note: Note that this will pull fresh code from git to the workspace folder
      and will not use your local changes beyond this module.  For this to work
      with the default.yaml do the following:
        - Add .json credentials, e.g. 'tensorflow_performance_upload_tb.json' to
         /auth_tokens or update configs/default.yaml.
        - Current code can be copied to workspace/git/benchmark_harness, do
          development from the workspace folder directly, or adjust the code
          below to put desired code into `sys.path`.

  Args:
    workspace (str): workspace to download git code and store logs in. Path is
      either absolute to the host or mounted path on docker if using docker.
    test_config (str): path to yaml config file.
    framework (str): framework to test.
  """

  def __init__(self, workspace, test_config, framework='tensorflow'):
    """Initalize the BenchmarkRunner with values."""
    self.workspace = workspace
    self.git_repo_base = os.path.join(self.workspace, 'git')
    self.logs_dir = os.path.join(self.workspace, 'logs')
    self.test_config = test_config
    self.framework = framework

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
      line_str = line.decode('utf-8')
      if line_str.strip('\n'):
        print(line_str.strip('\n'))
        if f:
          f.write(line_str.strip('\n') + '\n')

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

  def _tf_model_bench(self, auto_config):
    """Runs tf model benchmarks.

    Args:
      auto_config: Configuration for running tf_model_bench tests.
    """
    # Module is loaded by this module.
    # pylint: disable=C6204
    import test_runners.tf_models.runner as runner

    bench_home = os.path.join(self.git_repo_base, 'tf_models')

    # call tf_garden runner with lists of tests from the test_config
    run = runner.TestRunner(
        os.path.join(self.logs_dir, 'tf_models'),
        bench_home,
        auto_test_config=auto_config)
    run.run_tests(auto_config['tf_models_tests'])

  def _keras_tf_model_bench(self, auto_config):
    """Runs keras tf model benchmarks.

    Args:
      auto_config: Configuration for running tf_model_bench tests.
    """
    # Module is loaded by this module.
    # pylint: disable=C6204
    import test_runners.keras_tf_models.runner as runner

    bench_home = os.path.join(self.git_repo_base, 'tf_models')

    # call tf_garden runner with lists of tests from the test_config
    run = runner.TestRunner(
        os.path.join(self.logs_dir, 'keras_tf_models'),
        bench_home,
        auto_test_config=auto_config)
    run.run_tests(auto_config['keras_tf_models_tests'])

  def _tf_cnn_bench(self, auto_config):
    """Runs tf cnn benchmarks.

    Args:
      auto_config: Configuration for running tf_model_bench tests.
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
    """Returns config representing tests to run."""
    config_path = None
    if self.test_config.startswith('/'):
      config_path = self.test_config
    else:
      config_path = os.path.join(os.path.dirname(__file__), self.test_config)
    f = open(config_path)
    return yaml.safe_load(f)

  def _clone_tf_repos(self):
    """Clone repos with modules containing tests or utility modules.

    After cloning and optionally moving to a specific branch or hash, repo
    information is stored in test_config for downstream tests to store
    as part of their results.
    """
    self._git_clone('https://github.com/tensorflow/benchmarks.git',
                    os.path.join(self.git_repo_base, 'benchmarks'))

    self._git_clone('https://github.com/tfboyd/models.git',
                    os.path.join(self.git_repo_base, 'tf_models'),
                    branch='resnet_perf_tweaks')

  def _make_logs_dir(self):
    try:
      os.makedirs(self.logs_dir)
    except OSError:
      if not os.path.isdir(self.logs_dir):
        raise

  def _store_repo_info(self, test_config):
    """Stores info about git repos in test_config.

    Note: Assumes benchmark_harness/oss_bench has been added to sys.path.

    Args:
      test_config: dict to add git repo info.
    """
    # Module cannot be loaded until after repo is cloned and added to sys.path.
    # pylint: disable=C6204
    import tools.git_info as git_info
    git_dirs = ['benchmark_harness', 'benchmarks', 'tf_models']
    test_config['git_repo_info'] = {}
    for repo_dir in git_dirs:
      full_path = os.path.join(self.git_repo_base, repo_dir)
      git_info_dict = {}
      git_info_dict['describe'] = git_info.git_repo_describe(full_path)
      git_info_dict['last_commit_id'] = git_info.git_repo_last_commit_id(
          full_path)
      test_config['git_repo_info'][repo_dir] = git_info_dict

  def run_tensorflow_tests(self, test_config):
    """Runs all TensorFlow based tests.

    Args:
      test_config: Config representing the tests to run.

    """
    # then kick off some tests via auto_run.
    self._clone_tf_repos()

    # Set python path by overwrite, which is not ideal.
    relative_tf_models_path = 'tf_models'
    os.environ['PYTHONPATH'] = os.path.join(self.git_repo_base,
                                            relative_tf_models_path)

    self._store_repo_info(test_config)

    # pylint: disable=C6204
    import tools.tf_version as tf_version

    # Sets system GPU info on test_config for child modules to consume.
    version, git_version = tf_version.get_tf_full_version()
    test_config['framework_version'] = version
    test_config['framework_describe'] = git_version

    # Run tf_cnn_bench tests if in config
    if 'tf_cnn_bench_configs' in test_config:
      tested = self.check_if_run(test_config, 'tf_cnn_bench')
      if not tested:
        self._tf_cnn_bench(test_config)
        self.update_state(test_config, 'tf_cnn_bench')
      else:
        print('Setup already tested for {} on {}'.format(
            'tf_cnn_bench', test_config))

    # Run tf_model_bench if list of tests is found
    if 'tf_models_tests' in test_config:
      tested = self.check_if_run(test_config, 'tf_models')
      if not tested:
        self._tf_model_bench(test_config)
        self.update_state(test_config, 'tf_models')
      else:
        print('Setup already tested for {} on {}'.format(
            'tf_models', test_config))

    # Run keras tf_model_bench if list of tests is found
    if 'keras_tf_models_tests' in test_config:
      tested = self.check_if_run(test_config, 'keras_tf_models')
      if not tested:
        self._keras_tf_model_bench(test_config)
        self.update_state(test_config, 'keras_tf_models')
      else:
        print('Setup already tested for {} on {}'.format(
            'keras_tf_models', test_config))

  def check_if_run(self, test_config, test):
    if test_config.get('track'):
      return tracker.check_state(
          self.workspace, self.framework, test_config['channel'],
          test_config['build_type'], test_config['framework_describe'], test)
    else:
      return False

  def update_state(self, test_config, test):
    if test_config.get('track'):
      tracker.update_state(self.workspace, self.framework,
                           test_config['channel'], test_config['build_type'],
                           test_config['framework_describe'], test)

  def run_mxnet_tests(self, test_config):
    """Runs all MXNet based tests.

    Args:
      test_config: Config representing the tests to run.
    """

    # Clone the mxnet repo so we know where it is.
    self._git_clone('https://github.com/apache/incubator-mxnet.git',
                    os.path.join(self.git_repo_base, 'mxnet_repo'))
    bench_home = os.path.join(self.git_repo_base,
                              'mxnet_repo/example/image-classification')

    # pylint: disable=C6204
    import mxnet as mx
    # pylint: disable=C6204
    from test_runners.mxnet import runner
    test_config['framework_version'] = mx.__version__
    test_config['framework_describe'] = mx.__version__

    tested = self.check_if_run(test_config, 'mxnet')
    if not tested:
      # Calls mxnet runner with lists of tests from the test_config
      run = runner.TestRunner(
          os.path.join(self.logs_dir, 'mxnet'),
          bench_home,
          auto_test_config=test_config)
      run.run_tests(test_config['mxnet_tests'])
      self.update_state(test_config, 'mxnet')
    else:
      print('Setup already tested for {} on {}'.format('mxnet', test_config))

  def run_pytorch_tests(self, test_config):
    """Runs all pytorch based tests.

    Args:
      test_config: Config representing the tests to run.
    """
    # Clone the pytorch examples repo.
    self._git_clone('https://github.com/pytorch/examples.git',
                    os.path.join(self.git_repo_base, 'pytorch_examples'))
    bench_home = os.path.join(self.git_repo_base, 'pytorch_examples')

    # pylint: disable=C6204
    import torch
    test_config['framework_version'] = torch.__version__
    test_config['framework_describe'] = torch.__version__
    # pylint: disable=C6204
    from test_runners.pytorch import runner

    tested = self.check_if_run(test_config, 'pytorch')
    if not tested:
      # Calls pytorch runner with lists of tests from the test_config
      run = runner.TestRunner(
          os.path.join(self.logs_dir, 'pytorch'),
          bench_home,
          auto_test_config=test_config)
      run.run_tests(test_config['pytorch_tests'])
      self.update_state(test_config, 'pytorch')
    else:
      print('Setup already tested for {} on {}'.format('pytorch', test_config))

  def run_tests(self):
    """Runs all tests based on the test_config."""
    self._make_logs_dir()
    test_config = self._load_config()
    if test_config['report_project'] != 'LOCAL':
      if test_config['report_auth'].startswith('/'):
        auth_token_path = test_config['report_auth']
      else:
        auth_token_path = os.path.join('/auth_tokens/',
                                       test_config['report_auth'])
      os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auth_token_path

    # Modify the python path for the libraries for the tests to run and then
    # import them.
    git_python_lib_paths = ['benchmark_harness/oss_bench']
    for lib_path in git_python_lib_paths:
      sys.path.append(os.path.join(self.git_repo_base, lib_path))

    if 'device' not in test_config or test_config['device'] != 'cpu':
      # Modules are loaded by this function.
      # pylint: disable=C6204
      import tools.nvidia as nvidia
      test_config['gpu_driver'], test_config[
          'accel_type'] = nvidia.get_gpu_info()

    # Modules are loaded by this function.
    # pylint: disable=C6204
    import tools.cpu as cpu_info
    cpu_data = {}
    cpu_data['model_name'], cpu_data['socket_count'], cpu_data[
        'core_count'], cpu_data['cpu_info'] = cpu_info.get_cpu_info()
    test_config['cpu_info'] = cpu_data
    global tracker
    # pylint: disable=C6204
    # pylint: disable=W0621
    import tools.tracker as tracker

    if self.framework == 'tensorflow':
      self.run_tensorflow_tests(test_config)
    elif self.framework == 'mxnet':
      self.run_mxnet_tests(test_config)
    elif self.framework == 'pytorch':
      self.run_pytorch_tests(test_config)
    else:
      raise ValueError('framework needs to be set to tensorflow or mxnet')


def main():
  runner = BenchmarkRunner(
      FLAGS.workspace, FLAGS.test_config, framework=FLAGS.framework)
  runner.run_tests()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test-config',
      type=str,
      default='configs/dev/default.yaml',
      help='Path to the test_config or default to run default config')
  parser.add_argument(
      '--workspace',
      type=str,
      default='/workspace',
      help='Path to the workspace')
  parser.add_argument(
      '--framework',
      type=str,
      default='tensorflow',
      help='Framework to be tested.')
  FLAGS, unparsed = parser.parse_known_args()

  main()
