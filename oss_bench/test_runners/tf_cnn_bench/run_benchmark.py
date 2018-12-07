"""Runs tf_cnn_benchmark based tests."""
from __future__ import print_function
import argparse
import datetime
import os
import time

import yaml

from test_runners.common import cluster_local
from test_runners.tf_cnn_bench import command_builder
from test_runners.tf_cnn_bench import reporting
from upload import result_info


class TestRunner(object):
  """Run benchmark tests and record results.

  Args:
    configs (str): Comma delimited string of paths to yaml config files that
      detail the tests to run.
    workspace (str): Path to workspace to store logs and results.
    bench_home (str): Path to tf_cnn_benchmark script.
    auto_test_config (dict): Supplemental config values from oss_test harness,
      e.g. tensorflow version and hashes for tf_cnn_benchmark repo.
    debug_level (int): Debug level with supported values 0 and 1.
  """

  def __init__(self,
               configs,
               workspace,
               bench_home,
               auto_test_config=None,
               debug_level=1):
    """Initalize the TestRunner with values."""
    self.auto_test_config = auto_test_config
    self.configs = configs
    self.workspace = workspace
    self.local_log_dir = os.path.join(self.workspace, 'logs')
    self.local_stdout_file = os.path.join(self.local_log_dir, 'stdout.log')
    self.local_stderr_file = os.path.join(self.local_log_dir, 'stderr.log')
    self.bench_home = bench_home
    self.debug_level = debug_level

    self._make_log_dir(self.local_log_dir)

  def _make_log_dir(self, local_log_dir):
    # Creates workspace and default log folder
    if not os.path.exists(local_log_dir):
      print('Making log directory:{}'.format(local_log_dir))
      os.makedirs(local_log_dir)

  def results_directory(self, run_config):
    """Determine and create the results directory.

    Args:
      run_config: Config representing the test to run.

    Returns:
      Path to store results of the test.
    """
    suite_dir_name = '{}_{}'.format(run_config['test_suite_start_time'],
                                    run_config['test_id'])
    datetime_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    test_result_dir = '{}'.format(datetime_str)
    result_dir = os.path.join(self.workspace, 'results', suite_dir_name,
                              test_result_dir)

    # Creates workspace and default log folder
    if not os.path.exists(result_dir):
      print('Making results directory:{}'.format(result_dir))
      os.makedirs(result_dir)

    return result_dir

  def run_benchmark(self, run_config, instance):
    """Run single distributed tests for the passed config.

    Args:
      run_config: Config representing the test to run.
      instance: Instance to run the tests against.

    Returns:
      Path to results of the test.
    """
    # Timestamp and other values added for reporting
    run_config['timestamp'] = int(time.time())
    run_config['workspace'] = self.workspace
    result_dir = self.results_directory(run_config)
    # Sets training dir to results folder
    if ('train_dir' in run_config and
        not run_config['train_dir'].startswith('/')):
      run_config['train_dir'] = os.path.join(result_dir,
                                             run_config['train_dir'])

    cmd = command_builder.build_run_command(run_config)
    run_config['cmd'] = cmd
    test_home = self.bench_home

    self._write_results_file(result_dir,
                             yaml.dump(run_config),
                             'config.yaml')

    extra_results = []
    i = 0
    worker_type = 'eval' if run_config.get('eval') else 'worker'
    cmd = 'cd {}; {}'.format(test_home, cmd)
    print('[{}] {} | Run benchmark({}):{}'.format(
        run_config.get('copy', '0'), worker_type, run_config['test_id'], cmd))
    stdout_file = os.path.join(result_dir,
                               '{}_{}_stdout.log'.format(worker_type, i))
    stderr_file = os.path.join(result_dir,
                               '{}_{}_stderr.log'.format(worker_type, i))
    exec_time = time.time()
    t = instance.ExecuteCommandInThread(
        cmd, stdout_file, stderr_file, print_error=True)
    t.join()

    worker_time = self._get_milliseconds_diff(exec_time)
    total_time = worker_time
    print('Worker time: {}ms'.format(worker_time))
    result_info.build_result_info(extra_results, worker_time, 'worker_time')

    # run_eval is only used to run an eval after training. A single eval test
    # without a preceding training run is just a "plain" worker test.
    if 'run_eval' in run_config and not run_config.get('eval'):
      eval_config = run_config.copy()
      eval_config.update(run_config['run_eval'])
      eval_cmd = command_builder.build_run_command(eval_config)
      eval_cmd = 'cd {}; {}'.format(test_home, eval_cmd)
      print('[{}] eval_worker | Run benchmark({}):{}'.format(
          run_config.get('copy', '0'), run_config['test_id'], eval_cmd))
      stdout_file = os.path.join(result_dir, 'eval_%d_stdout.log' % i)
      stderr_file = os.path.join(result_dir, 'eval_%d_stderr.log' % i)
      eval_exec_time = time.time()
      t = instance.ExecuteCommandInThread(
          eval_cmd, stdout_file, stderr_file, print_error=True)
      t.join()
      eval_time = self._get_milliseconds_diff(eval_exec_time)
      print('Eval time: {}ms'.format(eval_time))
      result_info.build_result_info(extra_results, eval_time, 'eval_time')
      total_time = self._get_milliseconds_diff(exec_time)

    print('Total time: {}ms'.format(total_time))
    result_info.build_result_info(extra_results, total_time, 'total_time')

    self._write_results_file(result_dir,
                             yaml.dump(extra_results),
                             'extra_results.yaml')

    return result_dir

  def run_test_suite(self, full_config):
    """Run benchmarks defined by full_config.

    Args:
      full_config: Config representing tests to run.
    """

    # Left over from system that could have multiple instances for distributed
    # tests. Currently uses first and only instance from list.
    instance = cluster_local.UseLocalInstances(
        virtual_env_path=full_config.get('virtual_env_path'))

    # Folder to store suite results
    full_config['test_suite_start_time'] = datetime.datetime.now().strftime(
        '%Y%m%dT%H%M%S')

    # Configs for the test suite
    test_suite = command_builder.build_test_config_suite(
        full_config, self.debug_level)

    for _, test_configs in enumerate(test_suite):
      last_config = None
      for _, test_config in enumerate(test_configs):
        last_config = test_config
        # Executes oom test or the normal benchmark.
        if test_config.get('oom_test'):
          low = test_config['oom_low']
          high = test_config['oom_high']
          next_val = high
          lowest_oom = high
          while next_val != -1:
            print('OOM testing--> low:{} high:{} next_val:{}'.format(
                low, high, next_val))
            test_config['batch_size'] = next_val
            result_dir = self.run_benchmark(test_config, instance)
            oom = reporting.check_oom(
                os.path.join(result_dir, 'worker_0_stdout.log'))
            if oom and next_val < lowest_oom:
              lowest_oom = next_val
            low, high, next_val = reporting.oom_batch_size_search(
                low, high, next_val, oom)
            print('Lowest OOM Value:{}'.format(lowest_oom))
        else:
          result_dir = self.run_benchmark(test_config, instance)

      suite_dir_name = '{}_{}'.format(last_config['test_suite_start_time'],
                                      last_config['test_id'])
      reporting.process_folder(
          os.path.join(self.workspace, 'results', suite_dir_name),
          report_config=self.auto_test_config)

  def load_yaml_configs(self, config_paths, base_dir=None):
    """Convert string of config paths into list of yaml objects.

    Args:
      config_paths: Paths to yaml configs to load.
      base_dir: Base directory prefixed to each config_path.

    Returns:
      List of config dicts.

    Raises:
      Exception: if config is empty or not found
    """
    configs = []
    for _, config_path in enumerate(config_paths):
      if base_dir is not None:
        config_path = os.path.join(base_dir, config_path)
      f = open(config_path)
      config = yaml.safe_load(f)
      f.close()
      if config:
        config['config_path'] = config_path
        configs.append(config)
      else:
        print('Config was empty:{}'.format(config_path))
        raise Exception('Config was empty:{}'.format(config_path))
    return configs

  def run_tests(self):
    """Run the tests."""
    # Loads up the configs.
    configs = self.load_yaml_configs(self.configs.split(','))

    # For each config (parent) loop over each sub_config.
    for _, global_config in enumerate(configs):
      base_dir = os.path.dirname(global_config['config_path'])
      sub_configs = self.load_yaml_configs(
          global_config['sub_configs'], base_dir=base_dir)
      for _, run_config in enumerate(sub_configs):
        full_config = run_config.copy()

        # Copy global configs into the sub_config to create the full_config
        # values from the global config will overwrite those in the sub_config
        # during via command_builder later in the process.
        if global_config:
          for k, v in global_config.items():
            if k != 'run_configs':
              full_config[k] = v

        self.run_test_suite(full_config)

  def _get_milliseconds_diff(self, start_time):
    """Convert seconds to int milliseconds."""
    return int(round((time.time() - start_time) * 1000))

  def _write_results_file(self, result_dir, data, name):
    """Writes data (string) to results directory."""
    config_file_out = os.path.join(result_dir, name)
    config_out = open(config_file_out, 'w')
    config_out.write(data)
    config_out.close()


def main():
  """Program main, called after args are parsed into FLAGS."""
  test_runner = TestRunner(FLAGS.config, FLAGS.workspace, FLAGS.bench_home)
  test_runner.run_tests()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Popular Flags
  parser.add_argument(
      '--config',
      type=str,
      default='configs/run_config.yaml',
      help='Config YAML to use')
  parser.add_argument(
      '--workspace',
      type=str,
      default='/tmp/benchmark_workspace',
      help='Local workspace to hold logs and results')
  parser.add_argument(
      '--debug_level',
      type=int,
      default=1,
      help='Set to debug level: 0, 1, 5. Default 1')
  parser.add_argument(
      '--bench_home',
      type=str,
      default=os.path.join(os.environ['HOME'], 'tf_cnn_bench'),
      help='Path to the benchmark scripts')

  FLAGS, unparsed = parser.parse_known_args()

  main()
