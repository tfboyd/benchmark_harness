"""Runs benchmarks on various cloud or local systems."""
from __future__ import print_function
import argparse
import datetime
import os
import time

import cluster_local
import command_builder
import reporting
import util
import yaml


class TestRunner(object):
  """Run benchmark tests and record results.

  Args:
    configs (str): 
    workspace (str): 
    bench_home (str): 
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
    self.local_local_dir = os.path.join(self.workspace, 'logs')
    self.local_stdout_file = os.path.join(self.local_local_dir, 'stdout.log')
    self.local_stderr_file = os.path.join(self.local_local_dir, 'stderr.log')
    self.bench_home = bench_home
    self.debug_level = debug_level

    # Creates workspace and default log folder
    if not os.path.exists(self.local_local_dir):
      print('Making log directory:{}'.format(self.local_local_dir))
      os.makedirs(self.local_local_dir)

  def results_directory(self, run_config):
    """Determine and create the results directory

    Args:
      run_config: Config representing the test to run.  
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
    """Run single distributed tests for the passed config

    Args:
      run_config: Config representing the test to run.
      instance: Instance to run the tests against.
    """
    # Timestamp and other values added for reporting
    run_config['timestamp'] = int(time.time())
    run_config['workspace'] = self.workspace

    test_home = self.bench_home
    result_dir = self.results_directory(run_config)

    # Write config to results folder
    config_file_out = os.path.join(result_dir, 'config.yaml')
    config_out = open(config_file_out, 'w')
    config_out.write(yaml.dump(run_config))
    config_out.close()

    # TODO(tobyboyd@): No longer distributed remove threads.
    worker_threads = []
    i = 0
    cmd = command_builder.BuildDistributedCommandWorker(run_config)
    cmd = 'cd {}; {}'.format(test_home, cmd)
    print('[{}] worker | Run benchmark({}):{}'.format(
        run_config.get('copy', '0'), run_config['test_id'], cmd))
    stdout_file = os.path.join(result_dir, 'worker_%d_stdout.log' % i)
    stderr_file = os.path.join(result_dir, 'worker_%d_stderr.log' % i)
    t = instance.ExecuteCommandInThread(
        cmd, stdout_file, stderr_file, util.ExtractToStdout, print_error=True)
    worker_threads.append(t)

    for t in worker_threads:
      t.join()

    return result_dir

  def run_test_suite(self, full_config, instance):
    """Run distributed benchmarks

    Args:
      configs: Configs representing the tests to run.
      instance: Instance to run the tests against.

    """
    # Folder to store suite results
    full_config['test_suite_start_time'] = datetime.datetime.now().strftime(
        '%Y%m%dT%H%M%S')

    # Configs for the test suite
    test_suite = command_builder.LoadYamlRunConfig(full_config,
                                                   self.debug_level)

    for i, test_configs in enumerate(test_suite):
      last_config = None
      for i, test_config in enumerate(test_configs):
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
      reporting.process_results_folder(
          os.path.join(self.workspace, 'results', suite_dir_name),
          report_config=self.auto_test_config)

  def local_benchmarks(self, full_config):
    """ Run Local benchmark tests

    """
    print('Running Local Benchmarks')
    # Left over from system that could have multiple instances for distributed
    # tests. Currently returns first and only instance from list.
    instances = cluster_local.UseLocalInstances(
        virtual_env_path=full_config.get('virtual_env_path'))
    self.run_test_suite(full_config, instances[0])

  def load_yaml_configs(self, config_paths, base_dir=None):
    """Convert string of config paths into list of yaml objects

      If configs_string is empty a list with a single empty object is returned
    """
    configs = []
    for _, config_path in enumerate(config_paths):
      if base_dir is not None:
        config_path = os.path.join(base_dir, config_path)
      f = open(config_path)
      config = yaml.safe_load(f)
      config['config_path'] = config_path
      configs.append(config)
      f.close()
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

        # Override config with global config, used to change projects
        # or any other field, these values will also end up overriding
        # settings in individual 'run_configs'
        if global_config:
          for k, v in global_config.iteritems():
            if k != 'run_configs':
              full_config[k] = v

        self.local_benchmarks(full_config)


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
