"""Runs benchmarks for TF Models."""
from __future__ import print_function
import argparse
import datetime
import os
import time

from six.moves import range
import yaml

from test_runners.common import cluster_local
from test_runners.pytorch import reporting



class TestRunner(object):
  """Run benchmark tests and record results.

  Args:
    workspace (str): Path to workspace to store logs and results.
    bench_home (str): Path to tf_models repo.
    auto_test_config (dict): Supplemental config values from oss_test harness,
      e.g. tensorflow version and hashes for tf_cnn_benchmark repo.
    imagenet_dir (str): path to imagenet data processed into TFRecords.
  """

  def __init__(self,
               workspace,
               bench_home,
               auto_test_config=None,
               imagenet_dir='/data/pytorch/imagenet'):
    """Initalize the TestRunner with values."""
    self.workspace = workspace
    self.local_log_dir = os.path.join(self.workspace, 'logs')
    self.local_stdout_file = os.path.join(self.local_log_dir, 'stdout.log')
    self.local_stderr_file = os.path.join(self.local_log_dir, 'stderr.log')
    self.bench_home = bench_home
    self.imagenet_dir = imagenet_dir

    if auto_test_config is None:
      self.auto_test_config = {}
    else:
      self.auto_test_config = auto_test_config

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

  def run_benchmark(self, test_config, instance, copy=0):
    """Run single distributed tests for the passed config.

    Args:
      test_config: Config representing the test to run.
      instance: Execution module to use.
      copy: Copy to run if test is a repeat.

    Returns:
      Path to results of the test.
    """
    # Timestamp and other values added for reporting
    result_dir = self.results_directory(test_config)
    test_config['timestamp'] = int(time.time())
    test_config['workspace'] = self.workspace
    cmd = self._cmd_builder(test_config)
    test_config['cmd'] = cmd
    total_batches = test_config['total_batches']

    test_home = os.path.join(self.bench_home, test_config['cmd_path'])

    # Write config to results folder
    config_file_out = os.path.join(result_dir, 'config.yaml')
    config_out = open(config_file_out, 'w')
    config_out.write(yaml.dump(test_config))
    config_out.close()

    # TODO(tobyboyd@): No longer distributed remove threads.
    worker_threads = []
    i = 0
    cmd = 'cd {}; {}'.format(test_home, cmd)
    print('[{}] worker | Run benchmark({}):{}'.format(
        copy, test_config['test_id'], cmd))
    stdout_file = os.path.join(result_dir, 'worker_%d_stdout.log' % i)
    stderr_file = os.path.join(result_dir, 'worker_%d_stderr.log' % i)
    t = instance.ExecuteCommandInThread(
        cmd, stdout_file, stderr_file, print_error=True)
    worker_threads.append(t)

    # Wait for log file to appear
    wait_time = 0
    while t.is_alive() and not os.path.isfile(stdout_file):
      print('Waiting for log file. Waited for {} seconds.'.format(wait_time))
      time.sleep(2)
      wait_time += 2

    # TODO(tobyboyd@) fix fragile check for batch to stop on.
    # Example: Epoch: [0][130/40037] Time 0.397
    batch_killer = '{}/'.format(total_batches)
    while t.is_alive():
      with open(stdout_file, 'r') as log:
        for line in log:
          if batch_killer in line:
            print('{} batches complete. Kill Thread.'.format(batch_killer))
            instance.kill_processes()
            break
        time.sleep(5)

    for t in worker_threads:
      t.join()

    return result_dir

  def run_test_suite(self, test_config):
    """Run benchmarks defined by full_config.

    Args:
      test_config: Config representing tests to run.
    """
    # Folder to store suite results
    test_config['test_suite_start_time'] = datetime.datetime.now().strftime(
        '%Y%m%dT%H%M%S')

    instance = cluster_local.UseLocalInstances()
    for i in range(test_config['repeat']):
      self.run_benchmark(test_config, instance, copy=i)

    suite_dir_name = '{}_{}'.format(test_config['test_suite_start_time'],
                                    test_config['test_id'])
    reporting.process_folder(
        os.path.join(self.workspace, 'results', suite_dir_name),
        report_config=self.auto_test_config)

  def build_resnet_test_config(self, test_id, test_args, batch_size=32, gpus=1,
                               total_batches=300, repeat=3):
    """Returns a base test config for ResNet50-v1.5 tests.

    Args:
      test_id: unique id for the test.
      test_args: dictionary of arguments overwrite and augment the defaults.
      batch_size: batch-size per GPU, which will be translated into total
        batch desired by the mxnet tests by multiplying it by the number of
        gpus.
      gpus: number of gpus to run against.
    """
    config = {}
    config['total_batches'] = total_batches
    # Relative path in the repo to the test folder.
    config['cmd_path'] = 'imagenet'

    # PyTorch Automatically uses all GPUs it can see.
    gpu_list = ','.join(str(x) for x in range(gpus))
    visible_devices = 'CUDA_VISIBLE_DEVICES={}'.format(gpu_list)
    if gpus > 1:
      multi_gpu = (" --dist-url 'tcp://127.0.0.1:6001' --dist-backend 'nccl'"
                   "--multiprocessing-distributed --world-size 1 --rank 0 ")
      config['pycmd'] = 'python3 main.py {} {} {}'.format('{}',
                                                          multi_gpu,
                                                          self.imagenet_dir)
    else:
      config['pycmd'] = '{} python3 main.py {} {}'.format(visible_devices,
                                                          '{}',
                                                          self.imagenet_dir)
    config['test_id'] = test_id
    config['repeat'] = self.auto_test_config.get('repeat', repeat)
    # Normalized name of model being tested
    config['model'] = 'resnet50_v1.5'
    config['gpus'] = gpus
    config['batch_size'] = batch_size
    args = {}
    config['args'] = args
    # print stats every epoch.
    args['print-freq'] = self.auto_test_config.get('print_freq', 1)
    args['epochs'] = 1
    args['arch'] = 'resnet50'
    args['batch-size'] = batch_size * gpus
    args['workers'] = self.auto_test_config.get('input_threads', 5)

    # Override any args with the tests args
    args.update(test_args)

    return config

  def _cmd_builder(self, test_config):
    """Builds command to run test.

    Translates `test_config` into an command to run a benchmark.
    `test_config['args']` contains the command line arguments and
    `test_config['pycmd']` is the python command to execute.

    Args:
      test_config: dict representing the test to run.

    Returns:
      str of the command to execute to run the test.
    """
    arg_str = ''
    for key, value in sorted(test_config['args'].items()):
      arg_str += '--{} {} '.format(key, value)
    return test_config['pycmd'].format(arg_str)

  def warmup_resnet_imagenet_128_gpu_8_real(self):
    """Cache imagenet dataset. 8 GPU with batch size 128*8."""
    test_id = 'warmup_resnet_imagenet.gpu_8.128.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=128, gpus=8,
                                           repeat=1, total_batches=1300)
    self.run_test_suite(config)

  def renset50_32_gpu_1_real(self):
    """Tests ResNet50 real data data on 1 GPU with batch size 32."""
    test_id = 'resnet50.gpu_1.32.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=32, gpus=1)
    self.run_test_suite(config)

  def renset50_64_gpu_1_real(self):
    """Tests ResNet50 real data data on 8 GPU with batch size 64."""
    test_id = 'resnet50.gpu_1.64.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=64, gpus=1)
    self.run_test_suite(config)

  def renset50_64_gpu_2_real(self):
    """Tests ResNet50 real data data on 2 GPU with batch size 64."""
    test_id = 'resnet50.gpu_2.64.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=64, gpus=2)
    self.run_test_suite(config)

  def renset50_64_gpu_4_real(self):
    """Tests ResNet50 real data data on 4 GPU with batch size 64."""
    test_id = 'resnet50.gpu_4.64.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=64, gpus=4)
    self.run_test_suite(config)

  def renset50_64_gpu_8_real(self):
    """Tests ResNet50 real data data on 8 GPU with batch size 64."""
    test_id = 'resnet50.gpu_8.64.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=64, gpus=8)
    self.run_test_suite(config)

  def renset50_128_gpu_1_real(self):
    """Tests ResNet50 real data data on 8 GPU with batch size 128."""
    test_id = 'resnet50.gpu_1.128.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=128, gpus=1)
    self.run_test_suite(config)

  def renset50_128_gpu_2_real(self):
    """Tests ResNet50 real data data on 2 GPU with batch size 128."""
    test_id = 'resnet50.gpu_2.128.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=128, gpus=2)
    self.run_test_suite(config)

  def renset50_128_gpu_4_real(self):
    """Tests ResNet50 real data data on 4 GPU with batch size 128."""
    test_id = 'resnet50.gpu_4.128.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=128, gpus=4)
    self.run_test_suite(config)

  def renset50_128_gpu_8_real(self):
    """Tests ResNet50 real data data on 8 GPU with batch size 128."""
    test_id = 'resnet50.gpu_8.128.real'
    args = {}
    config = self.build_resnet_test_config(test_id, args, batch_size=128, gpus=8)
    self.run_test_suite(config)

  def run_tests(self, test_list):
    for t in test_list:
      getattr(self, t)()


def main():
  """Program main, called after args are parsed into FLAGS.

  Example:
    python runner.py --workspace=/workspace
    --bench-home=/mxnet_repo/incubator-mxnet/example/image-classification
    --train-data-dir=/mxnet_repo/train/data

  """
  test_runner = TestRunner(
      FLAGS.workspace, FLAGS.bench_home, imagenet_dir=FLAGS.train_data_dir)
  test_runner.run_tests(FLAGS.test_list.split(','))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Popular Flags
  parser.add_argument(
      '--workspace',
      type=str,
      default='/tmp/benchmark_workspace',
      help='Local workspace to hold logs and results')
  parser.add_argument(
      '--bench-home',
      type=str,
      default='',
      help='Path to pytorch/examples repo.')
  parser.add_argument(
      '--test-list',
      type=str,
      default='renset50v1_32_gpu_1',
      help='Comma separated list of tests to run.')
  parser.add_argument(
      '--train-data-dir',
      type=str,
      default='/data/pytorch/imagenet',
      help='Path to training data directory.')

  FLAGS, unparsed = parser.parse_known_args()

  main()
