"""Runs benchmarks for TF Models."""
from __future__ import print_function
import argparse
import datetime
import os
import time

import reporting
from test_runners.common import cluster_local
from test_runners.common import util
import yaml


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
               imagenet_dir='/data/imagenet'):
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
    # set the model_dir to be inside the results_dir
    test_config['args']['model_dir'] = os.path.join(result_dir, 'checkpoint')
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
      time.sleep(5)
      wait_time += 5

    batch_killer = '[{}]'.format(total_batches)
    while t.is_alive():
      with open(stdout_file, 'r') as log:
        for line in log:
          if batch_killer in line:
            print('{} batches complete. Kill Thread.'.format(batch_killer))
            instance.kill_processes()
            break
        time.sleep(10)

    for t in worker_threads:
      t.join()

    # Model dir is over 200MB for most runs and data is not needed.
    util.delete_files_in_folder(test_config['args']['model_dir'])

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
    for i in xrange(test_config['repeat']):
      self.run_benchmark(test_config, instance, copy=i)

    suite_dir_name = '{}_{}'.format(test_config['test_suite_start_time'],
                                    test_config['test_id'])
    reporting.process_folder(
        os.path.join(self.workspace, 'results', suite_dir_name),
        report_config=self.auto_test_config)

  def build_resnet_test_config(self,
                               test_id,
                               test_args,
                               batch_size=32,
                               gpus=1,
                               dtype='fp32',
                               version=1,
                               use_synth=False):
    """Returns a base test config for ResNet50-v2 tests.

    Args:
      test_id: unique id for the test.
      test_args: dictionary of arguments overwrite and augment the defaults.
      batch_size: batch-size per GPU, which will be translated into total
        batch (batch-size*GPUs) desired by the official/resnet..
      gpus: Number of gpus to run against.
      dtype: fp16 or fp32. defaults to fp32
      version: Version of ResNet50, default is 1.
      use_synth: If True use synthetic data.
    """
    config = {}
    config['total_batches'] = 400
    # Relative path in the repo to the test folder.
    config['cmd_path'] = 'official/resnet'
    config['pycmd'] = 'imagenet_main.py'
    config['test_id'] = test_id
    config['repeat'] = 3
    # Normalized name of model being tested
    if version == 1:
      config['model'] = 'resnet50'
    else:
      config['model'] = 'resnet50v2'

    # Reporting looks for this value
    if dtype == 'fp16':
      config['use_fp16'] = True
    else:
      config['use_fp16'] = False

    config['gpus'] = gpus
    config['batch_size'] = batch_size
    args = {}
    config['args'] = args
    if use_synth:
      args['use_synthetic_data'] = ''
    else:
      args['data_dir'] = self.imagenet_dir
    # Default for running on GPU
    args['intra_op_parallelism_threads'] = 1
    # Default to ResNet50v1
    args['resnet_version'] = version
    args['resnet_size'] = 50
    args['batch_size'] = batch_size * gpus
    args['num_gpus'] = gpus
    args['dtype'] = dtype
    args['hooks'] = 'ExamplesPerSecondHook LoggingTensorHook'

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
    for key, value in sorted(test_config['args'].iteritems()):
      arg_str += '--{} {} '.format(key, value)
    return 'python {} {}'.format(test_config['pycmd'], arg_str)

  def resnet50v2_64_gpu_1_real(self):
    """Tests ResNet50v2 real data data on 1 GPU with batch size 32."""
    test_id = 'garden.resnet50v2.gpu_1.64.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=1, version=2)
    self.run_test_suite(config)

  def resnet50v2_64_gpu_8_real(self):
    """Tests ResNet50v2 real data data on 8 GPU with batch size 32."""
    test_id = 'garden.resnet50v2.gpu_8.64.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=8, version=2)
    self.run_test_suite(config)

  def resnet50v2_64_gpu_1(self):
    """Tests ResNet50v2 synth data data on 1 GPU with batch size 64."""
    test_id = 'garden.resnet50v2.gpu_1.64'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=1, version=2, use_synth=True)
    self.run_test_suite(config)

  def resnet50v2_64_gpu_8(self):
    """Tests ResNet50v2 synth data data on 8 GPU with batch size 64."""
    test_id = 'garden.resnet50v2.gpu_8.64'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=8, version=2, use_synth=True)
    self.run_test_suite(config)

  def resnet50_64_gpu_1_real(self):
    """Tests ResNet50 real data data on 1 GPU with batch size 64."""
    test_id = 'garden.resnet50.gpu_1.64.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=1, version=1)
    self.run_test_suite(config)

  def resnet50_64_gpu_8_real(self):
    """Tests ResNet50 real data data on 8 GPU with batch size 64."""
    test_id = 'garden.resnet50.gpu_8.64.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=8, version=1)
    self.run_test_suite(config)

  def resnet50_64_gpu_1(self):
    """Tests ResNet50 synth data data on 1 GPU with batch size 64."""
    test_id = 'garden.resnet50.gpu_1.64'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=1, version=1, use_synth=True)
    self.run_test_suite(config)

  def resnet50_64_gpu_8(self):
    """Tests ResNet50 synth data data on 8 GPU with batch size 64."""
    test_id = 'garden.resnet50.gpu_8.64'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=64, gpus=8, version=1, use_synth=True)
    self.run_test_suite(config)

  def resnet50_128_gpu_1_real(self):
    """Tests ResNet50 real data data on 1 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_1.128.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=1, version=1)
    self.run_test_suite(config)

  def resnet50_128_gpu_2_real(self):
    """Tests ResNet50 real data data on 2 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_2.128.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=2, version=1)
    self.run_test_suite(config)

  def resnet50_128_gpu_4_real(self):
    """Tests ResNet50 real data data on 4 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_4.128.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=4, version=1)
    self.run_test_suite(config)

  def resnet50_128_gpu_8_real(self):
    """Tests ResNet50 real data data on 8 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_8.128.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=8, version=1)
    self.run_test_suite(config)

  def resnet50_128_gpu_1(self):
    """Tests ResNet50 synth data data on 1 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_1.128'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=1, version=1, use_synth=True)
    self.run_test_suite(config)

  def resnet50_128_gpu_2(self):
    """Tests ResNet50 synth data data on 2 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_2.128'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=2, version=1, use_synth=True)
    self.run_test_suite(config)

  def resnet50_128_gpu_4(self):
    """Tests ResNet50 synth data data on 4 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_4.128'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=4, version=1, use_synth=True)
    self.run_test_suite(config)

  def resnet50_128_gpu_8(self):
    """Tests ResNet50 synth data data on 8 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_8.128'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=8, version=1, use_synth=True)
    self.run_test_suite(config)

  def resnet50_128_gpu_1_real_fp16(self):
    """Tests ResNet50 FP16 real data data on 1 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_1.128.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=1, version=1, dtype='fp16')
    self.run_test_suite(config)

  def resnet50_128_gpu_8_real_fp16(self):
    """Tests ResNet50 FP16 real data data on 8 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_8.128.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=128, gpus=8, version=1, dtype='fp16')
    self.run_test_suite(config)

  def resnet50_128_gpu_1_fp16(self):
    """Tests ResNet50 FP16 synth data data on 1 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_1.128.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=128,
        gpus=1,
        version=1,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50_128_gpu_8_fp16(self):
    """Tests ResNet50 FP16 synth data data on 8 GPU with batch size 128."""
    test_id = 'garden.resnet50.gpu_8.128.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=128,
        gpus=8,
        version=1,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50_256_gpu_1_fp16_real(self):
    """Tests ResNet50 FP16 real data data on 1 GPU with batch size 256."""
    test_id = 'garden.resnet50.gpu_1.256.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=256, gpus=1, version=1, dtype='fp16')
    self.run_test_suite(config)

  def resnet50_256_gpu_8_fp16_real(self):
    """Tests ResNet50 FP16 real data data on 8 GPU with batch size 256."""
    test_id = 'garden.resnet50.gpu_8.256.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=256, gpus=8, version=1, dtype='fp16')
    self.run_test_suite(config)

  def resnet50_256_gpu_1_fp16(self):
    """Tests ResNet50 FP16 synth data data on 1 GPU with batch size 256."""
    test_id = 'garden.resnet50.gpu_1.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=1,
        version=1,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50_256_gpu_2_fp16(self):
    """Tests ResNet50 FP16 synth data data on 2 GPU with batch size 256."""
    test_id = 'garden.resnet50.gpu_2.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=2,
        version=1,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50_256_gpu_4_fp16(self):
    """Tests ResNet50 FP16 synth data data on 4 GPU with batch size 256."""
    test_id = 'garden.resnet50.gpu_4.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=4,
        version=1,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50_256_gpu_8_fp16(self):
    """Tests ResNet50 FP16 synth data data on 8 GPU with batch size 256."""
    test_id = 'garden.resnet50.gpu_8.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=8,
        version=1,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50v2_256_gpu_1_fp16_real(self):
    """Tests ResNet50v2 FP16 real data data on 1 GPU with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_1.256.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=256, gpus=1, version=2, dtype='fp16')
    self.run_test_suite(config)

  def resnet50v2_256_gpu_2_fp16_real(self):
    """Tests ResNet50v2 FP16 real data data on 2 GPU with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_2.256.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=256, gpus=2, version=2, dtype='fp16')
    self.run_test_suite(config)

  def resnet50v2_256_gpu_4_fp16_real(self):
    """Tests ResNet50v2 FP16 real data data on 4 GPU with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_4.256.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=256, gpus=4, version=2, dtype='fp16')
    self.run_test_suite(config)

  def resnet50v2_256_gpu_8_fp16_real(self):
    """Tests ResNet50v2 FP16 real data data on 8 GPU with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_8.256.fp16.real'
    args = {}
    config = self.build_resnet_test_config(
        test_id, args, batch_size=256, gpus=8, version=2, dtype='fp16')
    self.run_test_suite(config)

  def resnet50v2_256_gpu_1_fp16(self):
    """Tests ResNet50v2 FP16 synth data data on 1 GPU with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_1.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=1,
        version=2,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50v2_256_gpu_2_fp16(self):
    """Tests ResNet50v2 FP16 synth data data on 2 GPUs with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_2.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=2,
        version=2,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50v2_256_gpu_4_fp16(self):
    """Tests ResNet50v2 FP16 synth data data on 4 GPUs with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_4.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=4,
        version=2,
        dtype='fp16',
        use_synth=True)
    self.run_test_suite(config)

  def resnet50v2_256_gpu_8_fp16(self):
    """Tests ResNet50v2 FP16 synth data data on 8 GPU with batch size 256."""
    test_id = 'garden.resnet50v2.gpu_8.256.fp16'
    args = {}
    config = self.build_resnet_test_config(
        test_id,
        args,
        batch_size=256,
        gpus=8,
        version=2,
        dtype='fp16',
        use_synth=True)
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
      help='Path to tensorflow/models repo.')
  parser.add_argument(
      '--test-list',
      type=str,
      default='resnet50v1_32_gpu_1',
      help='Comma separated list of tests to run.')
  parser.add_argument(
      '--train-data-dir',
      type=str,
      default='/data/tensorflow/imagenet',
      help='Path to training data directory.')

  FLAGS, unparsed = parser.parse_known_args()

  main()
