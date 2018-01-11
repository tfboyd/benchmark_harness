"""Build command line to execute tf_cnn_benchmarks."""
from __future__ import print_function


def BuildDistributedCommandWorker(run_config):
  """Build command to start distributed worker."""

  run_script = 'python tf_cnn_benchmarks.py'

  # Prepends environment variables
  if 'env_vars' in run_config:
    run_script = '{} {}'.format(run_config['env_vars'], run_script)

  # Build command line
  run_cmd_list = []

  # Arguments with no name change that are passed through from the configs.
  pass_through_args = [
      'data_format', 'batch_size', 'num_batches', 'model', 'data_dir',
      'optimizer', 'learning_rate', 'sync_on_finish', 'weight_decay',
      'data_name', 'variable_update', 'num_intra_threads', 'num_inter_threads',
      'mkl', 'num_warmup_batches', 'forward_only', 'kmp_blocktime', 'device',
      'staged_vars', 'staged_grads', 'cross_replica_sync', 'all_reduce_spec',
      'use_datasets', 'batch_group_size', 'use_nccl', 'use_fp16'
  ]

  for arg in pass_through_args:
    if arg in run_config:
      run_cmd_list.append('--{}={}'.format(arg, run_config[arg]))

  if 'ps_server' in run_config:
    run_cmd_list.append('--local_parameter_device={}'.format(
        run_config['ps_server']))

  if 'gpus' in run_config:
    run_cmd_list.append('--num_gpus={}'.format(run_config['gpus']))

  # Forces no distortions, which is the most common for benchmarks.
  run_cmd_list.append('--nodistortions')
  if 'display_every' in run_config:
    run_cmd_list.append('--display_every={}'.format(
        run_config['display_every']))
  else:
    run_cmd_list.append('--display_every=10')

  # Creates a trace file for each model+gpu combo
  if 'trace_file' in run_config:
    trace_file = '{}_{}'.format(run_config['trace_file'], run_config['model'])
    if 'gpus' in run_config:
      trace_file = trace_file + '_' + str(run_config['gpus']) + '.txt'
    else:
      trace_file += '.txt'
    run_cmd_list.append('--trace_file=' + trace_file)

  run_cmd = '{} {}'.format(run_script, ' '.join(run_cmd_list))

  return run_cmd


def GpuDecode(raw_gpu_input):
  """Handles different entries options for workers and ps_servers."""
  if type(raw_gpu_input) is int:
    return str(raw_gpu_input)
  else:
    return raw_gpu_input.split(',')


def LoadYamlRunConfig(full_config, debug_level):
  """Processes config file into list of configs.

  Reads the config made up of repeating 'run_configs'  The first first config as
  is treated as the base. Each config entry after the first is merged with the
  base (first) config.  The idea being the first config is the base and the
  subsequent configs are variations that override the base config

  Additionally, multiple configs are created based on the the following fields:
  'model' (list of models to test), gpu (list of number of GPUs to test), and
  repeat (number of times to run the test).

  Args:
    full_config: full run_config normally loaded from yaml
    debug_level: controls level of output
  """

  # base config that subsequent configs merge with
  base_config = None
  suite = []
  for config in full_config['run_configs']:
    test_configs = []
    suite.append(test_configs)
    if base_config is None:
      base_config = config
    else:
      # merges config with base config
      base = base_config.copy()
      base.update(config)
      config = base

    # Copy root settings into config.  Anything
    # at the root with override anything in run_configs
    for k, v in full_config.iteritems():
      if k != 'run_configs':
        base_config[k] = v

    if config.get('repeat') is not None:
      repeat = int(config['repeat'])
      for i in range(repeat):
        # Creates copy so each one can have an index, e.g. 'copy'
        repeat_model_config = config.copy()
        repeat_model_config['copy'] = i
        test_configs.append(repeat_model_config)
    else:
      test_configs.append(config)
    if debug_level > 0:
      print('Config:{} \n{}'.format(config['test_id'], config))
  return suite
