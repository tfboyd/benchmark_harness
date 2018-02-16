"""Build command line to execute tf_cnn_benchmarks."""
from __future__ import print_function


def build_run_command(run_config):
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
      'use_datasets', 'batch_group_size', 'use_nccl', 'use_fp16',
      'nodistortions', 'gpu_thread_mode', 'hierarchical_copy', 'use_tf_layers'
  ]

  for arg in pass_through_args:
    if arg in run_config:
      if run_config[arg] == 'FLAG_ONLY':
        run_cmd_list.append('--{}'.format(arg))
      else:
        run_cmd_list.append('--{}={}'.format(arg, run_config[arg]))

  if 'ps_server' in run_config:
    run_cmd_list.append('--local_parameter_device={}'.format(
        run_config['ps_server']))

  if 'gpus' in run_config:
    run_cmd_list.append('--num_gpus={}'.format(run_config['gpus']))

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


def build_test_config_suite(full_config, debug_level):
  """Processes config file into list of configs.

  Reads the config made up of repeating 'run_configs'  The first first
  `run_config ` is treated as the base. Each config entry after the first is
  merged with the base (first) config.  The idea being the first config is
  the base and the subsequent configs are variations that override the base
  config.  The configs at the root level of the `full_config` are copied
  into each `run_config` overwriting and appending the values in `run_config`.
  This makes it possible to set `data_dir` at the top level and make all tests
  use real data without having to write all new test files.

  A copy of each config is created based on number of repeats indicated in the
  `repeat` field.

  Args:
    full_config: full run_config normally loaded from yaml
    debug_level: controls level of output

  Returns:
    Array of arrays containing test_configs where each test config in the
    array is the same test_id with each config (copy) representing an iteration
    to run. For example: suite[0] could contain 3 config objects all for test_id
    resnet50.gpu_1 indicting the test is to be run 3 times exactly the same way.
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

    # Copies fields, with the exception of `run_configs` from the root config
    # into the run_config overwriting and supplementing fields in the
    # run_config.
    for k, v in full_config.iteritems():
      if k != 'run_configs':
        config[k] = v

    # Sloppy hack to avoid having real and synthetic test names
    if 'data_dir' in config and 'test_id_hacks' in config:
      config['test_id'] = '{}.real'.format(config['test_id'])

    # Creates a copy for each time the test is to be run.
    repeat = int(config['repeat'])
    for i in range(repeat):
      repeat_model_config = config.copy()
      repeat_model_config['copy'] = i
      test_configs.append(repeat_model_config)

    if debug_level > 0:
      print('Config:{} \n{}'.format(config['test_id'], config))
  return suite
