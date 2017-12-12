"""Runs benchmarks on various cloud or local systems."""
import argparse
import datetime
import os
import StringIO
import tarfile
import time
#import cluster
#import cluster_gce
import cluster_local
#import cluster_ssh
import command_builder
import reporting
import util
import yaml
import pwd

# System variables
PS_PORT = 50000
WORKER_PORT = 50001

class TestRunner(object):
  """Run benchmark tests and record results.

  Args:
    configs (str): 
    workspace (str): 
    bench_home (str): 
    ssh_key (str, optional):
    username (str, optional): 
    tf_url (str, optional):
    mount (Boolean, optional):
    sudo (Boolean, optional):
    action (str, optional):

  """
  def __init__(self,
              configs,
              workspace,
              bench_home,
              auto_test_config=None,
              ssh_key=None,
              username=None,
              tf_url = None,
              mount = None,
              sudo=False,
              action=None,
              debug_level=1):
    """Initalize the TestRunner with values."""
    self.auto_test_config = auto_test_config
    self.configs = configs
    self.workspace = workspace
    self.local_local_dir = os.path.join(self.workspace, 'logs')
    self.local_stdout_file = os.path.join(self.local_local_dir, 'stdout.log')
    self.local_stderr_file = os.path.join(self.local_local_dir, 'stderr.log')
    self.bench_home = bench_home
    self.sudo = sudo
    self.action = action
    self.ssh_key = ssh_key
    self.tf_url = tf_url
    self.debug_level = debug_level

    # Creates workspace and default log folder
    if not os.path.exists(self.local_local_dir):
      print('Making log directory:{}'.format(self.local_local_dir))
      os.makedirs(self.local_local_dir)

    # Set global username
    self.username = 'ubuntu'

    print('Using {} as username to access remote hosts.'.format(self.username))


  def setup_servers(self, instance, full_config):
    """Setup remote servers with tools and software to run tests

    Args:
      instance: instance object representing the server
    """
    print('Setting up instance: {}'.format(instance.instance_id))
    self.kill_running_processes(instance)

    if self.tf_url is not None or 'tf_url' in full_config:
      # Command line overrides config tf_url
      tf_url = self.tf_url if self.tf_url else full_config['tf_url']
      self.install_tensorflow(instance, tf_url)


  def kill_running_processes(self, instance):
    """Kill benchmark processes for a clean start

    This is useful if there is a need to stop in the middle.  The ps_servers, for
    example will not always stop.  This kills all processes related to the 
    benchmark run owned by the user.

    Args:
      instance: instance object representing the server

    """
    sudo = ''
    if self.sudo:
      sudo = 'sudo '

    cmd = '{}pkill -f "python tf_cnn" -u {}'.format(sudo, self.username)
    instance.ExecuteCommandAndStreamOutput(
        cmd,
        self.local_stdout_file,
        self.local_stderr_file,
        util.ExtractToStdout,
        print_error=True,
        ok_exit_status=[0, 1, -1])


  def install_tensorflow(self, instance, tf_path):
    """Installs the ensorflow pip package located at the tf_url passed in.
    
    pip command is executed directly on the tf_url so it will also install
    from PyPI if that "tensorflow" is used as tf_url rather than a link to
    s3 or some other http style location.  

    Args:
      instance: instance object representing the server
      tf_path: url or path to the tensorflow to install

    """
    print('{}: Installing tensorflow:{}'.format(instance.instance_id, tf_path))
    sudo = ''
    if self.sudo:
      sudo = 'sudo '
    # --force-reinstall creates issues on occasion
    cmd = '{}pip install --quiet --upgrade {}'.format(sudo, tf_path)
    # Error will print if install fails
    t = instance.ExecuteCommandInThread(
        cmd,
        self.local_stdout_file,
        self.local_stderr_file,
        util.ExtractToStdout,
        print_error=True)
    t.join()
    print('{}: Tensorflow installed'.format(instance.instance_id))


  def results_directory(self, run_config):
    """Determine and create the results directory

    Args:
      run_config: Config representing the test to run.  
    """
    suite_dir_name = '{}_{}'.format(run_config['test_suite_start_time'], run_config['test_id'])
    datetime_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    test_result_dir = '{}'.format(datetime_str)
    result_dir = os.path.join(self.workspace, 'results', suite_dir_name, test_result_dir)

    # Creates workspace and default log folder
    if not os.path.exists(result_dir):
      print('Making results directory:{}'.format(result_dir))
      os.makedirs(result_dir)

    return result_dir


  def run_benchmark(self, run_config, instances):
    """Run single distributed tests for the passed config

    Args:
      run_config: Config representing the test to run.
      instances: Instances to run the tests against.
    """
    # Timestamp and other values added for reporting
    run_config['timestamp'] = int(time.time())
    run_config['workspace'] = self.workspace

    worker_list = map(
        int, command_builder.WorkerUtil(run_config['workers']).split(','))
    if 'ps_servers' in run_config:
      ps_list = map(
          int, command_builder.WorkerUtil(run_config['ps_servers']).split(','))
    else:
      ps_list = []
    assert len(worker_list) > 0

    test_home = self.bench_home
    result_dir = self.results_directory(run_config)

    # Write config to results folder
    config_file_out = os.path.join(result_dir, 'config.yaml')
    config_out = open(config_file_out, 'w')
    config_out.write(yaml.dump(run_config))
    config_out.close()

    ps_hosts = ','.join(
        ['%s:%d' % (instances[i].hostname, PS_PORT) for i in ps_list])
    worker_hosts = ','.join(
        ['%s:%d' % (instances[i].hostname, WORKER_PORT) for i in worker_list])

    ps_threads = []
    for i, ps in enumerate(ps_list):
      cmd = command_builder.BuildDistributedCommandPS(run_config, worker_hosts,
                                                      ps_hosts, i)
      cmd = 'cd {}; {}'.format(test_home, cmd)
      print('[{}] ps_server | Run benchmark({}):{}'.format(
          run_config.get('copy', '0'), run_config['test_id'], cmd))

      stdout_file = os.path.join(result_dir, 'ps_%d_stdout.log' % i)
      stderr_file = os.path.join(result_dir, 'ps_%d_stderr.log' % i)
      # Starts the ps server on the instance matching the index in the list.
      # Which allows control over which servers are ps servers.
      ps_threads.append(instances[int(ps)].ExecuteCommandInThread(
          cmd, stdout_file, stderr_file, util.ExtractErrorToConsole))

    worker_threads = []
    for i, worker in enumerate(worker_list):
      cmd = command_builder.BuildDistributedCommandWorker(
          run_config, worker_hosts, ps_hosts, i)
      cmd = 'cd {}; {}'.format(test_home, cmd)
      print('[{}] worker | Run benchmark({}):{}'.format(
          run_config.get('copy', '0'), run_config['test_id'], cmd))
      stdout_file = os.path.join(result_dir, 'worker_%d_stdout.log' % i)
      stderr_file = os.path.join(result_dir, 'worker_%d_stderr.log' % i)
      if i == 0:
        t = instances[worker].ExecuteCommandInThread(
            cmd, stdout_file, stderr_file, util.ExtractToStdout, print_error=True)
      else:
        t = instances[worker].ExecuteCommandInThread(
            cmd,
            stdout_file,
            stderr_file,
            util.ExtractErrorToConsole,
            print_error=True)
      worker_threads.append(t)

    for t in worker_threads:
      t.join()

    return result_dir


  def run_test_suite(self, full_config, instances):
    """Run distributed benchmarks

    Args:
      configs: Configs representing the tests to run.
      instances: Instances to run the tests against.

    """
    # Folder to store suite results
    full_config['test_suite_start_time'] =  datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    
    # Configs for the test suite
    test_suite = command_builder.LoadYamlRunConfig(full_config, self.debug_level)

    # Setup instances with code and install TensorFlow if desired
    for instance in instances:
      self.setup_servers(instance, full_config)

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
            result_dir = self.run_benchmark(test_config, instances)
            oom = reporting.check_oom(
                os.path.join(result_dir, 'worker_0_stdout.log'))
            if oom and next_val < lowest_oom:
              lowest_oom = next_val
            low, high, next_val = reporting.oom_batch_size_search(
                low, high, next_val, oom)
            print 'Lowest OOM Value:{}'.format(lowest_oom)
        else:
          result_dir = self.run_benchmark(test_config, instances)

        # Cleans up by killing proceses and closing ssh cilents
        for instance in instances:
          # Kills running processes but seems to create more issues
          # sometimes than just killing the ssh sessions.
          self.kill_running_processes(instance)
          instance.CleanSshClient()

        # Wait for 15 seconds just to let things settle
        print 'Waiting 15 seconds for services to shutdown.'
        time.sleep(5)
      suite_dir_name = '{}_{}'.format(last_config['test_suite_start_time'], last_config['test_id'])
      reporting.process_results_folder(
          os.path.join(self.workspace, 'results', suite_dir_name),report_config=self.auto_test_config)


  def local_benchmarks(self, full_config):
    """ Run Local benchmark tests

    """
    print('Running Local Benchmarks')

    with cluster_local.UseLocalInstances(
        virtual_env_path=full_config.get('virtual_env_path')) as instances:
      self.run_test_suite(full_config, instances)

  def reset_select_attributes(self, full_config):
    """Reset select attributes between configs.
    
    This is a hack to reset some attributes between runs that is a remnant
    before encapsulating the code into a class.

    """
    if full_config.get('cloud_type') in ['gce', 'ssh', 'local'
                                        ] and self.username == 'ubuntu':
      # os.getlogin does not work in docker containers
      self.username = pwd.getpwuid(os.getuid())[0]
    else:
      self.username = username

    if full_config.get('cloud_type') in ['aws', 'gce']:
      self.sudo  = True
    else:
      self.sudo  = False

  def load_yaml_configs(self, config_paths, base_dir=None):
    """Convert string of config paths into list of yaml objects

      If configs_string is empty a list with a single empty object is returned
    """
    configs = []
    for _, config_path in enumerate(config_paths):
      if base_dir is not None:
        config_path = os.path.join(base_dir,config_path) 
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
    for i, global_config in enumerate(configs):
      base_dir = os.path.dirname(global_config['config_path'])
      sub_configs = self.load_yaml_configs(global_config['sub_configs'], base_dir=base_dir)
      for j, run_config in enumerate(sub_configs):
        full_config = run_config.copy()

        # Override config with global config, used to change projects
        # or any other field, these values will also end up overriding
        # settings in individual 'run_configs'
        if global_config:
          for k, v in global_config.iteritems():
            if k != 'run_configs':
              full_config[k] = v

        # Initialize global variables and setup the workspace
        self.reset_select_attributes(full_config)

        if self.action is None:
          self.local_benchmarks(full_config)
        else:
          if self.action == 'results':
            reporting.process_results_folder(FLAGS.results_folder)
          else:
            print('Action unknown, doing nothing:{}'.format(self.action))


def Main():
  """Program main, called after args are parsed into FLAGS."""
  test_runner = TestRunner(FLAGS.config,
                        FLAGS.workspace,
                        FLAGS.tf_cnn_bench_dir,
                        ssh_key = FLAGS.ssh_key,
                        username = FLAGS.username,
                        tf_url = None if FLAGS.tf_url == '' else FLAGS.tf_url,
                        mount = FLAGS.mount,
                        sudo = False if FLAGS.sudo == '' else FLAGS.sudo,
                        action = None if FLAGS.action == '' else FLAGS.action)
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
      '--tf_url',
      type=str,
      default='',
      help='If set this version of TensorFlow is installed')
  parser.add_argument(
      '--ssh_key',
      type=str,
      default='.ssh/tf_perf_aws.pem',
      help='Set to ssh_key path relative to users home folder')
  parser.add_argument(
      '--debug_level',
      type=int,
      default=1,
      help='Set to debug level: 0, 1, 5. Default 1')
  parser.add_argument(
      '--mount',
      type=str,
      default=True,
      help='Set to '
      ' to not mount data drives')
  parser.add_argument(
      '--bench_home',
      type=str,
      default=os.path.join(os.environ['HOME'], 'tf_cnn_bench'),
      help='Path to the benchmark scripts')
  # Less used Flags
  parser.add_argument(
      '--action',
      type=str,
      default='',
      help='If set this action is taken rather than run the entire script')
  parser.add_argument(
      '--results_folder',
      type=str,
      default='',
      help='full or relative path to local result to process')
  parser.add_argument(
      '--username',
      type=str,
      default='ubuntu',
      help='Username for ssh sessions to remote hosts')
  parser.add_argument(
      '--sudo',
      type=str,
      default='',
      help='If true, sudo can be used if False sudo cannot be used.')

  FLAGS, unparsed = parser.parse_known_args()

  Main()
