"""Run auto tests."""
from subprocess import call
import subprocess
import argparse
import os
import sys
import yaml

parser = argparse.ArgumentParser()

# Globals set in the main.
WORKSPACE = ''
# Git repo and log paths.
git_repo_base = ''
LOGS_DIR = ''


def run_local_command(cmd, stdout=None):
  """Run a command in a subprocess and log result.

  Args:
    cmd (str): Command to
    stdout (str, optional): File to write standard out.
  """
  if stdout is None:
    stdout=os.path.join(LOGS_DIR, 'log.txt')
  print cmd
  f = None
  if stdout:
    f = open(stdout, 'a')
    f.write(cmd + '\n')
  for line in _run_local_command(cmd):
    if (line.strip('\n')):
      print(line.strip('\n'))
      if f:
        f.write(line.strip('\n') + '\n')


def _run_local_command(cmd):
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  while (True):
    retcode = p.poll()  #returns None while subprocess is running
    line = p.stdout.readline()
    yield line
    if (retcode is not None):
      break

def _git_clone(git_repo, local_folder, branch=None, sha_hash=None):

  if os.path.isdir(local_folder):
    git_clone_or_pull = 'git -C {} pull'.format(local_folder)
  else:
    git_clone_or_pull = 'git clone {} {}'.format(git_repo, local_folder)
  run_local_command(git_clone_or_pull)

  if branch is not None:
    branch_cmd = 'git -C {} checkout {}'.format(local_folder, branch)
    run_local_command(branch_cmd)

  if sha_hash is not None:
    sync_to_hash_cmd = 'git -C {} reset --hard {}'.format(local_folder, sha_hash)
    run_local_command(sync_to_hash_cmd)

def _call_tf_benchmarks_tests(auto_config):
  import test_runners.tf_cnn_bench.run_benchmark as run_benchmark
  rel_config_paths = auto_config['tf_cnn_bench_configs']
  config_paths = []
  for rel_config_path in rel_config_paths:
    config_paths.append(os.path.join(GIT_REPO_BASE, rel_config_path))

  config = ','.join(config_paths)

  tf_cnn_bench_path = os.path.join(GIT_REPO_BASE,
                        'benchmarks/scripts/tf_cnn_benchmarks')
  
  test_runner = run_benchmark.TestRunner(config,
                        os.path.join(LOGS_DIR, 'tf_cnn_workspace'),
                        tf_cnn_bench_path,
                        auto_test_config=auto_config)
  test_runner.run_tests()

def _load_config():
  """Returns auto_run config for the environment."""
  config_path = None
  if FLAGS.test_config == 'default':
    config_path = os.path.join(os.path.dirname(__file__), 'configs/default.yaml')
  else:
    config_path = FLAGS.test_config
  f = open(config_path)
  return yaml.safe_load(f)


def _clone_repos():
  _git_clone('https://github.com/tensorflow/benchmarks.git',
              os.path.join(GIT_REPO_BASE, 'benchmarks'),
              sha_hash='267d7e81977f23998078f39afd48e9a97c3acf5a')
  _git_clone('https://github.com/tfboyd/benchmark_harness.git',
              os.path.join(GIT_REPO_BASE, 'benchmark_harness'))


def main():
  global WORKSPACE, GIT_REPO_BASE, LOGS_DIR
  WORKSPACE = FLAGS.workspace
  GIT_REPO_BASE = os.path.join(WORKSPACE, 'git')
  LOGS_DIR = os.path.join(WORKSPACE, 'logs')
  try: 
    os.makedirs(LOGS_DIR)
  except OSError:
    if not os.path.isdir(LOGS_DIR):
      raise
  test_config = _load_config()
  print test_config

  os.environ[
      'GOOGLE_APPLICATION_CREDENTIALS'] = '/service_account_auth_tokens/tensorflow_performance_upload_tb.json'

  # pick a directory, download tfboyd for auto_run and then tf_cnn_benchmarks
  # then kick off some tests via auto_run.  
  _clone_repos()


  # Upgrade SAVE script to match what Anjili and I discussed.

  # run_benchmark code should just need to know the test workspace and then
  # it could find the code via the git paths to tf_cnn_benchmark.  

  # Modify the python path for the libraries for the tests to run and then
  # import them.
  git_python_lib_paths = ['benchmark_harness/oss_bench']  
  for lib_path in git_python_lib_paths:
    sys.path.append(os.path.join(GIT_REPO_BASE, lib_path))
  _call_tf_benchmarks_tests(test_config)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--debug_level',
      type=int,
      default=1,
      help='Set to debug level: 0, 1, 5. Default 1')

  parser.add_argument(
      '--test_config',
      type=str,
      default='default',
      help='Path to the test_config or default to run default config') 

  parser.add_argument(
      '--workspace',
      type=str,
      default='/workspace',
      help='Path to the workspace') 

  FLAGS, unparsed = parser.parse_known_args()

  main()