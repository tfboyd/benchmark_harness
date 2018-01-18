"""Run auto tests."""
from __future__ import print_function
import argparse
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()

WORKSPACE = ''
# path to store the git repos
BOOTSTRAP_LOG = ''


def run_local_command(cmd, stdout=None):
  """Run a command in a subprocess and log result.

  Args:
    cmd (str): Command to
    stdout (str, optional): File to write standard out.
  """
  if stdout is None:
    stdout = BOOTSTRAP_LOG
  f = None
  print(cmd)
  if stdout:
    f = open(stdout, 'a')
    f.write(cmd + '\n')
  for line in _run_local_command(cmd):
    if line.strip('\n'):
      print(line.strip('\n'))
      if f:
        f.write(line.strip('\n') + '\n')


def _run_local_command(cmd):
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  while True:
    retcode = p.poll()
    line = p.stdout.readline()
    yield line
    if retcode is not None:
      break


def safety_check():
  """Checks if system is open for testing.

  Watches the system periodically for 5 minutes to see if the GPUs are being
  used.

  Returns:
    True if system looks available and GPUs are free.
  """
  # Modules are added to sys.path at runtime.
  # pylint: disable=C6204
  import tools.nvidia as nvidia
  totaltime = 0
  maxtime = 300
  print('Checking if existing processes are running on GPUs for {} seconds'
        .format(maxtime))
  while True:
    if not nvidia.is_ok_to_run():
      return False
    if totaltime > maxtime:
      return True
    print('No GPU processes found watching for {} more seconds'.format(
        maxtime - totaltime))
    time.sleep(20)  # Delay for 1 minute (60 seconds).
    totaltime += 20


def git_clone(git_repo, local_folder, branch=None, sha_hash=None):
  """Clone, update, or synce a repo.

  If the clone already exists the repo will be updated via a pull.

  Args:
    git_repo (str): Command to
    local_folder (str): Where to clone repo into.
    branch (str, optional): Branch to checkout.
    sha_hash (str, optional): Hash to sync to.
  """
  if FLAGS.git_clone:
    if os.path.isdir(local_folder):
      git_clone_or_pull = 'git -C {} pull'.format(local_folder)
    else:
      git_clone_or_pull = 'git clone {} {}'.format(git_repo, local_folder)
    run_local_command(git_clone_or_pull)

    if branch is not None:
      branch_cmd = 'git -C {} checkout {}'.format(local_folder, branch)
      run_local_command(branch_cmd)

    if sha_hash is not None:
      sync_to_hash_cmd = 'git -C {} reset --hard {}'.format(
          local_folder, sha_hash)
      run_local_command(sync_to_hash_cmd)
  else:
    print(
        'Repo({}) not cloned or updated. Pass --git-clone=True to clone'.format(
            git_repo))


def run_tests():
  """Builds and runs docker image with specified test config."""
  docker_save_name = FLAGS.docker_save_tag

  # do a fresh build of the docker images
  docker_build = 'docker build --pull -t {} {}'.format(docker_save_name,
                                                       FLAGS.docker_folder)
  run_local_command(docker_build)

  # kick off the tests via docker
  run_benchmarks = (
      'nvidia-docker run --rm -v {}:/auth_tokens -v {}:/workspace {} python '
      '/workspace/git/benchmark_harness/oss_bench/harness/controller.py '
      '--workspace=/workspace --test-config={} --framework={}')
  run_benchmarks = run_benchmarks.format(FLAGS.auth_token_dir, FLAGS.workspace,
                                         docker_save_name, FLAGS.test_config,
                                         FLAGS.framework)

  run_local_command(run_benchmarks)


def main():
  global WORKSPACE, BOOTSTRAP_LOG
  WORKSPACE = FLAGS.workspace
  BOOTSTRAP_LOG = './log.txt'

  # Adding 'package' to the system path to make modules available and simplify
  # usage by not having to 'use python -m'.
  sys.path.append('../')

  if FLAGS.gpu_process_check:
    if not safety_check():
      print('Existing GPU processes were running aborting test run!!!')
      return

  git_workspace = os.path.join(FLAGS.workspace, 'git')
  git_clone(
      'https://github.com/tfboyd/benchmark_harness.git',
      os.path.join(git_workspace, 'benchmark_harness'),
      branch=FLAGS.harness_branch)

  if FLAGS.framework == 'tensorflow' or FLAGS.framework == 'mxnet':
    run_tests()
  else:
    raise ValueError('FLAGS.framework is an unknown value{}'.format(
        FLAGS.framework))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--gpu-process-check',
      type=bool,
      default=False,
      help='Set to true to not run if there are active GPU processes.')
  parser.add_argument(
      '--harness-branch',
      type=str,
      default=None,
      help='Set to harness branch to use, used for testing harness.')
  parser.add_argument(
      '--docker-folder',
      type=str,
      default='./docker/nightly_gpu',
      help='Folder with docker build file.')
  parser.add_argument(
      '--docker-save-tag',
      type=str,
      default='tobyboyd/tf-gpu',
      help='Name and optional tag for docker image created by build.')
  parser.add_argument(
      '--workspace',
      type=str,
      default='/usr/local/google/home/tobyboyd/auto_run_play',
      help='Workspace that will be mounted to the docker.')
  parser.add_argument(
      '--auth-token-dir',
      type=str,
      default='/auth_tokens',
      help='Directory with service authentication tokens mounted to docker at '
      '/auth_tokens')
  parser.add_argument(
      '--test-config',
      type=str,
      default='configs/dev/default.yaml',
      help=
      'Absolute Path to the test_config as mounted on docker or default to run '
      'default config.')
  parser.add_argument(
      '--framework',
      type=str,
      default='tensorflow',
      help='Framework to be tested.')
  parser.add_argument(
      '--git-clone',
      type=bool,
      default=False,
      help='If true repos will be automatically updated, which for some '
      'systems is a needless security risk.')
  FLAGS, unparsed = parser.parse_known_args()
  main()
