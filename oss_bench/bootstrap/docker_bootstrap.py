"""Run auto tests."""
from __future__ import print_function
import argparse
import os
import subprocess

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


def main():
  global WORKSPACE, BOOTSTRAP_LOG
  WORKSPACE = FLAGS.workspace
  BOOTSTRAP_LOG = './log.txt'
  git_clone('https://github.com/tfboyd/benchmark_harness.git',
            os.path.join(FLAGS.workspace, 'harness'))

  docker_base = 'tensorflow/tensorflow:latest-gpu'
  docker_test = 'tobyboyd/tf-gpu'

  # update docker pull
  docker_pull = 'docker pull {}'.format(docker_base)
  run_local_command(docker_pull)

  # do a fresh build of the docker images
  docker_build = 'docker build -t {} ./docker/'.format(docker_test)
  run_local_command(docker_build)

  # kick off the tests via docker
  run_benchmarks = (
      'nvidia-docker run --rm -v {}:/auth_tokens -v {}:/workspace {}'
      ' python /workspace/harness/oss_bench/harness/controller.py '
      '--workspace=/workspace --test-config={}')
  run_benchmarks = run_benchmarks.format(FLAGS.auth_token_dir, FLAGS.workspace,
                                         docker_test, FLAGS.test_config)

  run_local_command(run_benchmarks)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default='/usr/local/google/home/tobyboyd/auto_run_play',
      help='Workspace that will be mounted to the docker.')
  parser.add_argument(
      '--auth-token-dir',
      type=str,
      default='',
      help='Directory with service authentication tokens mounted to docker at '
      '/auth_tokens')
  parser.add_argument(
      '--test-config',
      type=str,
      default='configs/default.yaml',
      help=
      'Absolute Path to the test_config as mounted on docker or default to run '
      'default config.')
  parser.add_argument(
      '--git-clone',
      type=bool,
      default=False,
      help='If true repos will be automatically updated, which for some '
      'systems is a needless security risk.')
  FLAGS, unparsed = parser.parse_known_args()
  main()
