"""Run auto tests."""
from subprocess import call
import subprocess
import argparse
import os
import sys
import yaml

parser = argparse.ArgumentParser()

workspace = '/workspace'
# path to store the git repos
logs_dir = os.path.join(workspace, 'logs')


def run_local_command(cmd, stdout=os.path.join(logs_dir, 'log.txt')):
  """Run a command in a subprocess and log result.

  Args:
    cmd (str): Command to
    stdout (str, optional): File to write standard out.
  """
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
  run_local_command(git_clone_or_pull, stdout=None)

  if branch is not None:
    branch_cmd = 'git -C {} checkout {}'.format(local_folder, branch)
    run_local_command(branch_cmd, stdout=None)

  if sha_hash is not None:
    sync_to_hash_cmd = 'git -C {} reset --hard {}'.format(local_folder, sha_hash)
    run_local_command(sync_to_hash_cmd, stdout=None)


def main():
  global workspace
  workspace = FLAGS.workspace
  _git_clone('https://github.com/tfboyd/benchmark_harness.git',
              os.path.join(FLAGS.workspace, 'harness'))

  docker_base = 'tensorflow/tensorflow:latest-gpu'
  docker_test = 'tobyboyd/tf-gpu'

  # update docker pull
  docker_pull = 'docker pull {}'.format(docker_base)
  run_local_command(docker_pull, stdout=None)

  # do a fresh build of the docker images
  docker_build = 'docker build -t {} ./docker/'.format(docker_test)
  run_local_command(docker_build, stdout=None)

  # kick off the tests via docker
  run_benchmarks = ('nvidia-docker run --rm' +
            ' -v /usr/local/google/home/tobyboyd/service_account_auth_tokens:/service_account_auth_tokens' +
            ' -v {}:/workspace {}' +
            ' python /workspace/harness/oss_bench/harness/test_controller.py')
  run_benchmarks = run_benchmarks.format(FLAGS.workspace, docker_test, workspace)
  print run_benchmarks

  run_local_command(run_benchmarks, stdout=None)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--workspace',
      type=str,
      default='/usr/local/google/home/tobyboyd/auto_run_play',
      help='Workspace that will be mounted to the docker.') 

  parser.add_argument(
      '--test_config',
      type=str,
      default='default',
      help='Path to the test_config or default to run default config') 

  FLAGS, unparsed = parser.parse_known_args()

  main()