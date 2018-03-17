"""Bootstrap that starts up docker and starts off desired test harness."""
from __future__ import print_function
import argparse
import os
import subprocess
import sys
import time
import yaml

parser = argparse.ArgumentParser()


class Bootstrap(object):
  """Builds docker image and starts test harness within the image.

  Args:
    docker_folder:
    workspace:
    test_config:
    bootstrap_config:
    docker_tag:
    auth_token_dir:
    harness_branch:
    framework:
    gpu_process_check:
    pure_docker:

  """

  def __init__(self,
               docker_folder,
               workspace,
               test_config,
               bootstrap_config=None,
               docker_tag=None,
               auth_token_dir=None,
               harness_branch=None,
               framework='tensorflow',
               gpu_process_check=True,
               pure_docker=False):
    self.docker_folder = docker_folder
    self.workspace = workspace
    self.test_config = test_config
    self.bootstrap_config = bootstrap_config
    self.bootstrap_log = './log.txt'
    self.auth_token_dir = auth_token_dir
    self.harness_branch = harness_branch
    self.framework = framework
    self.gpu_process_check = gpu_process_check
    self.git_workspace = os.path.join(workspace, 'git')
    self.pure_docker = pure_docker
    self.docker_tag = docker_tag

  def run_local_command(self, cmd, stdout=None):
    """Run a command in a subprocess and log result.

    Args:
      cmd (str): Command to
      stdout (str, optional): File to write standard out.
    """
    if stdout is None:
      stdout = self.bootstrap_log
    f = None
    print(cmd)
    if stdout:
      f = open(stdout, 'a')
      f.write(cmd + '\n')
    for line in self._run_local_command(cmd):
      if line.strip('\n'):
        print(line.strip('\n'))
        if f:
          f.write(line.strip('\n') + '\n')

  def _run_local_command(self, cmd):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    while True:
      retcode = p.poll()
      line = p.stdout.readline()
      yield line
      if retcode is not None:
        break

  def existing_process_check(self):
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
      time.sleep(20)
      totaltime += 20

  def git_clone(self, git_repo, local_folder, branch=None, sha_hash=None):
    """Clone, update, or synce a repo.

    If the clone already exists the repo will be updated via a pull.

    Args:
      git_repo (str): Command to
      local_folder (str): Where to clone repo into.
      branch (str, optional): Branch to checkout.
      sha_hash (str, optional): Hash to sync to.
    """
    if os.path.isdir(local_folder):
      git_clone_or_pull = 'git -C {} pull'.format(local_folder)
    else:
      git_clone_or_pull = 'git clone {} {}'.format(git_repo, local_folder)
    self.run_local_command(git_clone_or_pull)

    if branch is not None:
      branch_cmd = 'git -C {} checkout {}'.format(local_folder, branch)
      self.run_local_command(branch_cmd)

    if sha_hash is not None:
      sync_to_hash_cmd = 'git -C {} reset --hard {}'.format(
          local_folder, sha_hash)
      self.run_local_command(sync_to_hash_cmd)

  def load_config(self, config_path):
    """Loads yaml file from path and returns dict."""
    if config_path:
      with open(config_path) as config:
        return yaml.safe_load(config)
    else:
      config = {}
      config['mount_point'] = []
      return config

  def build_docker_mounts(self, config):
    """Returns args to mount drives via docker a string."""
    mount_str = ''
    for mount in config['mount_point']:
      mount_str += ' -v {}:{}'.format(mount['folder_path'],
                                      mount['docker_path'])
    return mount_str

  def build_docker_cmd(self, config, docker_image):
    """Returns docker command to start docker and kick off tests."""
    mounts = self.build_docker_mounts(config)
    docker = 'nvidia-docker'
    if self.pure_docker:
      docker = 'docker'

    extra_args = ''
    # PyTorch needs this to function due to shared memory.
    if self.framework == 'pytorch':
      extra_args = '--ipc=host'

    run_cmd = (
        '{} run {} --rm {} {} python '
        '/workspace/git/benchmark_harness/oss_bench/harness/controller.py '
        '--workspace=/workspace --test-config={} --framework={}')
    run_cmd = run_cmd.format(docker, extra_args, mounts, docker_image,
                             self.test_config, self.framework)
    return run_cmd

  def run_tests(self):
    """Builds and runs docker image with specified test config."""
    if self.gpu_process_check:
      if not self.existing_process_check():
        print('Existing GPU processes were running, aborting test run!!!')
        return

    # Pulls down the harness to run in the docker.
    self.git_clone(
        'https://github.com/tfboyd/benchmark_harness.git',
        os.path.join(self.git_workspace, 'benchmark_harness'),
        branch=self.harness_branch)

    # Build latest docker image.
    # Build with --no-cache as some Dockerfile have pip installs and the rest of
    # the docker may not be changing.
    docker_build = 'docker build --no-cache --pull -t {} {}'.format(
        self.docker_tag, self.docker_folder)
    self.run_local_command(docker_build)

    # Builds docker command, starts docker, and executes the command.
    bootstrap_config = self.load_config(self.bootstrap_config)
    auth_tokens_mnt = dict([('folder_path', self.auth_token_dir),
                            ('docker_path', '/auth_tokens')])
    workspace_mnt = dict([('folder_path', self.workspace), ('docker_path',
                                                            '/workspace')])

    bootstrap_config['mount_point'].append(auth_tokens_mnt)
    bootstrap_config['mount_point'].append(workspace_mnt)

    run_benchmarks = self.build_docker_cmd(bootstrap_config, self.docker_tag)

    self.run_local_command(run_benchmarks)


def main():

  # Adding 'package' to the system path to make modules available and simplify
  # usage by not having to 'use python -m'.
  sys.path.append('../')

  bootstrap = Bootstrap(
      FLAGS.docker_folder,
      FLAGS.workspace,
      FLAGS.test_config,
      FLAGS.bootstrap_config,
      docker_tag=FLAGS.docker_save_tag,
      framework=FLAGS.framework,
      auth_token_dir=FLAGS.auth_token_dir,
      harness_branch=FLAGS.harness_branch,
      gpu_process_check=FLAGS.gpu_process_check,
      pure_docker=FLAGS.pure_docker)
  bootstrap.run_tests()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--gpu-process-check',
      type=bool,
      default=True,
      help='Set to true to not run if there are active GPU processes.')
  # Allows gpu-process-check to be turned off.
  parser.add_argument(
      '--no-gpu-process-check', dest='gpu_process_check', action='store_false')
  parser.add_argument(
      '--pure-docker',
      type=bool,
      default=False,
      help='Set to true to not use nvidia-docker')
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
      '--bootstrap-config',
      type=str,
      default=None,
      help='Path to local config for the bootstrap.  Mostly for mounting data.')
  parser.add_argument(
      '--framework',
      type=str,
      default='tensorflow',
      help='Framework to be tested.')
  FLAGS, unparsed = parser.parse_known_args()
  main()
