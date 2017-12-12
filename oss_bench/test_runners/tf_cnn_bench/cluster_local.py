"""Cluster implementation that runs locally without SSH."""
import os
import time
import util
from contextlib import contextmanager
from subprocess import call
import threading
import subprocess


class LocalInstance(object):

  def __init__(self, host, virtual_env_path=''):
    self.hostname = host
    self.virtual_env_path = virtual_env_path

  def CleanSshClient(self):
    print('SshClient cleaned..nothing to do')

  @property
  def state(self):
    return 'unknown'

  @property
  def instance_id(self):
    return self.hostname

  def ExecuteCommandAndWait(self, cmd, print_error=False):
    cmd = self.addVirtualEnv(cmd)
    self.runLocalCommand(cmd)

  def ExecuteCommandAndReturnStdout(self, cmd):
    cmd = self.addVirtualEnv(cmd)
    self.runLocalCommand(cmd)

  def ExecuteCommandAndStreamOutput(self,
                                    cmd,
                                    stdout_file=None,
                                    stderr_file=None,
                                    line_extractor=None,
                                    print_error=False,
                                    ok_exit_status=[0]):
    cmd = self.addVirtualEnv(cmd)
    self.runLocalCommand(cmd)

  def ExecuteCommandInThread(self,
                             command,
                             stdout_file=None,
                             stderr_file=None,
                             line_extractor=None,
                             print_error=False):
    command = self.addVirtualEnv(command)
    t = threading.Thread(
        target=self.runLocalCommand, args=[command, stdout_file])
    t.start()
    return t

  def RetrieveFile(self, remote_file, local_file):
    print('RetrieveFile not suported')

  def UploadFile(self, local_file, remote_file):
    print('UploadFile Not Supported from local:{} to remote:{}'.format(
        local_file, remote_file))

  def addVirtualEnv(self, cmd):
    """Adds virtual env to command if configured"""
    if self.virtual_env_path:
      cmd = 'source {};{}'.format(self.virtual_env_path, cmd)
    return cmd

  def runLocalCommand(self, cmd, stdout=None):
    f = None
    if stdout:
      f = open(stdout, 'a')
      f.write(cmd + '\n')
    for line in RunLocalCommand(cmd):
      if (line.strip('\n')):
        print(line.strip('\n'))
        if f:
          f.write(line.strip('\n') + '\n')


def RunLocalCommand(
    cmd,):
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  while (True):
    retcode = p.poll()  #returns None while subprocess is running
    line = p.stdout.readline()
    yield line
    if (retcode is not None):
      break


@contextmanager
def UseLocalInstances(virtual_env_path=''):
  """Creates instances to ssh into based on host names.

  Args:
    virtual_env_path: path to the virtual environment to use.
  """
  instances = []
  try:
    instance = LocalInstance('localhost', virtual_env_path=virtual_env_path)
    instances.append(instance)
    print('All {} instances ready!!!'.format(len(instances)))
    yield instances
  finally:
    CloseInstances(instances)


def CloseInstances(instances):
  print('Closing local instance..doing nothing...')
