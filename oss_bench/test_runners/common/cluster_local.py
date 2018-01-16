"""Util to execute commands in the local shell"""
import os
import time
from contextlib import contextmanager
from subprocess import call
import threading
import subprocess


class LocalInstance(object):

  def __init__(self, host, virtual_env_path=''):
    self.hostname = host
    self.virtual_env_path = virtual_env_path

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
                                    print_error=False,
                                    ok_exit_status=[0]):
    cmd = self.addVirtualEnv(cmd)
    self.runLocalCommand(cmd)

  def ExecuteCommandInThread(self,
                             command,
                             stdout_file=None,
                             stderr_file=None,
                             print_error=False):
    command = self.addVirtualEnv(command)
    t = threading.Thread(
        target=self.runLocalCommand, args=[command, stdout_file])
    t.start()
    return t

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


def RunLocalCommand(cmd,):
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  while (True):
    retcode = p.poll()
    line = p.stdout.readline()
    yield line
    if (retcode is not None):
      break


def UseLocalInstances(virtual_env_path=''):
  """Returns an instance to run tests against

  Args:
    virtual_env_path: path to the virtual environment to use.
  """
  instance = LocalInstance('localhost', virtual_env_path=virtual_env_path)
  return instance
