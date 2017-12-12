import os
import time
import util
from contextlib import contextmanager
from subprocess import call

class SSHInstance(object):

  def __init__(self, host, ssh_key='', username='', virtual_env_path='', password=None):
    self.hostname = host
    self.opened_ssh_client = []
    self.username = username
    self.ssh_key=ssh_key
    self.virtual_env_path = virtual_env_path
    self.password = password

  def __del__(self):
    self.CleanSshClient()

  def WaitUntilReady(self):
    """Wait until instance is ready for testing"""
    while True:
      ssh_client = self.CreateSshClient()
      if ssh_client:
        break
      print('Sleep 30 seconds waiting for instance({}) to come up'.format(
          self.instance_id))
      time.sleep(30)

  def CreateSshClient(self):
    assert self.hostname is not None
    ssh_client = util.SshToHost(
        self.hostname, ssh_key=self.ssh_key, username=self.username, password=self.password)
    self.opened_ssh_client.append(ssh_client)
    return ssh_client

  def reuse_ssh_client(self):
    assert self.hostname is not None
    if not hasattr(self, 'ssh_client') or self.ssh_client == None:
      self.ssh_client = util.SshToHost(
          self.hostname, ssh_key=self.ssh_key, username=self.username, password=self.password)
    return self.ssh_client

  def CleanSshClient(self):
    if hasattr(self, 'ssh_client'):
      self.opened_ssh_client.append(self.ssh_client)
    for ssh_client in self.opened_ssh_client:
      try:
        ssh_client.close()
      except:
        pass
    self.opened_ssh_client = []
    self.ssh_client = None

  @property
  def state(self):
    return 'unknown'

  @property
  def instance_id(self):
    return self.hostname

  def ExecuteCommandAndWait(self, cmd, print_error=False):
    cmd = self.addVirtualEnv(cmd)
    util.ExecuteCommandAndWait(
        self.reuse_ssh_client(), cmd, print_error=print_error)

  def ExecuteCommandAndReturnStdout(self, cmd):
    cmd = self.addVirtualEnv(cmd)
    return util.ExecuteCommandAndReturnStdout(self.reuse_ssh_client(), cmd)

  def ExecuteCommandAndStreamOutput(self,
                                    cmd,
                                    stdout_file=None,
                                    stderr_file=None,
                                    line_extractor=None,
                                    print_error=False,
                                    ok_exit_status=[0]):
    cmd = self.addVirtualEnv(cmd)
    return util.ExecuteCommandAndStreamOutput(
        self.reuse_ssh_client(),
        cmd,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        line_extractor=line_extractor,
        print_error=print_error,
        ok_exit_status=ok_exit_status)

  def ExecuteCommandInThread(self,
                             command,
                             stdout_file=None,
                             stderr_file=None,
                             line_extractor=None,
                             print_error=False):
    command = self.addVirtualEnv(command)
    ssh_client = self.CreateSshClient()
    return util.ExecuteCommandInThread(
        ssh_client,
        command,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        line_extractor=line_extractor,
        print_error=print_error)

  def RetrieveFile(self, remote_file, local_file):
    sftp_client = self.reuse_ssh_client().open_sftp()
    sftp_client.get(remote_file, local_file)
    sftp_client.close()

  def UploadFile(self, local_file, remote_file):
    print('Upload file from local:{} to remote:{}'.format(local_file, remote_file))
    sftp_client = self.reuse_ssh_client().open_sftp()
    sftp_client.put(local_file, remote_file)
    sftp_client.close()


  def addVirtualEnv(self, cmd):
    """Adds virtual env to command if configured"""
    if self.virtual_env_path:
      cmd = 'source {};{}'.format(self.virtual_env_path, cmd)
    return cmd

@contextmanager
def UseSSHInstances(hosts=None,
                    username=None,
                    ssh_key=None,
                    password=None,
                    virtual_env_path=''):
  """Creates instances to ssh into based on host names.

  Args:
    hosts: List of hostnames or ip addresses.
    username: username to login with
    ssh_key: ssh_key to use to login with.  
  """
  instances = []
  try:
    for host in hosts:
      instances.append(SSHInstance(host, username=username, ssh_key=ssh_key, password=password, virtual_env_path=virtual_env_path))

    for instance in instances:
      print('Waiting for instance({}) to be ready.'.format(instance.instance_id))
      instance.WaitUntilReady()

    print('All {} instances ready!!!'.format(len(instances)))
    yield instances
  finally:
    CloseInstances(instances)


def CloseInstances(instances):
  print('Closing leaving instances as-is...')


