"""Extract information about the system GPU."""
from __future__ import print_function
import subprocess


def get_gpu_info():
  """Returns driver and gpu info using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same.
  The first result in the list is returned.

  Returns:
    Tuple of device driver version and gpu name.
  """
  cmd = 'nvidia-smi --query-gpu=driver_version,gpu_name --format=csv'
  retcode, result = _run_local_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and len(lines) > 1:
    gpu_info = lines[1].split(',')
    return gpu_info[0].strip(), gpu_info[1].strip()
  else:
    print('nvidia-smi did not return as expected:{}'.format(result))
    return '', ''


def _run_local_command(cmd):
  """Structures for a variety of different test results.

  Args:
    cmd: Command to execute
  Returns:
    Tuple of the command return value and the standard out in as a string.
  """
  p = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
  stdout = ''
  while True:
    retcode = p.poll()
    stdout += p.stdout.readline()
    if retcode is not None:
      return retcode, stdout
