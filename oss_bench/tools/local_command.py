"""Run command locally."""
import subprocess


def run_local_command(cmd):
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
    tmp_stdout = p.stdout.readline()
    out_str = tmp_stdout.decode('utf-8')
    stdout += out_str
    if retcode is not None:
      return retcode, stdout
