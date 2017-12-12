"""Run a auto tests."""
from subprocess import call
import subprocess

def run_local_command(cmd, stdout='./log.txt'):
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





def main():
  run_local_command('git log')





if __name__ == '__main__':
  main()