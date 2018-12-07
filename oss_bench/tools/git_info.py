"""Extract information about local git repos."""
from __future__ import print_function
import tools.local_command as local_command


def git_repo_describe(git_dir):
  """Returns describe for git_dir.

  Args:
    git_dir: git directory to run describe on.

  Returns:
    str with git describe info.

  Raises:
    Exception: If return value of the command is non-zero.
  """
  cmd = 'git -C {} describe --always'.format(git_dir)
  retval, stdout = local_command.run_local_command(cmd)
  if retval != 0:
    raise Exception('Command ({}) failed to run:{}'.format(cmd, stdout))
  return stdout.strip()


def git_repo_last_commit_id(git_dir):
  """Returns last_commit_id for git_dir.

  Args:
    git_dir: git directory to run describe on.

  Returns:
    str of last commit_id.

  Raises:
    Exception: If return value of the command is non-zero.
  """
  cmd = 'git -C {} log --format="%H" -n 1'.format(git_dir)
  retval, stdout = local_command.run_local_command(cmd)
  if retval != 0:
    raise Exception('Command ({}) failed to run:{}'.format(cmd, stdout))
  return stdout.strip()
