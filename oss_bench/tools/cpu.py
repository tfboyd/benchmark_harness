"""Extract CPU info."""
from __future__ import print_function
import tools.local_command as local_command


def get_cpu_info():
  """Returns driver and gpu info using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same with
  one exception.  If the first result is a Quadro, the heuristic assumes
  this may be a workstation and takes the second entry.

  Returns:
    Tuple of device driver version and gpu name.
  """
  model_name = _model_name()
  core_count = _core_count()
  socket_count = _socket_count()
  cpu_info = _cpu_info()

  return model_name, socket_count, core_count, cpu_info


def _model_name():
  cmd = "cat /proc/cpuinfo | grep 'model name' | sort --unique"
  retcode, result = local_command.run_local_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    model_name_parts = lines[0].split(':')
    return model_name_parts[1].strip()
  else:
    print('Error getting cpuinfo model name: {}'.format(result))
    return ''


def _core_count():
  cmd = "cat /proc/cpuinfo | grep 'cpu cores' | sort --unique"
  retcode, result = local_command.run_local_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    core_count_parts = lines[0].split(':')
    # Cores * sockets = total cores for the system.
    core_count = int(core_count_parts[1].strip())
    total_cores = core_count * _socket_count()
    return total_cores
  else:
    print('Error getting cpuinfo core count: {}'.format(result))
    return -1


def _socket_count():
  cmd = 'grep -i "physical id" /proc/cpuinfo | sort -u | wc -l'
  retcode, result = local_command.run_local_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and lines:
    return int(lines[0])
  else:
    print('Error getting cpuinfo scocket count: {}'.format(result))
    return -1


def _cpu_info():
  cmd = 'cat /proc/cpuinfo'
  retcode, result = local_command.run_local_command(cmd)
  if retcode == 0:
    return result
  else:
    print('Error getting cpuinfo: {}'.format(result))
    return ''
