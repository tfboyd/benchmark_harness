"""Extract information about the system GPU."""
from __future__ import print_function
import re
import local_command


def get_gpu_info():
  """Returns driver and gpu info using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same with
  one exception.  If the first result is a Quadro, the heuristic assumes
  this may be a workstation and takes the second entry.

  Returns:
    Tuple of device driver version and gpu name.
  """
  cmd = 'nvidia-smi --query-gpu=driver_version,gpu_name --format=csv'
  retcode, result = local_command.run_local_command(cmd)
  lines = result.splitlines()
  if retcode == 0 and len(lines) > 1:
    gpu_info = lines[1].split(',')
    if 'Quadro' in gpu_info[1] and len(lines) > 2:
      gpu_info = lines[2].split(',')
      return gpu_info[0].strip(), gpu_info[1].strip()
    else:
      return gpu_info[0].strip(), gpu_info[1].strip()
  else:
    print('nvidia-smi did not return as expected:{}'.format(result))
    return '', ''


def get_running_processes():
  """Returns list of `dict` objects representing running processes on GPUs."""
  retcode, result = local_command.run_local_command('nvidia-smi')
  lines = result.splitlines()
  if retcode == 0 and len(lines) > 1:
    # Goes to the first line with the word Processes, jumps down one and then
    # parses the list of processes.
    look_for_processes = False
    processes = []
    for line in lines:
      # Summary line starts with images/sec
      if line.find('Processes') > 0:
        look_for_processes = True

      if look_for_processes:
        p = re.compile('[0-1]+')
        m = p.search(line)
        if m and m.span()[0] == 5:
          line_parts = line.strip().replace('|', '').split()
          processes.append(line_parts)

    return processes

  else:
    print('nvidia-smi did not return as expected:{}'.format(result))
    return '', ''


def is_ok_to_run():
  """Returns true if the system is free to run GPU tests.

  Checks the list of processes and if the list is empty or if the list of
  processes does not contain actual ML jobs returns true.  Non-ML Jobs
  like 'Xorg' or even 'cinnamon' are not a problem.

  Note: Currently this method returns true if no python processes are found.
    Which seems more sane than a long list of processes that do not matter. On
    clean systems the process list should be and normally is zero.
  """
  processes = get_running_processes()
  for process in processes:
    # Checks process name position for process named 'python'.
    if 'python' == process[3]:
      return False
  return True
