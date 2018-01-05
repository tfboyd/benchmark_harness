"""Extract TensorFlow version info."""
from __future__ import print_function
import subprocess
import tensorflow as tf


def get_tf_full_version():
  """Returns driver and gpu info using nvidia-smi.

  Note: Assumes if the system has multiple GPUs that they are all the same with
  one exception.  If the first result is a Quadro, the heuristic assumes
  this may be a workstation and takes the second entry.  

  Returns:
    Tuple of device driver version and gpu name.
  """
  tf_version = tf.__git_version__
  return tf_version
