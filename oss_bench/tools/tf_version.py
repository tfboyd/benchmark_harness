"""Extract TensorFlow version info."""
from __future__ import print_function
import tensorflow as tf


def get_tf_full_version():
  """Returns TensorFlow version as reported by TensorFlow.

    Note: The __git__version__ can be confusing as the TensorFlow version
    number in the string often points to an older branch due to git merges.
    The git hash is still correct.  The best option is to use the numeric
    version from __version__ and the hash from __git_version__.

  Returns:
    Tuple of __version__, __git_version__
  """
  return tf.__version__, tf.__git_version__
