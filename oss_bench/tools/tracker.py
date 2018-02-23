"""Checks and saves state of tests that have been run."""
from __future__ import print_function

import hashlib
import os
import yaml

STATE_FILE = 'test_tracker.yaml'

def check_state(workspace, framework, channel, build_type, version, test):
  """Returns true if existing info is found, false otherwise."""
  state_object = _get_state_object(workspace)
  key = _hash_key(framework, channel, build_type, version)
  if key in state_object:
    if test in state_object[key]['tests']:
      return True
  return False


def update_state(workspace, framework, channel, build_type, version, test):
  """Update state object"""
  state_object = _get_state_object(workspace)
  key = _hash_key(framework, channel, build_type, version)
  entry = None
  if key in state_object:
    entry = state_object[key]
  else:
    entry = {}
    entry['framework'] = framework
    entry['channel'] = channel
    entry['build_type'] = build_type
    entry['version'] = version
    entry['tests'] = []
    state_object[key] = entry

  if test not in entry['tests']:
    entry['tests'].append(test)

  _save_state_object(workspace, state_object)


def _get_state_object(workspace):
  """Returns the state object to be queried."""

  state_file = os.path.join(workspace, STATE_FILE)
  if os.path.exists(state_file):
    with open(state_file) as f:
      state_tracker = yaml.safe_load(f)
      return state_tracker
  else:
    return {}


def _save_state_object(workspace, test_tracker):
  """Saves state object and overwrites existing file."""

  config_file_out = os.path.join(workspace, STATE_FILE)
  with open(config_file_out, 'w') as state_file:
    state_file.write(yaml.dump(test_tracker))


def _hash_key(framework, channel, build_type, version):
  """Returns hash of the unique values."""
  key_str = '{}{}{}{}'.format(framework, channel, build_type, version)
  return hashlib.sha1(key_str).hexdigest()
