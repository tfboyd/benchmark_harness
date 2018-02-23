"""Tests tracker module."""
import unittest

import os
import tracker
from mock import patch

TEST_DIR = './'

class TestTracker(unittest.TestCase):


  @patch('tools.tracker._get_state_object')
  def test_check_state_false(self, mock_get_state_object):
    """Tests state check is false when tests is not found."""
    mock_get_state_object.return_value = self._mock_state_object()
    found = tracker.check_state(None,'tensorflow','RC','OTB.GPU','1.4_02102018','tf_cnn_bench')
    self.assertFalse(found)

  @patch('tools.tracker._get_state_object')
  def test_check_state_true(self, mock_get_state_object):
    """Tests state check is true when test is found."""
    mock_get_state_object.return_value = self._mock_state_object()
    found = tracker.check_state(None,'mxnet','FINAL','OTB.GPU','1_0_0','mxnet')
    self.assertTrue(found)

  @patch('tools.tracker._save_state_object')
  @patch('tools.tracker._get_state_object')
  def test_update_state_round_trip(self, mock_get_state_object, mock_save_state_object):
    """Tests check state not found, update state, and is then found."""
    mock_get_state_object.return_value = self._mock_state_object()
    found = tracker.check_state(None,'tensorflow','RC','OTB.GPU','1.4_02102018','tf_models')
    self.assertFalse(found)
    tracker.update_state(None,'tensorflow','RC','OTB.GPU','1.4_02102018','tf_models')
    saved_object = mock_save_state_object.call_args[0][1]
    mock_get_state_object.return_value = saved_object
    found_post_update = tracker.check_state(None,'tensorflow','RC','OTB.GPU','1.4_02102018','tf_models')
    self.assertTrue(found_post_update)

  @patch('tools.tracker._save_state_object')
  @patch('tools.tracker._get_state_object')
  def test_update_state_new_key(self, mock_get_state_object,mock_save_state_object):
    """Tests update state when new key needs to be created."""
    mock_get_state_object.return_value = self._mock_state_object()
    tracker.update_state(None,'tensorflow','FINAL','OTB.GPU','1.4_02102018','tf_models')
    saved_object = mock_save_state_object.call_args[0][1]
    hash_key = tracker._hash_key('tensorflow','FINAL','OTB.GPU','1.4_02102018')
    self.assertIn(hash_key, saved_object)
    self.assertIn('tf_models', saved_object[hash_key]['tests'])

  def test_get_state_object_first_time(self):
    """Tests get state when a tracker file has not been saved yet."""
    state = tracker._get_state_object(TEST_DIR)
    self.assertEqual({},state)

  def test_save_state_object_round_trip(self):
    """Tests save state (end-to-end no mocks) and reads it back."""
    state_file = os.path.join(TEST_DIR, tracker.STATE_FILE)
    if os.path.exists(state_file):
      os.remove(state_file)
    tracker._save_state_object(TEST_DIR,self._mock_state_object())
    tracker_object = tracker._get_state_object(TEST_DIR)
    os.remove(os.path.join(TEST_DIR, tracker.STATE_FILE))
    self.assertEqual(self._mock_state_object(),tracker_object)

  def _mock_state_object(self):
    """Returns state object for testing."""
    state_object = {}
    entry_1 = {}
    entry_1['framework'] = 'tensorflow'
    entry_1['channel'] = 'RC'
    entry_1['build_type'] = 'OTB.GPU'
    entry_1['version'] = '1.4_02102018'
    entry_1['tests'] = ['foobar']
    entry_key_1 = tracker._hash_key('tensorflow','RC','OTB.GPU','1.4_02102018')
    state_object[entry_key_1] = entry_1
    entry_2 = {}
    entry_2['channel'] = 'FINAL'
    entry_2['build_type'] = 'OTB.GPU'
    entry_1['version'] = '1_0_0'
    entry_2['tests'] = ['mxnet']
    entry_key_2 = tracker._hash_key('mxnet', 'FINAL','OTB.GPU','1_0_0')
    state_object[entry_key_2] = entry_2
    return state_object


