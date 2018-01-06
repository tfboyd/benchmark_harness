"""Tests git_info module."""
import unittest

import git_info
from mock import patch


class TestGitInfo(unittest.TestCase):

  @patch('tools.local_command.run_local_command')
  def test_git_repo_describe(self, run_local_command_mock):
    """Tests repo describe success."""
    run_local_command_mock.return_value = [0, 'bfd2f67']
    repo_describe = git_info.git_repo_describe('/Foo')
    self.assertEqual('bfd2f67', repo_describe)

  @patch('tools.local_command.run_local_command')
  def test_git_repo_describe_error(self, run_local_command_mock):
    """Tests repo describe when error is thrown."""
    run_local_command_mock.return_value = [1, 'Some random Error..']
    with self.assertRaises(Exception):
      git_info.git_repo_describe('/Foo')

  @patch('tools.local_command.run_local_command')
  def test_git_repo_last_commit_id(self, run_local_command_mock):
    """Tests get repo last commit_id success."""
    run_local_command_mock.return_value = [
        0, 'bfd2f676fe5d21c66f30732648e58c4ca3f3d5b5'
    ]
    commit_id = git_info.git_repo_last_commit_id('/Foo')
    self.assertEqual('bfd2f676fe5d21c66f30732648e58c4ca3f3d5b5', commit_id)

  @patch('tools.local_command.run_local_command')
  def test_git_repo_last_commit_id_error(self, run_local_command_mock):
    """Tests get repo last commit_id when error is thrown."""
    run_local_command_mock.return_value = [1, 'Some random error...']
    with self.assertRaises(Exception):
      git_info.git_repo_last_commit_id('/Foo')
