"""Example usage of gce_instance module."""
from __future__ import print_function

import argparse
import os

import gce_instance


def main():

  if FLAGS.reuse:
    with gce_instance.reuse_gce_instance(
        project=FLAGS.project,
        zone=FLAGS.zone,
        instance_tag=FLAGS.instance_tag,
        ssh_key=os.path.join(os.environ['HOME'], '.ssh/google_compute_engine'),
        close_behavior=None,
        username='ubuntu') as existing_instances:

      print('reuse{}'.format(existing_instances))
      result = existing_instances[0].execute_command_and_return_stdout('date')
      print('result:  {}'.format(result))
      result = existing_instances[0].execute_command_and_return_stdout('pwd')
      print('result:  {}'.format(result))

  else:
    with gce_instance.create_gce_instance(
        project=FLAGS.project,
        zone=FLAGS.zone,
        service_account=None,
        num_instances=1,
        image_id='tf-ubuntu-1604-20180531-396',
        instance_type='n1-standard-2|nvidia-tesla-k80|1',
        instance_tag=FLAGS.instance_tag,
        ssh_key=os.path.join(os.environ['HOME'], '.ssh/google_compute_engine'),
        close_behavior=None,
        username='ubuntu') as instances:
      print('created instances!!!!!!!! {}'.format(instances))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--project',
      type=str,
      default=None,
      help='GCE project to use.')
  parser.add_argument(
      '--zone',
      type=str,
      default=None,
      help='Zone to use.')
  parser.add_argument(
      '--instance_tag',
      type=str,
      default=None,
      help='Tag for the instance.')
  parser.add_argument(
      '--reuse',
      type=bool,
      default=False,
      help='True to find existing instances by tag and use them')
  FLAGS, unparsed = parser.parse_known_args()
  main()
