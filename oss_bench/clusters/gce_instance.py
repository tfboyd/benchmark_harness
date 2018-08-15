"""Creates and controls GCE instances."""
from __future__ import print_function

from contextlib import contextmanager
import time

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials
import util


class GCEInstance(object):
  """Represents a GCE instance."""

  def __init__(self, instance, ssh_key='', username=''):
    self.instance = instance
    self.hostname = instance['private_ip']
    self.public_host = instance['public_ip']
    self.ssh_key = ssh_key
    self.opened_ssh_client = []
    self.username = username
    self.project = instance['project']
    self.zone = instance['zone']

    # API Service
    self.compute = get_compute_service()

  def __del__(self):
    self.clean_ssh_client()

  def wait_until_ready(self):
    """Waits until instance is ready for testing.

    GCE does not have a status monitor to check if the server is usable. Running
    only means the VM has "started" not that it is ready to use.

    Raises:
      Exception: When the number of instances found is not equal to 1.
    """
    while True:
      instances = lookup_gce_instance(
          self.project, self.zone, None, None, name=self.instance_id)
      if len(instances) != 1:
        raise Exception(
            'Unexpected number of instances {} returned, 1 expected for '
            'instance_id:{}'.format(len(instances), self.instance_id))
      instance = instances[0]
      if instance.state.lower() == 'running':
        print('running...')
        if instance.instance.get('public_ip', None):
          self.public_host = instance.instance.get('public_ip')
          print('public_ip:{}'.format(self.public_host))
          ssh_client = self.create_ssh_client()
          if ssh_client:
            break
          print('no ssh_client')
      print('Sleep 30 seconds waiting for instance({}) to come up'.format(
          self.instance_id))
      time.sleep(30)

  def wait_for_operation(self, operation):
    """Pulls status of operation until complete."""
    wait_for_operation(self.compute, self.project, self.zone, operation)

  def create_ssh_client(self):
    assert self.public_host is not None
    ssh_client = util.SshToHost(
        self.public_host, ssh_key=self.ssh_key, username=self.username)
    self.opened_ssh_client.append(ssh_client)
    return ssh_client

  def reuse_ssh_client(self):
    assert self.public_host is not None
    if not hasattr(self, 'ssh_client') or not self.ssh_client:
      self.ssh_client = util.SshToHost(
          self.public_host, ssh_key=self.ssh_key, username=self.username)
    return self.ssh_client

  def clean_ssh_client(self):
    if hasattr(self, 'ssh_client'):
      self.opened_ssh_client.append(self.ssh_client)
    for ssh_client in self.opened_ssh_client:
      try:
        ssh_client.close()
      except:
        pass
    self.opened_ssh_client = []
    self.ssh_client = None

  @property
  def state(self):
    return self.instance['status']

  def start(self):
    request = self.compute.instances().start(
        project=self.project, zone=self.zone, instance=self.instance_id)
    response = request.execute()
    self.wait_for_operation(response['name'])

  def stop(self):
    request = self.compute.instances().stop(
        project=self.project, zone=self.zone, instance=self.instance_id)
    response = request.execute()
    self.wait_for_operation(response['name'])

  def stop_and_wait_until_stopped(self):
    print('StopAndWaitUntilStopped not implemented')

  def terminate(self):
    request = self.compute.instances().delete(
        project=self.project, zone=self.zone, instance=self.instance_id)
    response = request.execute()
    self.wait_for_operation(response['name'])

  def terminate_and_wait_until_terminated(self):
    print('TerminateAndWaitUntilTerminated not implemented')

  @property
  def instance_id(self):
    return self.instance['name']

  def execute_command_and_wait(self, cmd, print_error=False):
    util.ExecuteCommandAndWait(
        self.reuse_ssh_client(), cmd, print_error=print_error)

  def execute_command_and_return_stdout(self, cmd):
    return util.ExecuteCommandAndReturnStdout(self.reuse_ssh_client(), cmd)

  def execute_command_and_stream_output(self,
                                        cmd,
                                        stdout_file=None,
                                        stderr_file=None,
                                        line_extractor=None,
                                        print_error=False,
                                        ok_exit_status=[0]):

    return util.ExecuteCommandAndStreamOutput(
        self.reuse_ssh_client(),
        cmd,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        line_extractor=line_extractor,
        print_error=print_error,
        ok_exit_status=ok_exit_status)

  def execute_command_in_thread(self,
                                command,
                                stdout_file=None,
                                stderr_file=None,
                                line_extractor=None,
                                print_error=False):
    ssh_client = self.create_ssh_client()
    return util.ExecuteCommandInThread(
        ssh_client,
        command,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        line_extractor=line_extractor,
        print_error=print_error)

  def retrieve_file(self, remote_file, local_file):
    sftp_client = self.reuse_ssh_client().open_sftp()
    sftp_client.get(remote_file, local_file)
    sftp_client.close()

  def upload_file(self, local_file, remote_file):
    print('Upload file from local:{} to remote:{}'.format(
        local_file, remote_file))
    sftp_client = self.reuse_ssh_client().open_sftp()
    sftp_client.put(local_file, remote_file)
    sftp_client.close()

  def attach_disk_maybe(self, disk_name):
    disks = self.instance.get('disks', [])
    find_disk = ('https://www.googleapis.com/compute/v1/projects/{}/zones/{}'
                 '/disks/{}').format(self.project, self.zone, disk_name)

    if find_disk not in disks:
      self.attach_disk(disk_name)
    else:
      print(
          'Disk {} already attached to {}'.format(disk_name, self.instance_id))

  def attach_disk(self, disk_name):
    """Attach a disk to the instance."""
    source = 'projects/{}/zones/{}/disks/{}'.format(self.project, self.zone,
                                                    disk_name)

    config = {'mode': 'READ_ONLY', 'source': source}

    request = self.compute.instances().attachDisk(
        project=self.project,
        zone=self.zone,
        instance=self.instance_id,
        body=config)
    response = request.execute()
    self.wait_for_operation(response['name'])

  def detach_disk_maybe(self, disk_name):
    disks = self.instance.get('disks', [])
    find_disk = ('https://www.googleapis.com/compute/v1/projects/{}/zones/{}'
                 '/disks/{}').format(self.project, self.zone, disk_name)

    if find_disk in disks:
      self.detach_disk(disk_name)

  def detach_disk(self, disk_name):
    request = self.compute.instances().detachDisk(
        project=self.project,
        zone=self.zone,
        instance=self.instance_id,
        deviceName=disk_name)
    response = request.execute()
    self.wait_for_operation(response['name'])


def lookup_gce_instance(project,
                        zone,
                        ssh_key,
                        username,
                        tag=None,
                        name=None,
                        compute=None):
  """Gets list of instances filtered by args passed."""
  if not compute:
    compute = get_compute_service()

  if tag:
    # filter = 'labels.{}:""'.format(tag)
    query_filter = 'name:{}*'.format(tag)
  elif name:
    query_filter = 'name:{}'.format(name)

  result = compute.instances().list(
      project=project, zone=zone, filter=query_filter).execute()
  instances = []
  if result.get('items'):
    for instance in result['items']:
      instances_meta = decode_instance_data(instance, project, zone)
      instances.append(GCEInstance(instances_meta, ssh_key, username=username))
  return instances


def decode_instance_data(instance_data, project, zone):
  """Parse select aspects from instance JSON object.

    Returns:
      simpler version of instance data
  """
  instance_meta = {}
  instance_meta['project'] = project
  instance_meta['zone'] = zone
  instance_meta['name'] = instance_data['name']
  instance_meta['status'] = instance_data['status']
  instance_meta['tags'] = instance_data['tags'].get('items', [])
  instance_meta['name'] = instance_data['name']

  orig_disks = instance_data['disks']
  disks = []
  for disk in orig_disks:
    disks.append(disk['source'])
  instance_meta['disks'] = disks

  # This works for instances this script creates
  orig_network = instance_data['networkInterfaces'][0]
  instance_meta['public_ip'] = orig_network['accessConfigs'][0].get('natIP')
  instance_meta['private_ip'] = orig_network['networkIP']

  # print('instance_meta:{}'.format(instance_meta))
  return instance_meta


def get_compute_service(version='v1'):
  """Gets the compute service API signed in."""
  credentials = GoogleCredentials.get_application_default()
  compute = googleapiclient.discovery.build(
      'compute', version, credentials=credentials)
  return compute


@contextmanager
def reuse_gce_instance(instance_tag=None,
                       ssh_key=None,
                       close_behavior=None,
                       username=None,
                       project=None,
                       zone=None):
  """Get all instances with the tag provided and return them."""
  instances = []
  try:
    instances = get_instances_to_reuse(
        project, zone, ssh_key, username, instance_tag=instance_tag)
    print('All {} instances ready!!!'.format(len(instances)))
    yield instances
  finally:
    close_instances(instances, close_behavior)


@contextmanager
def create_gce_instance(num_instances=1,
                        image_id='',
                        instance_type='n1-standard-2|nvidia-tesla-k80|1',
                        ssh_key='',
                        instance_tag='tf',
                        close_behavior=None,
                        username='',
                        project=None,
                        zone=None,
                        service_account=None):
  """Creates instances and then looks them back up via ReuseInstances."""
  instances = []
  try:
    compute = get_compute_service()
    responses = []
    for i in range(num_instances):
      name = '{}-{}'.format(instance_tag, i)
      print('Creating Instance {} of type {}'.format(name, instance_type))
      responses.append(
          create_instance(compute, project, zone, image_id, instance_type, name,
                          instance_tag, service_account))

      # Waits for all of the create instances to finish
    print('Wait for create instance operations to finish...')
    for create_response in responses:
      wait_for_operation(compute, project, zone, create_response['name'])

    instances = get_instances_to_reuse(
        project, zone, ssh_key, username, instance_tag=instance_tag)
    print('All {} instances ready!!!'.format(len(instances)))
    yield instances
  finally:
    close_instances(instances, close_behavior)


def close_instances(instances, close_behavior):
  """Stop the instance and cleanup."""
  if close_behavior is not None:
    for instance in instances:
      if close_behavior == 'terminate':
        print('terminating {}...'.format(instance.instance_id))
        instance.terminate()
      elif close_behavior == 'stop':
        print('Stopping {}...'.format(instance.instance_id))
        instance.stop()
  else:
    print('Closing leaving instances as-is...')


def get_instances_to_reuse(project, zone, ssh_key, username, instance_tag=None):
  """Lookup, start, and return existing instances."""
  if instance_tag:
    instances = lookup_gce_instance(
        project, zone, ssh_key, username, tag=instance_tag)
  else:
    raise ValueError('Unable to get instance with args passed')

  if not instances:
    # Mostly pointless for now but nice to know if 0.
    raise ValueError(
        'No instances found for instance_tag={}'.format(instance_tag))

  for instance in instances:
    if instance.state.lower() != 'running':
      print('Current instance({}) state:{}, trying to start.'.format(
          instance.instance_id, instance.state))
      instance.start()

  for instance in instances:
    print('Waiting for instance({}) to be ready.'.format(instance.instance_id))
    instance.wait_until_ready()

  return instances


def create_instance(compute, project, zone, image_id, instance_type, name, tag,
                    service_account):
  """Creates GCE instance with an async call."""
  base_instance, accel, accel_count = parse_instance_type(instance_type)
  config = return_create_instance_config(project,
                                         zone,
                                         image_id,
                                         base_instance,
                                         name,
                                         tag,
                                         accelerator=accel,
                                         accelerator_count=accel_count,
                                         service_account=service_account)

  request = compute.instances().insert(project=project, zone=zone, body=config)
  response = request.execute()
  return response


def parse_instance_type(instance_type):
  """Breaks instance type down into components.

  Args:
    instance_type: string representing the GCE instance type.  Expected format:
    base_instance|accelerator|accelerator_count, e.g.
    machine-type n1-standard-2|nvidia-tesla-k80|1

  Returns:
    A tuple of strings: base instance (cpus), accelerator type, and accelerator
    count.  Nones are returned and warning messages printed to the console if
    accelerator type and count are not provided.

  Raises:
    ValueError: if the instance_type is in an unknown format.
  """

  parts = instance_type.split('|')
  if len(parts) == 3:
    return parts[0], parts[1], parts[2]
  elif len(parts) == 1:
    return parts[0], None, None
  else:
    raise ValueError(
        'Instance type in unknown format, "base_instance|accelerator|count" '
        'expected:{}'.format(instance_type))


def wait_for_operation(compute, project, zone, operation):
  """Pulls status of operation until complete."""
  print('Waiting for operation to finish...')
  while True:
    result = compute.zoneOperations().get(
        project=project, zone=zone, operation=operation).execute()

    if result['status'] == 'DONE':
      # print('done...{}'.format(result))
      if 'error' in result:
        raise Exception(
            'Error executing operation:{}:{}'.format(result['error'], result))
      return result

    time.sleep(1)


def return_create_instance_config(
    project,
    zone,
    image_id,
    instance_type,
    instance_name,
    tag,
    accelerator=None,
    service_account=None,
    accelerator_count=None):
  """Returns a config object to create a GCE instance."""

  full_zone = 'projects/{}/zones/{}'.format(project, zone)
  full_instance_type = '{}/machineTypes/{}'.format(full_zone, instance_type)
  full_image = 'projects/{}/global/images/{}'.format(project, image_id)
  base_zone = '-'.join(zone.split('-')[:-1])

  config = {
      'name':
          instance_name,
      'zone':
          full_zone,
      'machineType':
          full_instance_type,
      'tags': {
          'items': [tag]
      },
      'disks': [{
          'type': 'PERSISTENT',
          'boot': True,
          'mode': 'READ_WRITE',
          'autoDelete': True,
          'deviceName': 'instance-1',
          'initializeParams': {
              'sourceImage': full_image,
              'diskType': full_zone + '/diskTypes/pd-standard',
              'diskSizeGb': '100'
          }
      }],
      'canIpForward':
          False,
      'networkInterfaces': [{
          'network':
              'projects/' + project + '/global/networks/default',
          'subnetwork':
              'projects/' + project + '/regions/' + base_zone +
              '/subnetworks/default',
          'accessConfigs': [{
              'name': 'External NAT',
              'type': 'ONE_TO_ONE_NAT'
          }]
      }],
      'description':
          '',
      'scheduling': {
          'preemptible': False,
          'onHostMaintenance': 'TERMINATE',
          'automaticRestart': True
      }
  }

  if service_account:
    config['serviceAccounts'] = [{
        'email': service_account,
        'scopes': [
            'https://www.googleapis.com/auth/devstorage.read_only',
            'https://www.googleapis.com/auth/logging.write',
            'https://www.googleapis.com/auth/monitoring.write',
            'https://www.googleapis.com/auth/servicecontrol',
            'https://www.googleapis.com/auth/service.management.readonly',
            'https://www.googleapis.com/auth/trace.append'
        ]
    }]

  if accelerator:
    config['guestAccelerators'] = [{
        'acceleratorCount':
            accelerator_count,
        'acceleratorType':
            'https://www.googleapis.com/compute/beta/' + full_zone +
            '/acceleratorTypes/' + accelerator
    }]

  return config
