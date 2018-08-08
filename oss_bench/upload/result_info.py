"""Structures for a variety of different test results."""


def build_test_result(test_id,
                      result,
                      result_type='total_time',
                      result_units='ms',
                      test_harness='unknown',
                      test_environment='unknown'):
  """Core information about the results of the test.

  Args:
      test_id (str): Id when combined with test_source should represent a unique
        test that maybe run on multiple system types, e.g. P100 or K80.
      result (float): Float value representing the result of the test.
      result_type (str): Type of result, total_time, exps_per_sec,
        oom_batch_size, or global_step_per_sec.  Defaults to total_time.
      result_units (str, optional): Unitest of the results, defaults to ms.
      test_harness (str, optional): Test collection, e.g. tf_cnn_benchmarks,
        keras_benchmarks, model_garden_convergence, or caffe2_bench.
      test_environment (str, optional): Location test was run.

  Returns:
    Tuple with test_result and result in results array.
  """
  test_result = {}
  test_result['test_id'] = unicode(test_id)
  test_result['test_harness'] = unicode(test_harness)
  test_result['test_environment'] = unicode(test_environment)

  results = []
  result = build_result_info(results, result, result_type, result_units)

  return test_result, results


def build_result_info(results,
                      result,
                      result_type='total_time',
                      result_units='ms'):
  """Appends result dict to end of results array.

  Args:
    results (str): Array to add result dict into.
    result (float): Float value representing the result of the test.
    result_type (str): Type of result, total_time, exps_per_sec,
      oom_batch_size, or global_step_per_sec.  Defaults to total_time.
    result_units (str, optional): Unit of the results, defaults to ms.

  Returns:
    results appended with new result dict.

  """
  result_entry = {}
  result_entry['result'] = result
  result_entry['result_type'] = result_type
  result_entry['result_units'] = result_units

  results.append(result_entry)
  return results


def build_system_info(platform=None,
                      platform_type=None,
                      accel_type=None,
                      cpu_cores=None,
                      cpu_type=None,
                      cpu_sockets=None):
  """Information about the system the test was executed on.

  Args:
    platform (str): Higher level platform, e.g. aws, gce, or workstation.
    platform_type (str): Type of platform, DGX-1, p3.8xlarge, or z420.
    accel_type (str, optional): Type of accelerator, e.g. K80 or P100.
    cpu_cores (int, optional): Number of physical cpu cores.
    cpu_type (str, optional): Type of cpu.
    cpu_sockets (int, optional): Number of sockets

  Returns:
    `dict` with system info.

  """
  system_info = {}
  if platform:
    system_info['platform'] = unicode(platform)
  if platform_type:
    system_info['platform_type'] = unicode(platform_type)
  if accel_type:
    system_info['accel_type'] = unicode(accel_type)
  if cpu_cores:
    system_info['cpu_cores'] = cpu_cores
  if cpu_type:
    system_info['cpu_type'] = unicode(cpu_type)
  if cpu_type:
    system_info['cpu_sockets'] = cpu_sockets
  return system_info


def build_test_info(framework='tensorflow',
                    framework_version=None,
                    framework_describe=None,
                    channel=None,
                    build_type=None,
                    batch_size=None,
                    model=None,
                    accel_cnt=None,
                    cmd=None):
  """Returns test info in a dict.

  Args:
    framework (str, optional): Framework being tested, e.g. tensorflow,
      mxnet, or caffe2.  Defaults to tensorflow.
    framework_version: Version of the framework tested.
    framework_describe: More info on the framework version, often git describe.
    channel: Release channel, e.g. HEAD, PR_CHECK, NIGHTLY, RC, or FINAL.
    build_type: Type of build, e.g. OTB-GPU, AVX, AVX-512, or MKL.
    batch_size (int, optional): Total batch size.
    model (str, optional): Model being tested.
    accel_cnt (int, optional): Number of accelerators being utilized.
    cmd (str, optional): Command run for the test with args. Useful to record
      for tests that are run as subprocesses.

  Returns:
    `dict` with test info.
  """
  test_info = {}
  if framework:
    test_info['framework'] = framework
  if channel:
    test_info['channel'] = channel
  if build_type:
    test_info['build_type'] = build_type
  if batch_size:
    test_info['batch_size'] = batch_size
  if model:
    test_info['model'] = model
  if accel_cnt:
    test_info['accel_cnt'] = accel_cnt
  if framework_version:
    test_info['framework_version'] = framework_version
  if framework_describe:
    test_info['framework_describe'] = framework_describe
  if cmd:
    test_info['cmd'] = cmd
  return test_info
