"""Structures for a vareity of different test results."""


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
    result_units (str, optional): Unitest of the results, defaults to ms.

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
                      cpu_type=None):
  """Information about the system the test was executed on.

  Args:
      platform (str): Higher level platform, e.g. aws, gce, or workstation.
      platform_type (str): Type of platform, DGX-1, p3.8xlarge, or z420.
      accel_type (str, optional): Type of accelerator, e.g. K80 or P100.
      cpu_cores (str, optional): Number of physical cpu cores.
      cpu_type (str, optional): Type of cpu.

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
  return system_info


def build_test_info(framework='tensorflow',
                    batch_size=None,
                    model=None,
                    accel_cnt=None):
  """Initialize TestInfo object.

  Args:
    framework (str, optional): Framework being tested, e.g. tesnsorflow,
      mxnet, or caffe2.  Defaults to tensorflow.
    batch_size (int, optional): Total batch size.
    model: Model being tested.
    accel_cnt (int, optional): Number of accelerators bieng utilized.

  Returns:
    `dict` with test info.
  """
  test_info = {}
  if framework:
    test_info['framework'] = framework
  if batch_size:
    test_info['batch_size'] = batch_size
  if model:
    test_info['model'] = model
  if accel_cnt:
    test_info['accel_cnt'] = accel_cnt
  return test_info
