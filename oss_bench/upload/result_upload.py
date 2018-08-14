"""Upload test results."""
from __future__ import print_function
import copy
from datetime import datetime
import json
import os
import pwd
import uuid

import pytz

import google.auth
from google.cloud import bigquery
from google.cloud.bigquery.dbapi import connect


def safe_unicode(s):
  if isinstance(s, str):
    return s
  else:
    return s.decode(encoding)


def upload_result(test_result,
                  result_info,
                  project,
                  dataset='benchmark_results_dev',
                  table='result',
                  test_info=None,
                  system_info=None,
                  extras=None):
  """Upload test result.

  Note: Using Insert likely limits the number of inserts per day to 1000
  according to the bigquery quota page. I (tobyboyd@) noticed this after the
  fact.  Will upgrade to streaming inserts in near future, which is nearly
  unlimited 50k+

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.

  Args:
    test_result: `dict` with core info.  Use `result_info.build_test_result`.
    result_info: `dict` with result info.  Use `result_info.build_test_result`.
    project: Project where BigQuery dataset is located.
    dataset: BigQuery dataset to use.
    table: BigQuery table to insert into.
    test_info: `dict` of test info. Use `result_info.build_test_info`.
    system_info: `dict` of system info. Use `result_info.build_system_info`.
    extras: `dict` of values that will be serialized to JSON.
  """

  # Project is disgarded in favor of what the user passes in.
  if project == 'LOCAL':
    print('test_result:{}', test_result)
    print('result_info:{}', result_info)
    print('test_info:{}', test_info)
    print('system_info:{}', system_info)
    print('extras:{}', extras)
  else:
    credentials, _ = google.auth.default()

    row = _build_row(credentials, test_result, result_info, test_info,
                     system_info, extras)

    client = bigquery.Client(project=project, credentials=credentials)
    conn = connect(client=client)
    cursor = conn.cursor()
    sql = """INSERT into {}.{} (result_id, test_id, test_harness,
             test_environment, result_info, user, timestamp, system_info,
             test_info, extras)
               VALUES
             (@result_id, @test_id, @test_harness, @test_environment,
             @result_info, @user, @timestamp, @system_info, @test_info, @extras)
             """.format(dataset, table)

    cursor.execute(sql, parameters=row)
    conn.commit()
    # Cursor and connection closes on their own as well.
    cursor.close()
    conn.close()


def _build_row(credentials,
               test_result,
               result_info,
               test_info=None,
               system_info=None,
               extras=None):
  """Builds row to be inserted into BigQuery.

  Note: BigQuery maps unicode() to STRING for python2.  If str is used that is
  mapped to BYTE.

  Args:
    credentials: Result of the test, strongly suggest use Result.
    test_result: `dict` with core info.  Use `result_info.build_test_result`.
    result_info: `dict` with result info.  Use `result_info.build_test_result`.
    test_info: `dict` of test info. Use `result_info.build_test_info`.
    system_info: `dict` of system info. Use `result_info.build_system_info`.
    extras: `dict` of values that will be serialized to JSON.

  Returns:
    `dict` to be inserted into BigQuery.
  """
  row = copy.copy(test_result)
  row['result_id'] = safe_unicode(str(uuid.uuid4()))
  # The user is set to the email address of the service account.  If that is not
  # found, then the logged in user is used as a last best guess.
  if hasattr(credentials, 'service_account_email'):
    row['user'] = credentials.service_account_email
  else:
    row['user'] = safe_unicode(pwd.getpwuid(os.getuid())[0])

  # gpylint warning suggests using a different lib that does not look helpful.
  # pylint: disable=W6421
  row['timestamp'] = datetime.utcnow().replace(tzinfo=pytz.utc)

  # BigQuery expects unicode object and maps that to datatype.STRING.
  row['result_info'] = safe_unicode(json.dumps(result_info))
  row['system_info'] = safe_unicode(json.dumps(system_info if system_info else None))
  row['test_info'] = safe_unicode(json.dumps(test_info) if test_info else None)
  row['extras'] = safe_unicode(json.dumps(extras))

  return row
