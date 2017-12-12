"""Manual test that inserts data into BigQuery."""
import result_info
import result_upload


def main():
  """Inserts a row into a development table to be manually verified.

  This is less of a test than a debug tool that would be better as an actual
  integration test.

  SELECT * FROM
  [google.com:tensorflow-performance:dev_test_result_upload.results] LIMIT 1000
  """
  extras = {}
  extras['newfield'] = 'newfield value'
  test_result, results = result_info.build_test_result('random_test_id', 123.4)
  system_info = result_info.build_system_info(
      platform='aws', platform_type='p3.8xlarge')
  test_info = result_info.build_test_info(batch_size=32)

  result_upload.upload_result(
      test_result,
      results,
      'google.com:tensorflow-performance',
      test_info=test_info,
      system_info=system_info,
      extras=extras)

if __name__ == '__main__':
  main()
