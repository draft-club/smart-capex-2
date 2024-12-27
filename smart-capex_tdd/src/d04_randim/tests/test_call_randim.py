import os
import unittest
from io import BytesIO
from unittest.mock import patch, MagicMock

from src.d00_conf.conf import conf
from src.d04_randim.call_randim import ApiRandim


class TestApiRandimTest(unittest.TestCase):
    def setUp(self):
        self.randim_api = ApiRandim()
        self.expected_headers = {
            'Authorization': '',
            'Cookie': 'JSESSIONID=820855AD21E0F686512C40B6E07DFDFE; '
                      'XSRF-TOKEN=d4662f13-52db-4f2d-8cc0-f9cfaefac94d'
        }

    @patch('src.d04_randim.call_randim.requests.get')
    def test_api_randim_test(self, mocked_get):
        mocked_get.return_value.status_code = 200
        result = self.randim_api.api_randim_test()
        self.assertIsInstance(result, dict)

    @patch('src.d04_randim.call_randim.requests.get')
    def test_api_randim_test_status_scode(self, mocked_get):
        mocked_get.return_value.status_code = 200
        result = self.randim_api.api_randim_test()
        self.assertEqual(200, result['Status_code'])

    @patch('src.d04_randim.call_randim.requests.get')
    def test_error_api_randim_test(self, mocked_get):
        mocked_get.return_value.status_code = 404
        result = self.randim_api.api_randim_test()
        self.assertEqual(404, result['Status_code'])

    @patch('src.d04_randim.call_randim.requests.post')
    def test_api_lte_compute_status_code_200(self, mock_request):
        # Mocking the load_files_lte_compute function
        with patch.object(ApiRandim,'load_files_lte_compute') as mock_load_file:
            # Mocking the return value of load_files_lte_compute
            mock_load_file.return_value = [
                ('inputLTE', ('config_file_fdd.json', MagicMock(), 'application/json')),
                ('importedFile', ('LTE_all_forecasted_FDD_from_capacity.xlsx',MagicMock(),
                            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))
            ]

            # Create a mock response
            mock_response = MagicMock()
            mock_response.status_code = 202
            mock_response.headers = {'Location': "mocked_location"}
            mock_response.content = b'Mocked Content'
            mock_request.return_value = mock_response

            # Call the function api_lte_compute
            result = self.randim_api.api_lte_compute('mocked_path_lte_foreacasted',
                                                     'mocked_path_config_file')
            # Assertions
            self.assertEqual(result['Status_code'], 202)
            self.assertEqual(result['id'], 'mocked_location')
            self.assertEqual(result['message'], b'Mocked Content')

            # Check if load_file_lte_compiute was called with the correct arguments
            mock_load_file.assert_called_with('mocked_path_config_file',
                                              'mocked_path_lte_foreacasted')
            #mock_request.post.assert_called_with('mock_base_url/lte/compute',
            #                                     headers={},
            #                                     data = {},
            #                                     files= [
            #    ('inputLTE', ('config_file_fdd.json', MagicMock(), 'application/json')),
            #    ('importedFile', ('LTE_all_forecasted_FDD_from_capacity.xlsx',MagicMock(),
            #                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))
            #],
            #                                     verify='mock_verify',
            #                                     timeout='mock_request_timeout'
#
            #    )

    @unittest.skip('Deprecetaed')
    @patch('src.d04_randim.call_randim.requests.post')
    def test_api_lte_compute_id_ask(self, mocked_post):
        path_lte_forecasted = (os.path.join(conf['PATH']['RANDIM'], 'LTE_all_forecasted_') +
                               conf['USE_CASE'] + '_from_capacity.xlsx')
        path_config_file = os.path.join(conf['PATH']['RANDIM'], 'run_OMA', conf['USE_CASE'],
                                        'template_builder',
                                        'config_file_') + conf['USE_CASE'] + '.json'
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {'Location': 'my_id'}
        mocked_post.return_value = mock_response
        result = self.randim_api.api_lte_compute(path_lte_forecasted, path_config_file)
        self.assertEqual('my_id', result['id'])
        self.assertEqual(202, result['Status_code'])


    @patch('src.d04_randim.call_randim.requests.post')
    def test_api_lte_compute_id_ask_error(self, mock_request):
        # Mocking the load_files_lte_compute function
        with patch.object(ApiRandim,'load_files_lte_compute') as mock_load_file:
            # Mocking the return value of load_files_lte_compute
            mock_load_file.return_value = [
                ('inputLTE', ('config_file_fdd.json', MagicMock(), 'application/json')),
                ('importedFile', ('LTE_all_forecasted_FDD_from_capacity.xlsx', MagicMock(),
                            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))
            ]

            # Create a Mock Reponse
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_request.return_value = mock_response

            # Call the function api_lte_compute
            result = self.randim_api.api_lte_compute('mocked_path_lte_foreacasted',
                                                     'mocked_path_config_file')
            # Assertions
            self.assertEqual(result['Status_code'], 404)

            # Check if load_file_lte_compute was called with the correct arguments
            mock_load_file.assert_called_with('mocked_path_config_file',
                                              'mocked_path_lte_foreacasted')

    @patch('requests.get')
    def test_check_progress_lte(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 200
        result = self.randim_api.check_progress_lte(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/lte/progress/my_ask',
            headers=self.expected_headers, timeout=350,verify=False)
        self.assertEqual(200, result['Status_code'])

    @patch('requests.get')
    def test_check_progress_lte_error(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 400
        result = self.randim_api.check_progress_lte(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/lte/progress/my_ask',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual(None, result)

    @patch('requests.get')
    def test_get_result_lte(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.content = 'my_content'
        result = self.randim_api.get_result_lte(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/lte/export/my_ask/1',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual('my_content', result)

    @patch('requests.get')
    def test_get_result_lte_error(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 400
        result = self.randim_api.get_result_lte(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/lte/export/my_ask/1',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual(400, result['Status_code'])
        self.assertIsInstance(result, dict)

        #############################################################
        # Densif
    @unittest.skip('TO MODIFY')
    @patch('src.d04_randim.call_randim.requests.post')
    def test_densification_compute_status_code_200(self, mocked_post):
        path_parameters = os.path.join(conf['PATH']['RANDIM'], 'densification',
                                       'parametersInput.json')
        path_congestion_forecasted = (
                    os.path.join(conf['PATH']['RANDIM'], 'congestion_forecasted') +
                    conf['USE_CASE'] + '_from_capacity.xlsx')
        path_topology_randim = os.path.join(conf['PATH']['RANDIM'], 'topology_randim.xlsx')
        mocked_post.return_value.status_code = 202
        result = self.randim_api.densification_compute(
            path_parameters, path_congestion_forecasted, path_topology_randim)
        self.assertEqual(202, result['Status_code'])

    @unittest.skip('TO MODIFY')
    @patch('src.d04_randim.call_randim.requests.post')
    def test_densification_compute_status_id_ask(self, mocked_post):
        path_parameters = os.path.join(conf['PATH']['RANDIM'],
                                       'densification', 'parametersInput.json')
        path_congestion_forecasted = (os.path.join(conf['PATH']['RANDIM'], 'congestion_forecasted')
                                      + conf[
                                          'USE_CASE'] + '_from_capacity.xlsx')
        path_topology_randim = os.path.join(conf['PATH']['RANDIM'], 'topology_randim.xlsx')
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {'Location': 'my_id'}
        mocked_post.return_value = mock_response
        result = self.randim_api.densification_compute(path_parameters, path_congestion_forecasted,
                                                       path_topology_randim)
        self.assertEqual('my_id', result['id_ask'])
        self.assertEqual(202, result['Status_code'])

    @unittest.skip('TO MODIFY')
    @patch('src.d04_randim.call_randim.requests.post')
    def test_api_densification_compute_id_ask_error(self, mocked_post):
        path_parameters = os.path.join(conf['PATH']['RANDIM'],
                                       'densification', 'parametersInput.json')
        path_congestion_forecasted = (
                    os.path.join(conf['PATH']['RANDIM'], 'congestion_forecasted') +
                    conf['USE_CASE'] + '_from_capacity.xlsx')
        path_topology_randim = os.path.join(conf['PATH']['RANDIM'], 'topology_randim.xlsx')
        mock_response = MagicMock()
        mock_response.status_code = 404
        mocked_post.return_value = mock_response
        result = self.randim_api.densification_compute(
            path_parameters, path_congestion_forecasted, path_topology_randim)
        self.assertEqual(404, result['Status_code'])


    @patch('requests.get')
    def test_check_progress_densification(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 200
        result = self.randim_api.check_progress_densification(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/densification/progress/my_ask',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual(200, result['Status_code'])

    @patch('requests.get')
    def test_check_progress_densification_error(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 400
        result = self.randim_api.check_progress_densification(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/densification/progress/my_ask',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual(None, result)

    @patch('requests.get')
    def test_get_result_densification(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.content = 'my_content'
        result = self.randim_api.get_result_densification(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/densification/export/my_ask/1',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual('my_content', result)

    @patch('requests.get')
    def test_get_result_densification_error(self, mocked_get):
        id_ask = 'my_ask'
        mocked_get.return_value.status_code = 400
        result = self.randim_api.get_result_densification(id_ask)
        self.expected_headers['Authorization'] = f'Bearer {self.randim_api.acces_token}'
        mocked_get.assert_called_once_with(
            'https://randim2.itn.intraorange/randim/densification/export/my_ask/1',
            headers=self.expected_headers, timeout=350, verify=False)
        self.assertEqual(400, result['Status_code'])
        self.assertIsInstance(result, dict)


class TestLoadFilesLTECompute(unittest.TestCase):
    def setUp(self):
        self.randim_api = ApiRandim()

    @patch('builtins.open')
    def test_load_files_lte_compute(self, mock_open_func):
        path_config_file = 'path_config_file.json'
        path_lte_forecasted = 'path_lte_forecasted.xlsx'

        config_file_content = b'config_data'
        lte_file_content = b'lte_data'

        # Mock the return values for open ()
        mock_config_file = BytesIO(config_file_content)
        mock_lte_file = BytesIO(lte_file_content)

        mock_open_func.side_effect = [mock_config_file, mock_lte_file]

        # call fucntion
        files = self.randim_api.load_files_lte_compute(path_config_file, path_lte_forecasted)
        print(files)
        expected_files = [(
            'inputLTE'
            , (
                'config_file_fdd.json'
                , mock_config_file,
                'application/json'
            )), (
            'importedFile'
            , (
                'LTE_all_forecasted_FDD_from_capacity.xlsx'
                , mock_lte_file,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ))]
        self.assertEqual(files[0][0], expected_files[0][0])
        self.assertEqual(files[0][1][0], expected_files[0][1][0])
        mock_open_func.assert_any_call(path_config_file, 'rb')
        mock_open_func.assert_any_call(path_lte_forecasted, 'rb')


if __name__ == '__main__':
    unittest.main()
