"""Module to call Randim """
import io
import json
import logging
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


class ApiRandim:
    """
    The ApiRandim class provides methods to interact with the Randim API for LTE and densification
    computations. It includes methods for authentication, submitting compute requests,
    checking progress, and retrieving results.

    """
    TIME_SLEEP_LTE = 300
    TIME_SLEEP_DENSIFICATION = 60
    REQUEST_TIMEOUT = 350
    TYPE_EXCEL_APPLICATION = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    def __init__(self):
        self.base_url = 'https://randim2.itn.intraorange/randim'
        self.check_url = f'{self.base_url}/2g/defaults'
        self.proxies = {'http': os.environ.get('HTTPPROXYRANDIM'),
                        'https': os.environ.get('HTTPSPROXYRANDIM')}
        self.verify = False
        self.headers = {
            'Authorization': '',
            'Cookie': 'JSESSIONID=820855AD21E0F686512C40B6E07DFDFE; '
                      'XSRF-TOKEN=d4662f13-52db-4f2d-8cc0-f9cfaefac94d'}
            #'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        self.acces_token = None

    def auth_orange(self):
        """
        The auth_orange method in the ApiRandim class authenticates with an external service using
        credentials stored in environment variables. It sends a GET request to the authentication
        URL, retrieves an access token from the response, logs the token, and stores it in the
        instance variable acces_token

        Returns
        -------
        acces_token: str
        """
        url = os.environ.get('AUTHURL')
        headers_auth = {
            'Authorization': os.environ.get('HEADERSAUTHORIZATION'),
            'Cookie': os.environ.get('HEADERSCOOKIE')
        }
        payload = {}
        response = requests.request("GET", url, headers=headers_auth, data=payload,
                                    timeout=ApiRandim.REQUEST_TIMEOUT)
        text_response = response.json()
        acces_token = text_response['access_token']
        logging.info("Access Token is ready")
        self.acces_token = acces_token
        return acces_token

    def api_randim_test(self):
        """
        The api_randim_test method checks the availability of the Randim API by sending a GET
        request to a specific endpoint. It returns a dictionary with the status and content of the
        response.

        Returns
        -------
        result_request: dict
            A dictionary containing a message, status code, and content of the response.

        """
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        print(self.check_url)
        request = requests.get(self.check_url, verify=self.verify, headers=self.headers,
                               timeout=ApiRandim.REQUEST_TIMEOUT)
        if request.status_code == 200:
            result_request = {'Message': 'Randim API is ready',
                              'Status_code': request.status_code,
                              'content': request.content}
            return result_request
        result_request = {'Message': 'Randim API is not ready',
                          'Status_code': request.status_code,
                          'content': request.content}
        return result_request

    def api_lte_compute(self, path_lte_forecasted, path_config_file):
        """
        The api_lte_compute method in the ApiRandim class sends a POST request to the Randim API to
        compute LTE forecasts. It handles file loading, authentication, and response processing.

        Parameters
        ----------
        path_lte_forecasted: str
            Path to the LTE forecasted file.
        path_config_file: str
            Path to the JSON configuration file.

        Returns
        -------
        result_request: dict
            A dictionary containing the task ID, response message, and status code if the request
            is successful.
            A dictionary containing the response message and status code
            if the request fails
        """
        files = self.load_files_lte_compute(path_config_file, path_lte_forecasted)
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        request = requests.post(f'{self.base_url}/lte/compute', headers=self.headers,
                                data={}, files=files, verify=self.verify,
                                timeout=ApiRandim.REQUEST_TIMEOUT)

        if request.status_code == 202:
            headers = request.headers
            id_ask = headers.get('Location')
            result_request = {'id': id_ask,
                              'message': request.content,
                              'Status_code': request.status_code}

            return result_request
        result_request = {'Message': request.content,
                          'Status_code': request.status_code}
        return result_request

    def load_files_lte_compute(self, path_config_file, path_lte_forecasted):
        """
        The load_files_lte_compute method reads the contents of a configuration file and an LTE
        forecasted file, then packages them into a list of tuples suitable for a
        multipart/form-data POST request.

        Parameters
        ----------
        path_config_file: str
            Path to the JSON configuration file.
        path_lte_forecasted: str
            Path to the LTE forecasted file.
        Returns
        -------
        files: list (of IO Files)
        """
        with open(path_config_file, 'rb') as config_file, open(path_lte_forecasted,
                                                               'rb') as lte_file:
            config_file_content = config_file.read()
            lte_file_content = lte_file.read()
            files = [
                ('inputLTE', ('config_file_fdd.json', io.BytesIO(config_file_content),
                              'application/json')),
                ('importedFile', ('LTE_all_forecasted_FDD_from_capacity.xlsx',
                                  io.BytesIO(lte_file_content), 'application',
                                  ApiRandim.TYPE_EXCEL_APPLICATION))
            ]
        return files

    def check_progress_lte(self, id_ask):
        """
        The check_progress_lte method checks the progress of an LTE compute request by sending a
        GET request to a specific endpoint and returns the status code and content if successful

        Parameters
        ----------
        id_ask: str
            The id of the task

        Returns
        -------
        dict: dict
            A dictionary containing the status code and content of the response if successful.
            None if the request fails.
        """
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        request = requests.get(f'{self.base_url}/lte/progress/{id_ask}', verify=self.verify,
                               headers=self.headers, timeout=ApiRandim.REQUEST_TIMEOUT)
        if request.status_code == 200:
            return {'Status_code': request.status_code,
                    'content': request.content}
        return None

    def get_result_lte(self, id_ask):
        """
        The get_result_lte method retrieves the result of an LTE compute request by sending a GET
        request to a specific endpoint and returns the response content if successful

        Parameters
        ----------
        id_ask: str
            Id of the task

        Returns
        -------
        If successful, returns the response content as bytes.
        If unsuccessful, returns a dictionary with the status code.

        """
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        request = requests.get(f'{self.base_url}/lte/export/{id_ask}/1', verify=self.verify,
                               headers=self.headers, timeout=ApiRandim.REQUEST_TIMEOUT)
        if request.status_code == 200:
            data = request.content
            return data
        return {'Status_code': request.status_code}

    def get_congestion_forecasted(self, path_forecasted_file, path_config_file,
                                  path_save_output_file):
        """
        This function launch all the step to get the result of the task /compute/lte
        First he request with post request /compute/lte
        Second he use get method to check progress
        Finally he use get method do download excel file

        Parameters
        ----------
        path_forecasted_file: str
            Path where LTE_all_forecasted file is located
        path_config_file: str
            Path were json configuration file is located
        path_save_output_file: str
            Path where we want to save excel file
        Returns
        -------
        dict_result: dict
            Result's information
        """
        print('First Step: Ask Randim to compute congestion')
        result_compute = self.api_lte_compute(path_forecasted_file, path_config_file)
        print(result_compute)
        id_ask = result_compute['id']
        print(f"The id of the task is {id_ask}")
        print('Check progress of Task')
        while True:
            result_check_progress = self.check_progress_lte(id_ask=id_ask)
            content = json.loads(result_check_progress['content'])
            statuses = content['statuses']
            if all(status in ('FINISHED', 'FAILED') for status in statuses):
                print(f"Task {id_ask} has finished")
                print('Download Result')
                result = self.get_result_lte(id_ask=id_ask)
                with open(path_save_output_file, 'wb') as f:
                    f.write(result)
                print(f"File Save in {path_save_output_file}")
                dict_result = {'status': 'Request Finished and file save',
                               'file_location': path_save_output_file}
                return dict_result

            status_count = {}
            for status in statuses:
                if status in status_count:
                    status_count[status] += 1
                else:
                    status_count[status] = 1
            for status, count in status_count.items():
                print(f"Nb status '{status}': {count}")

            time.sleep(ApiRandim.TIME_SLEEP_LTE)
            # Densification 5 minutes

    def densification_compute(self, path_parameters, path_congestion_forecasted,
                              path_topology_randim):
        """
        Function to call Randim endpoint for densification

        Parameters
        ----------
        path_parameters: str
        path_congestion_forecasted: str
        path_topology_randim: str

        Returns
        -------
        dict: dict
        """
        files = self.load_files_densification_compute(path_congestion_forecasted, path_parameters,
                                                      path_topology_randim)
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        request = requests.post(f'{self.base_url}/densification/compute', headers=self.headers,
                                files=files, data={}, verify=self.verify,
                                timeout=ApiRandim.REQUEST_TIMEOUT)

        if request.status_code == 202:
            headers = request.headers
            id_ask = headers.get('Location')
            return {"id_ask": id_ask,
                    "message": request.content,
                    "Status_code": request.status_code}

        return {'Status_code': request.status_code,
                'Message': request.content}

    def load_files_densification_compute(self, path_congestion_forecasted, path_parameters,
                                         path_topology_randim):
        with open(
                path_parameters, 'rb') as parameters, open(
            path_congestion_forecasted, 'rb') as congestion_forecasted, open(
            path_topology_randim, 'rb') as topology_randim:
            files = [
                ('densificationInput', ('config_file_fdd.json', parameters, 'application/json')),
                ('randimFile', ('congestion_forecasted_FDD.xlsx', congestion_forecasted,
                                ApiRandim.TYPE_EXCEL_APPLICATION)),
                ('topologyFile', ('topology_randim.xlsx', topology_randim,
                                  ApiRandim.TYPE_EXCEL_APPLICATION))
            ]
        return files

    def check_progress_densification(self, id_ask):
        """
        This function is to check the progress of compute request for densification

        Parameters
        ----------
        id_ask: str
            The id of the task

        Returns
        -------
        dict: dict
        """
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        request = requests.get(f'{self.base_url}/densification/progress/{id_ask}',
                               verify=self.verify, headers=self.headers,
                               timeout=ApiRandim.REQUEST_TIMEOUT)
        if request.status_code == 200:
            return {'Status_code': request.status_code,
                    'content': request.content}
        return None

    def get_result_densification(self, id_ask):
        """
        This function is to get result of compute request

        Parameters
        ----------
        id_ask: str
            Id of the task

        Returns
        -------
        data: bytes
        """
        self.headers['Authorization'] = f'Bearer {self.auth_orange()}'
        request = requests.get(f'{self.base_url}/densification/export/{id_ask}/1',
                               verify=self.verify, headers=self.headers,
                               timeout=ApiRandim.REQUEST_TIMEOUT)
        if request.status_code == 200:
            data = request.content
            return data
        return {'Status_code': request.status_code}

    def get_result_randim(self, path_parameters, path_congestion_forecasted, path_topology_randim,
                          path_save_output_file):
        """
        This function launch all the step to get the result of the task /compute/lte
        First he request with post request /compute/lte
        Second he use get method to check progress
        Finally he use get method do download excel file

        Parameters
        ----------
        path_forecasted_file: str
            Path where LTE_all_forecasted file is located
        path_config_file: str
            Path were json configuration file is located
        path_save_output_file: str
            Path where we want to save excel file
        Returns
        -------
        dict_result: dict
            Result's information
        """
        print('First Step: Ask Randim to compute Densification')
        result_compute = self.densification_compute(path_parameters, path_congestion_forecasted,
                                                    path_topology_randim)
        print(result_compute)
        id_ask = result_compute['id_ask']
        print(f"The id of the task is {id_ask}")
        print('Check progress of Task')
        while True:
            result_check_progress = self.check_progress_densification(id_ask=id_ask)
            content = json.loads(result_check_progress['content'])
            print(content)
            status = content['status']
            if status == "FINISHED":
                print(f"Task {id_ask} has finished")
                print('Download Result')
                result = self.get_result_densification(id_ask=id_ask)
                with open(path_save_output_file, 'wb') as f:
                    f.write(result)
                print(f"File Save in {path_save_output_file}")
                dict_result = {'status': 'Request Finished and file save',
                               'file_location': path_save_output_file}
                return dict_result
            time.sleep(ApiRandim.TIME_SLEEP_DENSIFICATION)


if __name__ == '__main__':
    randim_api = ApiRandim()
