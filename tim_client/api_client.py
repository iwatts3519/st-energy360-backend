"""
TIM Python Client

TIM Engine Swagger: https://timws.tangent.works/v4/swagger-ui.html
TIM Engine Redoc:   https://timws.tangent.works/v4/redoc.html
"""
import pandas as pd
from pandas import DataFrame
from typing import Dict, List, Tuple, Union
import yaml
import requests
from time import sleep
from requests import Response
import json
import logging
from logging import Logger
import functools
from datetime import datetime
import os
import inspect
from pathlib import Path

from tim_client.credentials import Credentials

from tim_client.prediction_model import PredictionModel
from tim_client.detection_model import DetectionModel

from tim_client.prediction import Prediction
from tim_client.detection import Detection

from tim_client.helpers import dict_to_dataframe, dataframe_to_request_dict

FREQUENCY_OF_REQUESTS = 3  # in seconds, how often we ask for status
MAX_NUMBER_OF_TRIES = 600  # number of tries, after which waiting for result is abondoned
DEFAULT_JSON_SAVING_FOLDER = 'logs/'


class ApiClient:

    __credentials: Credentials = None
    __logger: Logger = None

    __default_headers: Dict = {}
    __verify_ssl: bool = True  # ONLY for debug purposes

    __save_json: bool = False
    __json_saving_folder_path: Path = Path(DEFAULT_JSON_SAVING_FOLDER)

    def __init__(self, credentials: Credentials, logger: Logger = None, default_headers: Dict = {}, verify_ssl: bool = True):
        self.__credentials = credentials
        self.__logger = logger

        self.__default_headers = default_headers
        self.__verify_ssl = verify_ssl

        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)

    @property
    def save_json(self) -> bool:
        return self.__save_json

    @save_json.setter
    def save_json(self, save_json: bool):
        self.__save_json = save_json
        self.__logger.info('Saving JSONs functionality has been %s', 'enabled' if save_json else 'disabled')

    @property
    def json_saving_folder_path(self) -> Path:
        return self.__json_saving_folder_path

    @json_saving_folder_path.setter
    def json_saving_folder_path(self, json_saving_folder_path: str):
        self.__json_saving_folder_path = Path(json_saving_folder_path)
        self.__logger.info('JSON destination folder changed to %s', str(self.__json_saving_folder_path))

    def wait_for_task_to_finish(self, uri: str) -> Tuple[int, Dict]:
        """Keep getting task result until task is finished or tries threshold is reached."""
        counter_of_tries: int = 0
        while counter_of_tries < MAX_NUMBER_OF_TRIES:
            get_status, get_response = self.send_tim_request('GET', uri)
            self.__logger.debug('Waiting for task to finish: %s: %s%s (%d/%d)', get_response["status"], get_response["progress"] if "progress" in get_response else 0.0, '%', counter_of_tries+1, MAX_NUMBER_OF_TRIES)

            if(200 <= get_status < 300):
                if self.is_request_finished(get_response['status']):
                    return get_status, get_response
                else:
                    counter_of_tries += 1
                    sleep(FREQUENCY_OF_REQUESTS)

            else:
                self.__logger.error(f'Get request error: {get_status} {get_response}')
                raise ValueError(f'Get request error, status code: {get_status}')

        self.__logger.error('Waiting for task to finish exceeded limit! Terminating.')
        raise ValueError('Waiting for task to finish exceeded limit')

    def prediction_build_model(self, data: DataFrame, configuration: Dict, wait_to_finish: bool = True, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> PredictionModel:
        """
        Build prediction model.

        :param data: Dataset in pandas DataFrame format
        :param configuration: Configuration object of prediction model
        :param wait_to_finish: Optional flag to wait for the built model, default is True

        :return: Build prediction model and wait to finish building if the wait_to_finish argument is True else returns build model with current status.
        """
        request_payload: Dict = {
            'configuration': configuration,
            'data': dataframe_to_request_dict(data, target_index=target_index, timestamp_index=timestamp_index, predictors_update=predictors_update)
        }
        
        now = datetime.now()
        self.__log_request(now, 'prediction-build-model', self.__credentials.get_auth_headers(), request_payload)

        self.__logger.debug('Sending POST request to /prediction/build-model')
        request_status, request_response = self.send_tim_request('POST', f'/prediction/build-model', request_payload)
        self.__logger.debug('Response from /prediction/build-model: %d', request_status)

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.prediction_build_model: {request_status} {request_response}')
            raise ValueError(f'ApiClient.prediction_build_model: {request_status}')

        request_uuid: str = request_response['requestUUID']
        self.__logger.debug('UUID of /prediction/build-model request: %s', request_uuid)

        if wait_to_finish:
            _, request_response = self.wait_for_task_to_finish(f'/prediction/build-model/{request_uuid}')

        self.__log_response(now, 'prediction-build-model', request_response)
        return PredictionModel.from_json(request_response)
    
    def prediction_build_model_detail(self, request_uuid: str) -> PredictionModel:
        """
        Get prediction build model detail.

        :param request_uuid: UUID of the predction model building request

        :return: Build prediction model with current build status. If failes returns None.
        """
        request_status, request_response = self.send_tim_request('GET', f'/prediction/build-model/{request_uuid}')

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.prediction_build_model_detail: {request_status} {request_response}')
            raise ValueError(f'ApiClient.prediction_build_model_detail: {request_status}')

        return PredictionModel.from_json(request_response)

    def prediction_predict(self, data: DataFrame, model: PredictionModel, configuration: Dict = None, wait_to_finish: bool = True, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> Prediction:
        """Make and return prediction."""
        request_payload: Dict = {
            'model': model.get_model(),
            'data': dataframe_to_request_dict(data, target_index=target_index, timestamp_index=timestamp_index, predictors_update=predictors_update)
        }

        if configuration is not None:
            request_payload['configuration'] = configuration

        now = datetime.now()
        self.__log_request(now, 'prediction-predict', self.__credentials.get_auth_headers(), request_payload)
        
        self.__logger.debug('Sending POST request to /prediction/predict')
        request_status, request_response = self.send_tim_request('POST', f'/prediction/predict', request_payload)
        self.__logger.debug('Response from /prediction/predict: %d', request_status)

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.prediction_predict: {request_status} {request_response}')
            raise ValueError(f'ApiClient.prediction_predict: {request_status}')

        request_uuid: str = request_response['requestUUID']
        self.__logger.debug('UUID of /prediction/predict request: %s', request_uuid)

        if wait_to_finish:
            _, request_response = self.wait_for_task_to_finish(f'/prediction/predict/{request_uuid}')
        
        self.__log_response(now, 'prediction-predict', request_response)
        return Prediction.from_json(request_response)
    
    def prediction_predict_detail(self, request_uuid: str) -> Prediction:
        request_status, request_response = self.send_tim_request('GET', f'/prediction/predict/{request_uuid}')

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.prediction_predict_detail: {request_status} {request_response}')
            raise ValueError(f'ApiClient.prediction_predict_detail: {request_status}')

        return Prediction.from_json(request_response)

    def prediction_build_model_predict(self, data: DataFrame, configuration: Dict, wait_to_finish: bool = True, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> Prediction:
        """
        Submit data and configuration to model building and consequent prediction process.

        :param data: Dataset in pandas DataFrame format
        :param configuration: Configuration object of prediction model
        :param wait_to_finish: Optional flag to wait for the built model, default is True

        :return: Build prediction model and wait to finish building if the wait_to_finish argument is True else returns build model with current status.
        """
        request_payload: Dict = {
            'configuration': configuration,
            'data': dataframe_to_request_dict(data, target_index=target_index, timestamp_index=timestamp_index, predictors_update=predictors_update)
        }

        now = datetime.now()
        self.__log_request(now, 'prediction-build-model-predict', self.__credentials.get_auth_headers(), request_payload)
        
        self.__logger.debug('Sending POST request to /prediction/build-model-predict')
        request_status, request_response = self.send_tim_request('POST', f'/prediction/build-model-predict', request_payload)
        self.__logger.debug('Response from /prediction/build-model-predict: %d', request_status)

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.prediction_build_model_predict: {request_status} {request_response}')
            raise ValueError(f'ApiClient.prediction_build_model_predict: {request_status}')

        request_uuid: str = request_response['requestUUID']
        self.__logger.debug('UUID of /prediction/build-model-predict request: %s', request_uuid)

        if wait_to_finish:
            _, request_response = self.wait_for_task_to_finish(f'/prediction/build-model-predict/{request_uuid}')

        self.__log_response(now, 'prediction-build-model-predict', request_response)
        return Prediction.from_json(request_response)
    
    def prediction_build_model_predict_detail(self, request_uuid: str) -> Prediction:
        request_status, request_response = self.send_tim_request('GET', f'/prediction/build-model-predict/{request_uuid}')

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.prediction_build_model_predict_detail: {request_status} {request_response}')
            raise ValueError(f'ApiClient.prediction_build_model_predict_detail: {request_status}')

        return Prediction.from_json(request_response)

    def detection_build_model(self, data: DataFrame, configuration: Dict = None, wait_to_finish: bool = True, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> DetectionModel:
        """Build and return anomaly detection model."""
        request_payload: Dict = {
            'data': dataframe_to_request_dict(data, target_index=target_index, timestamp_index=timestamp_index, predictors_update=predictors_update)
        }

        if configuration is not None:
            request_payload['configuration'] = configuration

        now = datetime.now()
        self.__log_request(now, 'detection-build-model', self.__credentials.get_auth_headers(), request_payload)
        
        self.__logger.debug('Sending POST request to /detection/build-model')
        request_status, request_response = self.send_tim_request('POST', f'/detection/build-model', request_payload)
        self.__logger.debug('Response from /detection/build-model: %d', request_status)

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.detection_build_model: {request_status} {request_response}')
            raise ValueError(f'ApiClient.detection_build_model: {request_status}')

        request_uuid: str = request_response['requestUUID']
        self.__logger.debug('UUID of /detection/build-model request: %s', request_uuid)

        if wait_to_finish:
            _, request_response = self.wait_for_task_to_finish(f'/detection/build-model/{request_uuid}')

        self.__log_response(now, 'detection-build-model', request_response)
        return DetectionModel.from_json(request_response)
    
    def detection_build_model_detail(self, request_uuid: str) -> DetectionModel:
        request_status, request_response = self.send_tim_request('GET', f'/detection/build-model/{request_uuid}')

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.detection_build_model_detail: {request_status} {request_response}')
            raise ValueError(f'ApiClient.detection_build_model_detail: {request_status}')

        return DetectionModel.from_json(request_response)

    def detection_rebuild_model(self, data: DataFrame, model: DetectionModel, configuration: Dict = None, wait_to_finish: bool = True, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> DetectionModel:
        """Rebuild and return anomaly detection model."""
        request_payload: Dict = {
            'data': dataframe_to_request_dict(data, target_index=target_index, timestamp_index=timestamp_index, predictors_update=predictors_update),
            'model': model.get_model()
        }

        if configuration is not None:
            request_payload['configuration'] = configuration

        now = datetime.now()
        self.__log_request(now, 'detection-rebuild-model', self.__credentials.get_auth_headers(), request_payload)
        
        self.__logger.debug('Sending POST request to /detection/rebuild-model')
        request_status, request_response = self.send_tim_request('POST', f'/detection/rebuild-model', request_payload)
        self.__logger.debug('Response from /detection/rebuild-model: %d', request_status)

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.detection_rebuild_model: {request_status} {request_response}')
            raise ValueError(f'ApiClient.detection_rebuild_model: {request_status}')

        request_uuid: str = request_response['requestUUID']
        self.__logger.debug('UUID of /detection/rebuild-model request: %s', request_uuid)

        if wait_to_finish:
            _, request_response = self.wait_for_task_to_finish(f'/detection/rebuild-model/{request_uuid}')

        self.__log_response(now, 'detection-rebuild-model', request_response)
        return DetectionModel.from_json(request_response)
    
    def detection_rebuild_model_detail(self, request_uuid: str) -> DetectionModel:
        request_status, request_response = self.send_tim_request('GET', f'/detection/rebuild-model/{request_uuid}')

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.detection_rebuild_model_detail: {request_status} {request_response}')
            raise ValueError(f'ApiClient.detection_rebuild_model_detail: {request_status}')

        return DetectionModel.from_json(request_response)

    def detection_detect(self, data: DataFrame, model: DetectionModel, wait_to_finish: bool = True, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> Detection:
        """Make anomaly detection and return anomaly indicator."""
        request_payload: Dict = {
            'data': dataframe_to_request_dict(data, target_index=target_index, timestamp_index=timestamp_index, predictors_update=predictors_update),
            'model': model.get_model()
        }
        
        now = datetime.now()
        self.__log_request(now, 'detection-detect', self.__credentials.get_auth_headers(), request_payload)

        self.__logger.debug('Sending POST request to /detection/detect')
        request_status, request_response = self.send_tim_request('POST', f'/detection/detect', request_payload)
        self.__logger.debug('Response from /detection/detect: %d', request_status)

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.detection_detect: {request_status} {request_response}')
            raise ValueError(f'ApiClient.detection_detect: {request_status}')

        request_uuid: str = request_response['requestUUID']
        self.__logger.debug('UUID of /detection/detect request: %s', request_uuid)

        if wait_to_finish:
            _, request_response = self.wait_for_task_to_finish(f'/detection/detect/{request_uuid}')

        self.__log_response(now, 'detection-detect', request_response)
        return Detection.from_json(request_response)
    
    def detection_detect_detail(self, request_uuid: str) -> Detection:
        request_status, request_response = self.send_tim_request('GET', f'/detection/detect/{request_uuid}')

        if request_status < 200 or request_status >= 300:
            self.__logger.error(f'ApiClient.detection_detect_detail: {request_status} {request_response}')
            raise ValueError(f'ApiClient.detection_detect_detail: {request_status}')

        return Detection.from_json(request_response)

    def send_tim_request(self, method: str, uri: str, payload: Dict = None, headers: Dict = {}) -> Tuple[int, Dict]:
        """Send HTTP request to TIM Engine."""
        method = method.upper()

        url = f'{self.__credentials.get_tim_url()}{uri}'

        all_headers = self.__default_headers.copy()
        all_headers.update(headers)
        all_headers.update(self.__credentials.get_auth_headers())

        # self.__logger.debug('Sending request: %s %s', method, url)
        response: Response = requests.request(method=method, url=url, headers=all_headers, json=payload)
        return response.status_code, response.json()

    def is_request_finished_successfully(self, status: str) -> bool:
        """
        Check if status represents successfully finished request.

        :param status: Status string returned from TIM Engine.

        :return: True if request is successfully finished.
        """
        return status in ('Finished', 'FinishedWithWarning')

    def is_request_failed(self, status: str) -> bool:
        """
        Check if status represents failed request.

        :param status: Status string returned from TIM Engine.

        :return: True if request has failed.
        """
        return status in ('NotFound', 'Failed')

    def is_request_finished(self, status: str) -> bool:
        """
        Check if status represents finished request (successfully or failed).

        :param status: Status string returned from TIM Engine.

        :return: True if request has finished.
        """
        return self.is_request_finished_successfully(status) or self.is_request_failed(status)

    def __create_directory_if_not_exist(self, destination_folder_path):
        if not os.path.isdir(destination_folder_path):
            self.__logger.debug('Creating new directory for JSON files: %s', str(destination_folder_path))
            os.mkdir(destination_folder_path)

    def __log_request(self, timestamp: datetime, endpoint: str, headers: Dict, payload: Dict):
        """Save TIM request to file."""
        if self.__save_json is False:
            return

        self.__create_directory_if_not_exist(self.__json_saving_folder_path)
        filename = f'{timestamp.strftime("%Y%m%d%H%M%S")}_request_{endpoint}.json'
        filepath = self.__json_saving_folder_path / filename

        output_json = {
            'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'tim_url': self.__credentials.get_tim_url(),
            'endpoint': endpoint,
            'headers': headers,
            'request_content': payload
        }

        self.__logger.debug('Saving request JSON to %s', str(filepath))

        try:
            with open(filepath, 'w') as outfile:
                json.dump(output_json, outfile)
        except Exception as e:
            self.__logger.error('Failed to save request JSON: %s', str(e))

    def __log_response(self, timestamp: datetime, endpoint: str, payload: Dict):
        """Save TIM response to file."""
        if self.__save_json is False:
            return

        uuid = payload['requestUUID'] if 'requestUUID' in payload else 'N/A'
        status = payload['status'] if 'status' in payload else 'N/A'
        
        self.__create_directory_if_not_exist(self.__json_saving_folder_path)
        filename = f'{timestamp.strftime("%Y%m%d%H%M%S")}_response_{endpoint}_{uuid}.json'
        filepath = self.__json_saving_folder_path / filename

        output_json = {
            'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'endpoint': endpoint,
            'request_uuid': uuid,
            'http_status': status,
            'response_content': payload
        }

        self.__logger.debug('Saving response JSON of %s to %s', uuid, str(filepath))
        
        try:
            with open(filepath, 'w') as outfile:
                json.dump(output_json, outfile)
        except Exception as e:
            self.__logger.error('Failed to save response JSON: %s', str(e))
