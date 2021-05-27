from pandas import DataFrame
from typing import Dict, List, Tuple
import logging
from logging import Logger

from tim_client.helpers import dict_to_dataframe


class Detection:
    """TIM Anomaly Detection result."""

    __logger: Logger = None
    status: str = None
    request_uuid: str = None
    events: List[Dict] = None
    result_explanations: List[Dict] = None
    progress: float = None
    engine_result: str = None
    requested_configuration: Dict = None
    anomaly_indicator: DataFrame = None
    normal_behavior: DataFrame = None

    def __init__(self,
                 status: str,
                 request_uuid: str,
                 events: List[Dict] = None,
                 result_explanations: List[Dict] = None,
                 progress: float = None,
                 engine_result: str = None,
                 requested_configuration: Dict = None,
                 anomaly_indicator: DataFrame = None,
                 normal_behavior: DataFrame = None,
                 logger: Logger = None
                 ):
        self.status = status
        self.request_uuid = request_uuid
        self.events = events
        self.result_explanations = result_explanations
        self.progress = progress
        self.engine_result = engine_result
        self.requested_configuration = requested_configuration
        self.anomaly_indicator = anomaly_indicator
        self.normal_behavior = normal_behavior
        
        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return f'Detection {self.request_uuid}: {self.status}'

    def get_anomaly_indicator(self) -> DataFrame:
        """Return pandas DataFrame containing anomaly indicator values."""
        return self.anomaly_indicator

    def get_normal_behavior(self) -> DataFrame:
        """Return pandas DataFrame containing normal behavior values."""
        return self.normal_behavior
    
    @classmethod
    def from_json(cls, data):
        return cls(
            status=data['status'] if 'status' in data else None,
            request_uuid=data['requestUUID'] if 'requestUUID' in data else None,
            events=data['events'] if 'events' in data else None,
            result_explanations=data['resultExplanations'] if 'resultExplanations' in data else None,
            progress=float(data['progress']) if 'progress' in data else None,
            engine_result=data['engineResult'] if 'engineResult' in data else None,
            requested_configuration=data['requestedConfiguration'] if 'requestedConfiguration' in data else None,
            anomaly_indicator=dict_to_dataframe(data['anomalyIndicator']['values'], {0: 'Timestamp', 1: 'Anomaly Indicator'}) if 'anomalyIndicator' in data and 'values' in data['anomalyIndicator'] else None,
            normal_behavior=dict_to_dataframe(data['normalBehavior']['values'], {0: 'Timestamp', 1: 'Normal Behavior'}) if 'normalBehavior' in data and 'values' in data['normalBehavior'] else None
        )
