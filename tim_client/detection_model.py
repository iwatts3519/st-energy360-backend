from typing import Dict, List
from pandas import DataFrame
import logging
from logging import Logger

from tim_client.helpers import dict_to_dataframe


class DetectionModel:
    """TIM model for Anomaly Detection."""

    __logger: Logger = None
    status: str = None
    request_uuid: str = None
    events: List[Dict] = None
    result_explanations: List[Dict] = None
    progress: float = None
    engine_result: str = None
    requested_configuration: Dict = None
    model: str = None
    data_offsets: List[Dict] = None
    predictors_importances: Dict = None
    anomaly_indicator: DataFrame = None
    normal_behavior: DataFrame = None
    sensitivity: float = None

    def __init__(self,
                 status: str,
                 request_uuid: str,
                 events: List[Dict] = None,
                 result_explanations: List[Dict] = None,
                 progress: float = None,
                 engine_result: str = None,
                 requested_configuration: Dict = None,
                 model: str = None,
                 data_offsets: List[Dict] = None,
                 predictors_importances: Dict = None,
                 anomaly_indicator: DataFrame = None,
                 normal_behavior: DataFrame = None,
                 sensitivity: float = None,
                 logger: Logger = None
                 ):
        self.status = status
        self.request_uuid = request_uuid
        self.events = events
        self.result_explanations = result_explanations
        self.progress = progress
        self.engine_result = engine_result
        self.requested_configuration = requested_configuration
        self.model = model
        self.data_offsets = data_offsets
        self.predictors_importances = predictors_importances
        self.anomaly_indicator = anomaly_indicator
        self.normal_behavior = normal_behavior
        self.sensitivity = sensitivity
        
        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return f'Detection Model {self.request_uuid}: {self.status}'
    
    def get_model(self) -> str:
        """Return string representing encrypted TIM Model."""
        return self.model

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
            model=data['model'] if 'model' in data else None,
            data_offsets=data['dataOffsets'] if 'dataOffsets' in data else None,
            predictors_importances=data['predictorsImportances'] if 'predictorsImportances' in data else None,
            anomaly_indicator=dict_to_dataframe(data['anomalyIndicator']['values'], {0: 'Timestamp', 1: 'Anomaly Indicator'}) if 'anomalyIndicator' in data and 'values' in data['anomalyIndicator'] else None,
            normal_behavior=dict_to_dataframe(data['normalBehavior']['values'], {0: 'Timestamp', 1: 'Normal Behavior'}) if 'normalBehavior' in data and 'values' in data['normalBehavior'] else None,
            sensitivity=float(data['sensitivity']) if 'sensitivity' in data else None
        )
