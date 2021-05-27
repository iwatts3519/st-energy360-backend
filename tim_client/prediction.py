from typing import Dict, List, Tuple
from pandas import DataFrame
import logging
from logging import Logger
import numpy as np
import copy

from tim_client.helpers import dict_to_dataframe, aggregated_predictions_to_dataframes


class Prediction:
    """TIM Prediction result."""

    __logger: Logger = None
    status: str = None
    request_uuid: str = None
    events: List[Dict] = None
    result_explanations: List[Dict] = None
    progress: float = None
    engine_result: str = None
    requested_configuration: Dict = None
    predictors_importances: Dict = None
    prediction: DataFrame = None
    prediction_intervals_lower_values: DataFrame = None
    prediction_intervals_upper_values: DataFrame = None
    aggregated_predictions: List[Dict] = None
    raw_predictions: List[Dict] = None
    data_difficulty: float = None

    def __init__(self,
                 status: str,
                 request_uuid: str,
                 events: List[Dict] = None,
                 result_explanations: List[Dict] = None,
                 progress: float = None,
                 engine_result: str = None,
                 requested_configuration: Dict = None,
                 predictors_importances: Dict = None,
                 prediction: DataFrame = None,
                 prediction_intervals_lower_values: DataFrame = None,
                 prediction_intervals_upper_values: DataFrame = None,
                 aggregated_predictions: List[Dict] = None,
                 raw_predictions: List[Dict] = None,
                 data_difficulty: float = None,
                 logger: Logger = None
                 ):
        self.status = status
        self.request_uuid = request_uuid
        self.events = events
        self.result_explanations = result_explanations
        self.progress = progress
        self.engine_result = engine_result
        self.requested_configuration = requested_configuration
        self.predictors_importances = predictors_importances
        self.prediction = prediction
        self.prediction_intervals_lower_values = prediction_intervals_lower_values
        self.prediction_intervals_upper_values = prediction_intervals_upper_values
        self.aggregated_predictions = aggregated_predictions
        self.raw_predictions = raw_predictions
        self.data_difficulty = data_difficulty

        if self.__logger is None:
            self.__logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return f'Prediction {self.request_uuid}: {self.status}'

    def get_prediction(self, include_intervals: bool = False) -> DataFrame:
        """Return pandas DataFrame containing predicton values."""
        if include_intervals:
            result = copy.deepcopy(self.prediction)

            if self.prediction_intervals_lower_values is not None:
                result = result.join(self.prediction_intervals_lower_values)
            else:
                result['LowerValues'] = np.nan

            if self.prediction_intervals_upper_values is not None:
                result = result.join(self.prediction_intervals_upper_values)
            else:
                result['UpperValues'] = np.nan

            return result
        else:
            return self.prediction

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
            predictors_importances=data['predictorsImportances'] if 'predictorsImportances' in data else None,
            prediction=dict_to_dataframe(data['prediction']['values'], {0: 'Timestamp', 1: 'Prediction'}) if 'prediction' in data and 'values' in data['prediction'] else None,
            prediction_intervals_lower_values=dict_to_dataframe(data['prediction']['predictionIntervals']['lowerValues'], {0: 'Timestamp', 1: 'LowerValues'}
                                                                ) if 'prediction' in data and 'predictionIntervals' in data['prediction'] and 'lowerValues' in data['prediction']['predictionIntervals'] else None,
            prediction_intervals_upper_values=dict_to_dataframe(data['prediction']['predictionIntervals']['upperValues'], {0: 'Timestamp', 1: 'UpperValues'}
                                                                ) if 'prediction' in data and 'predictionIntervals' in data['prediction'] and 'upperValues' in data['prediction']['predictionIntervals'] else None,
            aggregated_predictions=aggregated_predictions_to_dataframes(data['aggregatedPredictions']) if 'aggregatedPredictions' in data else None,
            raw_predictions=data['rawPredictions'] if 'rawPredictions' in data else None,
            data_difficulty=float(data['dataDifficulty']) if 'dataDifficulty' in data else None
        )
