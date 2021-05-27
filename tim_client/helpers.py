import pandas as pd
import yaml
from pandas import DataFrame
from typing import Dict, List, Tuple
from datetime import datetime
from dateutil import parser
import logging
from copy import deepcopy


logger = logging.getLogger(__name__)


def dict_to_dataframe(dict_data: Dict, column_names: Dict[int, str] = {0: 'Timestamp', 1: 'Prediction'}) -> DataFrame:
    """Convert [Predictor] structure to pandas DataFrame."""
    df: DataFrame = pd.DataFrame(dict_data.items())

    df = df.rename(column_names, axis=1)
    df[column_names[0]] = df[column_names[0]].apply(lambda x: parse_timestamp(x))
    df.set_index(column_names[0], inplace=True)

    return df


def aggregated_predictions_to_dataframes(aggregated_predictions_list: List) -> List:
    aggregated_predictions = deepcopy(aggregated_predictions_list)
    try:
        for ag_pred in aggregated_predictions:
            ag_pred['values'] = dict_to_dataframe(ag_pred['values'], {0: 'Timestamp', 1: 'Prediction'})
        return aggregated_predictions
    except:
        logger.warning('Cannot convert aggregated predictions to Dataframes - returning raw aggregated predictions instead')
        return aggregated_predictions_list


def dataframe_to_request_dict(df: DataFrame, target_index: int = 1, timestamp_index: int = 0, predictors_update: Dict = None) -> Dict:
    # TODO Missing values?
    csv_string = df.to_csv(index=False, sep=',', na_rep='NaN')
    timestamp_column_name = df.columns[timestamp_index]
    target_column_name = df.columns[target_index]

    request_dict = {
        "csv": csv_string,
        "timestampColumn": timestamp_column_name,
        "targetColumn": target_column_name
    }

    if predictors_update is not None:
        request_dict['updates'] = predictors_update

    return request_dict


def parse_timestamp(timestamp_string: str) -> datetime:
    """
    Convert ISOstring date time to datetime object.

    :param date_string: String with date in ISOstring format.

    :return: datetime object.
    """
    try:
        datetime_object: datetime = parser.parse(timestamp_string)
        return datetime_object
    except:
        logger.warning('Unrecognized timestamp format %s , using default pattern %s', timestamp_string, "%Y-%m-%d %H:%M:%S")
        try:
            return datetime.strptime(timestamp_string, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            logger.error('Failed to parse timestamp %s : %s', timestamp_string, str(e))
            raise(e)


def load_dataset_from_csv_file(file_path: str, sep: str = ';') -> DataFrame:
    """Load CSV file to pandas DataFrame."""
    try:
        return pd.read_csv(file_path, sep=sep)
    except Exception as e:
        logger.error('Failed to load CSV file %s : %s', file_path, str(e))
        raise(e)


def load_configuration_from_yaml_file(file_path: str) -> Dict:
    """Load YAML file to Python dictionary."""
    try:
        with open(file_path) as yaml_file:
            yaml_dict: Dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    except Exception as e:
        logger.error('Failed to load YAML file %s : %s', file_path, str(e))
        raise(e)

    return yaml_dict
