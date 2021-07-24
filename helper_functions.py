import pandas as pd
import json
import numpy as np
from time import time
import logging
import requests
import tim_client
import sqlite3

backtest = 0


def generate_full_historical():
    solcast_df = pd.read_csv('Data/Solcast_Historical.csv',
                             sep=',',
                             usecols=[1, 3, 4, 5, 6, 7, 8, 9],
                             header=0,
                             names=['timestamp', 'Temp', 'SA', 'CO', 'DHI', 'DNI', 'GHI', 'SZ'],
                             parse_dates=['timestamp'])

    solcast_cols = ['timestamp', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
    solcast_df = solcast_df.reindex(solcast_cols, axis=1)
    solcast_df['timestamp'] = pd.to_datetime(solcast_df['timestamp'], utc=True)
    historical_predictions_df = pd.read_csv('./Data/historical_predictions.csv')
    historical_predictions_df = clean_solcast_data(historical_predictions_df)
    historical_predictions_df.drop('PV_obs', axis=1, inplace=True)
    historical_predictions_df['timestamp'] = pd.to_datetime(historical_predictions_df['timestamp'], utc=True)
    test_merge_df = pd.concat([solcast_df, historical_predictions_df], ignore_index=True)
    test_merge_df.drop_duplicates(subset='timestamp', keep='last', inplace=True, ignore_index=True)
    test_merge_df.to_csv('./Data/MainHistorical.csv', sep=',', index=False)
    return test_merge_df


def get_missing_index(df):
    missing = pd.isna(df.PV_obs)
    end = len(df)
    start = 0
    check = 0
    for i, value in enumerate(missing):
        if not value and check == 0:
            continue
        elif not value and check == 1:
            end = i
            break
        else:
            if check == 0:
                start = i
                check = 1
    print(f'Start is {start}')
    print(f'End is {end}')
    print(f'End minus Start is {end - start}')
    return end, end - start


def add_solcast_historical(df, update_historical):
    if update_historical:
        solcast_df = pd.read_csv('Data/Solcast_Historical.csv',
                                 sep=',',
                                 usecols=[1, 3, 4, 5, 6, 7, 8, 9],
                                 header=0,
                                 names=['timestamp', 'Temp', 'SA', 'CO', 'DHI', 'DNI', 'GHI', 'SZ'],
                                 parse_dates=['timestamp'])

        solcast_cols = ['timestamp', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
        solcast_df = solcast_df.reindex(solcast_cols, axis=1)
        solcast_df['timestamp'] = pd.to_datetime(solcast_df['timestamp'], utc=True)
    else:
        solcast_df = pd.read_csv('./Data/MainHistorical.csv',
                                 sep=',')
        solcast_df['timestamp'] = pd.to_datetime(solcast_df['timestamp'], utc=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    solcast_historical_df = df.merge(solcast_df, on='timestamp', how='left', indicator=True)

    solcast_historical_df = solcast_historical_df.reindex(
        ['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp'],
        axis=1)
    return solcast_historical_df


def get_solcast_forecast():
    new_cols = ['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
    download = input("Do you want to download new weather data - Y?").lower()
    df_old = pd.read_csv('Data/historical_predictions.csv', sep=',')
    if download == 'y':
        print("Dowloading New Data")
        creds = {'api_key': '1h3MqOk4r2Vb2X_9uexzYFkUVWzBHz6w'}
        solcast_url = 'https://api.solcast.com.au/weather_sites/9d7c-6430-2d41-5c4b/forecasts?format=json'
        response = requests.get(solcast_url, params=creds)
        print(response.status_code)
        forecast_json = response.json()
        forecast_df = pd.DataFrame(list(forecast_json.values())[0])
        forecast_df.to_csv('./Data/new_forecast.csv', sep=',')
    else:
        print("Using Existing Data")
        forecast_df = pd.read_csv('./Data/new_forecast.csv', sep=',')
    df_old.to_csv('./DATA/df_old.csv', sep=',', index=False)
    forecast_df.to_csv('./DATA/forecast_df.csv', sep=',', index=False)
    df_new = pd.concat([df_old, forecast_df], ignore_index=True)
    df_new.drop_duplicates(subset='period_end', keep='last', inplace=True, ignore_index=True)
    df_new.to_csv('./Data/historical_predictions.csv', sep=',', index=False)
    df_new = clean_solcast_data(df_new)
    return df_new


def merge_data_frames(df1, df2):
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], utc=True).dt.tz_localize(None)
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], utc=True).dt.tz_localize(None)
    new_df = df1.merge(df2, on='timestamp', how='left', indicator=True)
    new_df.PV_obs_x = np.where(new_df.PV_obs_x.isna(),
                               new_df.PV_obs_y,
                               new_df.PV_obs_x
                               )
    for column in new_df.columns:
        if column.endswith('_y'):
            new_df.drop(column, axis=1, inplace=True)
    new_df.drop(['_merge'], axis=1, inplace=True)
    new_df.columns = ['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
    new_df = new_df.reindex(['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp'],
                            axis=1)
    return new_df


def process_missing_values(df):
    df_len = len(df)
    end, prediction = get_missing_index(df)
    if end == df_len:
        return df, True
    short_df = df[:end]
    model_df = build_model(short_df, False, short_df['PV_obs'].isna().sum())

    updated_df = merge_data_frames(df, model_df)
    return updated_df, False


def build_model(df, indicator, md_h):
    backnum = int(len(df) * .4)
    md_h = int(md_h)
    # --------------------------------------------------------
    with open('credentials.json') as f:
        credentials_json = json.load(f)  # loading the credentials from credentials.json

    TIM_URL = 'https://timws.tangent.works/v4/api'  # URL to which the requests are sent

    SAVE_JSON = False  # if True - JSON requests and responses are saved to JSON_SAVING_FOLDER
    JSON_SAVING_FOLDER = './logs/'  # folder where the requests and responses are stored

    LOGGING_LEVEL = 'INFO'

    level = logging.getLevelName(LOGGING_LEVEL)
    logging.basicConfig(level=level,
                        format='[%(levelname)s] %(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(message)s')
    logger = logging.getLogger(__name__)

    credentials = tim_client.Credentials(credentials_json['license_key'], credentials_json['email'],
                                         credentials_json['password'], tim_url=TIM_URL)
    api_client = tim_client.ApiClient(credentials)
    api_client.save_json = SAVE_JSON
    api_client.json_saving_folder_path = JSON_SAVING_FOLDER
    # --------------------------------------------------------
    configuration_backtest = {
        "usage": {
            "predictionTo": {
                "baseUnit": "Sample",
                "offset": md_h
            },
            "backtestLength": backnum
            # "modelQuality": [{'day': i, 'quality': 'Automatic'} for i in range(0, 6)]
        },
        "dataNormalization": False,
        # "features": [
        #     "Polynomial",
        #     "TimeOffsets",
        #     "PiecewiseLinear",
        #     "Intercept",
        #     "PeriodicComponents",
        #     "DayOfWeek",
        #     "MovingAverage"
        # ],
        "maxModelComplexity": 50,
        "timeSpecificModels": True,
        "allowOffsets": True,
        "memoryPreprocessing": True,
        "interpolation": {
            "type": "Linear",
            "maxLength": 6
        },
        "predictionIntervals": {
            "confidenceLevel": 90
        },
        "extendedOutputConfiguration": {
            "returnSimpleImportances": True,
            "returnExtendedImportances": True,
            'returnAggregatedPredictions': True,
            "predictionBoundaries": {
                "type": "Explicit",
                "maxValue": 100,
                "minValue": 0
            }
        }
    }

    start = time()
    model = api_client.prediction_build_model_predict(df, configuration_backtest)
    end = time()
    print(f'It took {end - start} seconds to build model')
    if not indicator:
        df = model.prediction.reset_index()
        df.columns = ['timestamp', 'PV_obs']
        return df
    else:
        return model


def clean_solcast_data(df):
    new_cols = ['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
    df.drop(['ghi90', 'ghi10', 'ebh', 'dni10', 'dni90', 'period'], axis=1, inplace=True)
    df['PV_obs'] = np.nan
    moved_cols = ['period_end', 'PV_obs', 'ghi', 'dni', 'dhi', 'azimuth', 'zenith', 'cloud_opacity', 'air_temp']
    df = df.reindex(moved_cols, axis=1)
    df.columns = new_cols
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
    df = df[df['timestamp'].dt.minute == 0]
    df = df.reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False).dt.tz_localize(None)
    return df
