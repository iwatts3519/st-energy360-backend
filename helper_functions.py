import pandas as pd
import json
import numpy as np
import logging
import requests
import tim_client
from datetime import datetime, timedelta


def get_deops(n):
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    n_days_past = (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")
    auth123 = ('iwatts', 'Iwatts371!')
    headers = {'sp': 'energykit', 'apikey': '45d8296f-aca7-46e7-888d-bd87f6e8e150'}
    while True:
        try:
            deops = requests.get(
                f'https://energykit.deop.siemens.com/assets/v0.1/rawTimeseriesByFeedId/5e9f1f46143d340018ed853f?limit'
                f'=1344&since={n_days_past}&until={now}',
                headers=headers, auth=auth123)
            break
        except:
            continue
    values = deops.json()
    df = pd.DataFrame(values['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df[['timestamp', 'value']]
    df.value = df.value / 1000
    df.rename(columns={'value': 'PV_obs'}, inplace=True)
    df = df[df['timestamp'].dt.minute == 0]
    df.to_csv('./Data/deops.csv', index=False)
    return df


def get_solcast_forecast():
    df_old = pd.read_csv('Data/historical_forecast.csv', sep=',')

    creds = {'api_key': '1h3MqOk4r2Vb2X_9uexzYFkUVWzBHz6w'}
    solcast_url = 'https://api.solcast.com.au/weather_sites/9d7c-6430-2d41-5c4b/forecasts?format=json'
    while True:
        try:
            response = requests.get(solcast_url, params=creds)
            break
        except:
            continue
    forecast_json = response.json()
    forecast_df = pd.DataFrame(list(forecast_json.values())[0])
    df_new = pd.concat([df_old, forecast_df], ignore_index=True)
    df_new.drop_duplicates(subset='period_end', keep='last', inplace=True, ignore_index=True)
    df_new.to_csv('./Data/historical_forecast.csv', sep=',', index=False)
    df_new = clean_solcast_data(df_new)
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True)
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
    while True:
        try:
            credentials = tim_client.Credentials(credentials_json['license_key'], credentials_json['email'],
                                                 credentials_json['password'], tim_url=TIM_URL)
            api_client = tim_client.ApiClient(credentials)
            api_client.save_json = SAVE_JSON
            api_client.json_saving_folder_path = JSON_SAVING_FOLDER
            break
        except:
            continue
    # --------------------------------------------------------
    configuration_backtest = {
        "usage": {
            "predictionTo": {
                "baseUnit": "Sample",
                "offset": md_h
            },
            "backtestLength": backnum,
            "modelQuality": [{'day': i, 'quality': 'High'} for i in range(0, 6)]
        },
        "maxModelComplexity": 10,
        "predictionIntervals": {
            "confidenceLevel": 50
        },
        "extendedOutputConfiguration": {
            "returnSimpleImportances": True,
            "returnExtendedImportances": True,
            'returnAggregatedPredictions': True
        },
    }

    while True:
        try:
            model = api_client.prediction_build_model_predict(df, configuration_backtest)
            break
        except:
            continue

    if not indicator:
        df = model.prediction.reset_index()
        df.columns = ['timestamp', 'PV_obs']
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

# def get_missing_index(df):
#     missing = pd.isna(df.PV_obs)
#     end = len(df)
#     start = 0
#     check = 0
#     for i, value in enumerate(missing):
#         if not value and check == 0:
#             continue
#         elif not value and check == 1:
#             end = i
#             break
#         else:
#             if check == 0:
#                 start = i
#                 check = 1
#     print(f'Start is {start}')
#     print(f'End is {end}')
#     print(f'End minus Start is {end - start}')
#     return end, end - start

# def process_missing_values(df):
#     df_len = len(df)
#     end, prediction = get_missing_index(df)
#     if end == df_len:
#         return df, True
#     short_df = df[:end]
#     model_df = build_model(short_df, False, short_df['PV_obs'].isna().sum())
#
#     updated_df = merge_data_frames(df, model_df)
#     return updated_df, False

# def add_solcast_historical(df, update_historical):
#     if update_historical:
#         solcast_df = pd.read_csv('Data/Solcast_Historical.csv',
#                                  sep=',',
#                                  usecols=[1, 3, 4, 5, 6, 7, 8, 9],
#                                  header=0,
#                                  names=['timestamp', 'Temp', 'SA', 'CO', 'DHI', 'DNI', 'GHI', 'SZ'],
#                                  parse_dates=['timestamp'])
#
#         solcast_cols = ['timestamp', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
#         solcast_df = solcast_df.reindex(solcast_cols, axis=1)
#         solcast_df['timestamp'] = pd.to_datetime(solcast_df['timestamp'], utc=True)
#     else:
#         solcast_df = pd.read_csv('./Data/MainHistorical.csv',
#                                  sep=',')
#         solcast_df['timestamp'] = pd.to_datetime(solcast_df['timestamp'], utc=True)
#     df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
#     solcast_historical_df = df.merge(solcast_df, on='timestamp', how='left', indicator=True)
#
#     solcast_historical_df = solcast_historical_df.reindex(
#         ['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp'],
#         axis=1)
#     return solcast_historical_df
