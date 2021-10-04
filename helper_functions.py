import pandas as pd
import json
import numpy as np
import logging
import requests
import tim_client
from datetime import datetime, timedelta


# This function gets the latest (n) days worth of data from the DEOPS system at Keele
def get_deops(n):
    # These two lines get the current date-time as a string, and the date-time n days ago
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    n_days_past = (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Using the above variables log into  DEOPS and retrieve the data
    auth123 = ('iwatts', 'Iwatts371!')
    headers = {'sp': 'energykit', 'apikey': '7edcd81d-f4fc-43ca-9d60-e7477ebcf0f1'}

    deops = requests.get(
        f'https://energykit.deop.siemens.com/assets/v0.1/rawTimeseriesByFeedId/5e9f1f46143d340018ed853f?limit'
        f'=1344&since={n_days_past}&until={now}',
        headers=headers, auth=auth123)

    # The data is returned in json format and then the values part of the json file is converted into a
    # dataframe
    values = deops.json()
    df = pd.DataFrame(values['data'])
    # Ensure the dataframe has the correct date-time format
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # Select only the columns of interest
    df = df[['timestamp', 'value']]
    # The api version of DEOPS returns Watts where as the CSV download used to create the historical data and used
    # elsewhere returns Kilowatts - hence the conversion
    df.value = df.value / 1000
    # Rename the columns to match other data frames used within the app
    df.rename(columns={'value': 'PV_obs'}, inplace=True)
    # Filter the dataframe to only use occurences on the hour as DEOPS gives data at 15 minute intervals
    df = df[2:-1]
    df = df.reset_index()
    agg_dict = {'timestamp': 'first', 'PV_obs': 'sum'}
    df = df.groupby(df.index // 4).agg(agg_dict)
    return df


def get_solcast_forecast():
    # Read in the current set of historical forecasts
    df_old = pd.read_csv('Data/historical_forecast.csv', sep=',')
    # Log into solcast and get a 7 day weather forecast
    creds = {'api_key': '1h3MqOk4r2Vb2X_9uexzYFkUVWzBHz6w'}
    solcast_url = 'https://api.solcast.com.au/weather_sites/9d7c-6430-2d41-5c4b/forecasts?format=json'

    response = requests.get(solcast_url, params=creds)

    # Extract data from json format and clean into a new dataframe
    forecast_json = response.json()
    forecast_df = pd.DataFrame(list(forecast_json.values())[0])
    # concatenate the original dataframe with the new one and write back to the same csv file, so that it is updated
    # with new values, getting rid of duplicates first
    df_new = pd.concat([df_old, forecast_df], ignore_index=True)
    df_new.drop_duplicates(subset='period_end', keep='last', inplace=True, ignore_index=True)
    df_new.to_csv('./Data/historical_forecast.csv', sep=',', index=False)
    # All of the data in this function so far is in Solcast format so I have written another function (below) that
    # cleans it into the format we need for this app
    df_new = clean_solcast_data(df_new)
    # Ensure timestamp is in correct date format
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], utc=True)
    return df_new


# A function used in several places to merge two dataframes together and keep in the correct format for this app
def merge_data_frames(df1, df2):
    # ensure that both dataframes have the same date format
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], utc=True).dt.tz_localize(None)
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], utc=True).dt.tz_localize(None)
    # Use pandas to perform a merge with indicator set to True, so that duplicate columns get a suffix of _x or _y
    new_df = df1.merge(df2, on='timestamp', how='left', indicator=True)
    # Uses the Numpy where function that takes PB_obs_x, created in the merge, and if it is NaN, replaces it with
    # PV_obs_y, else leaves it as PV_obs_x
    new_df.PV_obs_x = np.where(new_df.PV_obs_x.isna(),
                               new_df.PV_obs_y,
                               new_df.PV_obs_x
                               )
    # Loop through the columns and delete any that end with _y
    for column in new_df.columns:
        if column.endswith('_y'):
            new_df.drop(column, axis=1, inplace=True)
    # Drop the merge column, rename the columns and reindex the dataframe
    new_df.drop(['_merge'], axis=1, inplace=True)
    new_df.columns = ['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']
    new_df = new_df.reindex(['timestamp', 'PV_obs', 'GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp'],
                            axis=1)
    return new_df


# This is the function that calls on the TIM api to make a prediction. It takes three arguments - a dataframe,
# a boolean indicator of whether you wish it to return a model or a dataframe, and md_h which is the model horizon to
# forecast
def build_model(df, indicator, md_h):
    # Used to ensure that 40% of the dataframe is used for backtesting, so that metrics can be returned
    backnum = int(len(df) * .4)

    # ensures that the forecast horizon is in integer format
    md_h = int(md_h)

    # Set up the connection to the TIM API
    with open('credentials.json') as f:
        credentials_json = json.load(f)  # loading the credentials from credentials.json

    TIM_URL = 'https://timws.tangent.works/v4/api'  # URL to which the requests are sent

    credentials = tim_client.Credentials(credentials_json['license_key'], credentials_json['email'],
                                         credentials_json['password'], tim_url=TIM_URL)
    api_client = tim_client.ApiClient(credentials)

    # The configuration settings for TIM to work - these are the minimal settings needed as in most cases TIM can
    # handle the other settings automatically - I have found that there is no advantage in using the other settings
    configuration_backtest = {
        "usage": {
            "predictionTo": {
                "baseUnit": "Sample",
                "offset": md_h
            },
            "backtestLength": backnum,
            "modelQuality": [{'day': i, 'quality': 'High'} for i in range(0, 6)]
        },
        "maxModelComplexity": 50,
        "predictionIntervals": {
            "confidenceLevel": 50
        },
        "extendedOutputConfiguration": {
            "returnSimpleImportances": True,
            "returnExtendedImportances": True,
            'returnAggregatedPredictions': True
        },
    }

    # Passes the dataframe and configuration to TIM and returns a model
    model = api_client.prediction_build_model_predict(df, configuration_backtest)

    # If in the function variables the indicator is set to False, then return a dataframe, else return a TIM model -
    # this was originally created to be used by the process_missing_values function below but is now used in
    # production to save the predictions as a dataframe which simplifies communication between the containers
    if not indicator:
        df = model.prediction.reset_index()
        df.columns = ['timestamp', 'PV_obs']
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        return df
    else:
        return model


def clean_solcast_data(df):
    # This function uses Pandas to transform the data from Solcast format to the format needed in the rest of the app
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

# The following functions were used to originally iteratively fill in the missing values in the raw data from Keele -
# however, as this process took over ten minutes to run once I had a configuration that worked I saved the results as
# a CSV file (Keele_Historical_Clean.csv) and now just load that in at the start - these might be needed if a new
# site with missing data was to be set up, hence I have left them here commented out

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
#
#
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
