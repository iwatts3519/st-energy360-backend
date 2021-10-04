from azure.storage.blob import BlobClient
import pandas as pd
import helper_functions as hf
import json
from datetime import datetime

# This line reads in code that contains historical data from DEOPS that has already iteratively had missing values
# filled using TIM. This is the most probable cause to date of the inaccuracies of the final model
keele_df = pd.read_csv('./Data/Keele_Historical_Clean.csv', parse_dates=['timestamp'])

# Accesses the last n days of DEOPS data - specifically PV_obs (production) data and timestamps and puts into a
# dataframe - function explained in helper functions script
deops_df = hf.get_deops(14)

# Gets a 7 day weather forecast from the Solcast API, add it to historical forecasts, cleans it, and returns it into
# a data frame - function explained in helper functions script
forecast_df = hf.get_solcast_forecast()

# Merge the latest DEOPS data into the latest solcast forecast data to create a data frame that conatins actuals and
# weather forecast data for the last n days - note that at this point the actual future weather data isn't used but
# the dataframe is reused to add a forecast later in this pipeline (the merge/-data_frames function only merges on
# the first dataframe in the list which only has data up until the last data from DEOPS)
current_df = hf.merge_data_frames(deops_df, forecast_df)

# the original keele_df (from Sep 2020) and current_df (explained above) dataframes are now concatenated into a full
# data frame, duplicates are dropped as there will be some overlap, and the dataframe is written back to the csv file
# at the start of this script (Keele_Historical_Clean.csv) which means we are always starting with an up to date
# dataset
keele_df = pd.concat([keele_df, current_df], ignore_index=True)
keele_df.drop_duplicates(subset='timestamp', keep='first', inplace=True, ignore_index=True)
keele_df.to_csv('./Data/Keele_Historical_Clean.csv', index=False)

# The forecast data is concatenated onto the historical data to create a full model ready to send to TIM
complete_df = pd.concat([keele_df, forecast_df], ignore_index=True)
# complete_df['timestamp'] = pd.to_datetime(complete_df['timestamp'], utc=True).dt.tz_localize(None)
complete_df.drop_duplicates(subset='timestamp', keep='first', inplace=True, ignore_index=True)
complete_df['PV_obs'] = round(complete_df['PV_obs'], 2)
complete_df.sort_values(by=['timestamp'], inplace=True)
# complete_df.drop_duplicates(subset='timestamp', keep='first', inplace=True, ignore_index=True)
# complete_df_dd.to_csv('./Data/forTimSTudio.csv', index=False)

# The complete dataframe is sent to the build_model function which will return a dataframe - the function tackles the
# arguments of dataframe, whether to return a model or a dataframe (False=dataframe), and the forecast horizon which
# in this case is calculated by counting the number of NaN data types in PV_obs
model1 = hf.build_model(complete_df, False, complete_df['PV_obs'].isna().sum())

# This block of code ensures that we always have a full picture of historical predictions, which we can use to
# compare with actuals to check accuracy and later will use to create an accuracy dataframe
hp_df = pd.read_csv('./Data/historical_predictions.csv', parse_dates=['timestamp'], dayfirst=True)
hp_df['timestamp'] = hp_df['timestamp'].astype('datetime64[ns]')

model1 = pd.concat([model1, hp_df], ignore_index=True)
model1.drop_duplicates(subset='timestamp', inplace=True, keep='first', ignore_index=True)
model1.sort_values(by='timestamp', inplace=True, axis=0)
model1.reset_index(inplace=True, drop=True)
model1.to_csv('./Data/historical_predictions.csv', sep=',', index=False)

# This saves the last 168 predictions into a csv file (1 weeks worth) and is ready to be uploaded to Microsoft Azure
# Blob Storage. It gets the index of the time as of the time of running and then uses that to find the index of the
# matching date-time in model 1, finally getting exactly a weeks worth into model2
now = datetime.now().strftime("%Y-%m-%d %H:00:00")
now = pd.to_datetime(now)
last_num = model1[model1['timestamp'] == now].index.tolist()[0]
model2 = model1[last_num:last_num + 168]
model2.to_csv('./Data/new_predictions.csv', index=False)

# The accuracy dataframe is created using the predictions from the model and the original actuals from DEOPS and
# stores in a CSV file ready to be uploaded to Microsoft Azure Blob Storage
acc_df = deops_df.merge(model1, how='left', on='timestamp')
acc_df.columns = ['timestamp', 'Actual', 'Prediction']
acc_df.dropna(inplace=True)
acc_df = acc_df.tail(72)
acc_df.to_csv('./Data/accuracy_frame.csv', index=False)

# Finally we can upload the files to the Azure Blob Storage Account, which will be read by the frontend of the app
# and used to create the dashboard.

# First we open the config.json file so that secure credentials can be accessed without being present in the code
with open('config.json') as f:
    credentials_json = json.load(f)

# This uploads the new_predictions.csv file created above
blob1 = BlobClient.from_connection_string(conn_str=credentials_json["CONNECTIONSTRING"],
                                          container_name=credentials_json["CONTAINERNAME"],
                                          blob_name=credentials_json["BLOBNAME"])
with open(credentials_json["LOCALFILENAME"], 'rb') as my_blob1:
    blob1.upload_blob(my_blob1, overwrite=True)
# This uploads the accuracy_frame.csv file created above
blob2 = BlobClient.from_connection_string(conn_str=credentials_json["CONNECTIONSTRING"],
                                          container_name=credentials_json["CONTAINERNAME"],
                                          blob_name=credentials_json["BLOBNAMEACC"])
with open(credentials_json["LOCALFILENAMEACC"], 'rb') as my_blob2:
    blob2.upload_blob(my_blob2, overwrite=True)
