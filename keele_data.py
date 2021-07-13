import pandas as pd
import helper_functions as hf

keele_df = pd.read_csv('Data/Keele_data.csv',
                       sep=';',
                       engine='python',
                       header=0,
                       usecols=[0, 1, 20],
                       names=['Date', 'Time', 'PV_obs'],
                       parse_dates={'timestamp': ['Date', 'Time']},
                       )
keele_df = hf.add_solcast_historical(keele_df)
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)

while True:
    keele_df, indicator = hf.process_missing_values(keele_df, len(keele_df))
    if indicator:
        break
forecast_df = hf.get_solcast_forecast()

complete_df = pd.concat([keele_df, forecast_df], ignore_index=True)
complete_df.to_csv('./Data/Complete_Df.csv', index=False)
