import pandas as pd
import helper_functions as hf

keele_df = pd.read_csv('./Data/Keele_data.csv',
                       sep=';',
                       engine='python',
                       header=0,
                       usecols=[0, 1, 20],
                       names=['Date', 'Time', 'PV_obs'],
                       parse_dates={'timestamp': ['Date', 'Time']},
                       )
keele_df['timestamp'] = pd.date_range(start=keele_df['timestamp'][0], periods=len(keele_df), freq='H')
keele_df_ah = hf.add_solcast_historical(keele_df, False)
keele_df_ah['timestamp'] = pd.to_datetime(keele_df_ah['timestamp'], utc=True).dt.tz_localize(None)
keele_df_pmv = keele_df_ah
while True:
    keele_df_pmv, indicator = hf.process_missing_values(keele_df_pmv)
    if indicator:
        break

forecast_df = hf.get_solcast_forecast()
hf.generate_full_historical()
complete_df = pd.concat([keele_df_pmv, forecast_df], ignore_index=True)
complete_df.drop_duplicates(subset='timestamp', keep='first', inplace=True, ignore_index=True)
complete_df['PV_obs'] = round(complete_df['PV_obs'], 1)

complete_df_dd = complete_df
complete_df_dd.drop_duplicates(subset='timestamp', keep='last', inplace=True, ignore_index=True)

complete_df_dd.to_csv('./Data/Complete_Df.csv', index=False)
