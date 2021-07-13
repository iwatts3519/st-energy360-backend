import tim_client
from time import time
import logging
import pandas as pd
import plotly as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
from copy import copy
import plotly.io as pio

pio.renderers.default = "browser"


def authenticate():
    print("Starting Function")
    start = time()
    with open('credentials.json') as f:
        credentials_json = json.load(f)  # loading the credentials from credentials.json

    TIM_URL = 'https://timws.tangent.works/v4/api'  # URL to which the requests are sent

    SAVE_JSON = True  # if True - JSON requests and responses are saved to JSON_SAVING_FOLDER
    JSON_SAVING_FOLDER = './keele_logs/'  # folder where the requests and responses are stored

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
    end = time()
    print(f'It took {end - start} seconds to authenticate')
    return api_client


def build_model(df, api_client):
    configuration = {
        "usage": {
            "usageTime": [
                {"type": "Day", "value": "*"},
                {"type": "Hour", "value": "7"},
                {"type": "Minute", "value": "00"}
            ],
            "predictionFrom": {"baseUnit": "Sample", "offset": 1},
            "predictionTo": {"baseUnit": "Day", "offset": 7},
        },
        "modelQuality": [{'day': i, 'quality': 'Automatic'} for i in range(0, 8)],
        # "dataNormalization": True,
        # "features": [
        #     "Polynomial",
        #     "TimeOffsets",
        #     "PiecewiseLinear",
        #     "Intercept",
        #     "PeriodicComponents",
        #     "DayOfWeek",
        #     "MovingAverage"
        # ],
        # "maxModelComplexity": 100,
        # "timeSpecificModels": True,
        # "allowOffsets": False,
        # "memoryPreprocessing": False,
        # "interpolation": {
        #     "type": "Linear",
        #     "maxLength": 1
        # },
        "extendedOutputConfiguration": {
            "returnPrediction": True,
            "returnAggregatedPredictions": True,
            "returnRawPredictions": True,
            "returnSimpleImportances": True,
            "returnExtendedImportances": True,
            # "predictionBoundaries": {
            #     "type": "None"
            # }
        }
    }

    update = {
        "uniqueName": "",
        "updateTime": [
            {"type": "Day", "value": "*"},
            {"type": "Hour", "value": "7"},
            {"type": "Minute", "value": "5"}
        ],
        "updateUntil": {"baseUnit": "Day", "offset": 2}
    }

    data_updates = [
        {
            "uniqueName": target_variable,
            "updateTime": [
                {"type": "Day", "value": "*"},
                {"type": "Hour", "value": "7"},
                {"type": "Minute", "value": "5"}
            ],
            "updateUntil": {"baseUnit": "Sample", "offset": 0}
        }
    ]

    for name in predictor_candidates:
        u = update.copy()
        u["uniqueName"] = name
        data_updates.append(u)

    start = time()

    model = api_client.prediction_build_model(data_train, configuration, predictors_update=data_updates)
    end = time()
    print(f'It took {end - start} seconds to build model')
    print(model.status)

    fig1 = go.Figure(go.Scatter(x=data.loc[:, timestamp], y=data.loc[:, target_variable], name=target_variable,
                                line=dict(color='black')))

    for i, ap in enumerate(model.aggregated_predictions):
        fig1.add_trace(go.Scatter(x=ap['values'].index, y=ap['values'].loc[:, 'Prediction'],
                                  name="In-sample prediction at: " + ap['predictionTime'] + ", for: day+" + str(
                                      ap['day'])))

    fig1.update_layout(height=500, width=1000, title_text="In-sample prediction")

    fig1.show()

    si_df = pd.DataFrame(model.predictors_importances['simpleImportances'])
    si_df.sort_values(by=['importance'], ascending=False)
    ei_df = pd.DataFrame(model.predictors_importances['extendedImportances'])
    ei_df = ei_df.sort_values(by=['time', 'importance'], ascending=False)

    fig2 = go.Figure(go.Bar(x=si_df['predictorName'], y=si_df['importance']))
    fig2.update_layout(height=400,
                       width=600,
                       title_text="Importances of predictors",
                       xaxis_title="Predictor name",
                       yaxis_title="Predictor importance (%)")
    fig2.show()
    return model


def make_prediction(df, model, indicator, api_client):
    prediction_configuration = {
        "predictionIntervals": {
            "confidenceLevel": 90  # confidence level of the prediction intervals (in %)
        },
        'extendedOutputConfiguration': {
            'returnExtendedImportances': True,
            'returnAggregatedPredictions': True

            # flag that specifies if the importances of features are returned in the response
        }
    }
    start = time()

    backtest = api_client.prediction_predict(df, model, prediction_configuration)

    end = time()

    print(f'It took {end - start} seconds to run keele forecast')

    if not indicator:
        return df
    else:
        return backtest

    # def build_model(df, indicator):
    #     backnum = int(len(df) * .4)
    #     build_configuration = {
    #         'usage': {
    #             'predictionTo': {
    #                 'baseUnit': 'Sample',
    #                 # units that are used for specifying the prediction horizon length (one of 'Day', 'Hour', 'QuarterHour',
    #                 # 'Sample')
    #                 'offset': 168  # number of units we want to predict into the future (24 hours in this case)
    #             },
    #             'backtestLength': backnum
    #             # number of samples that are used for backtesting (note that these samples are excluded from model building
    #             # period)
    #         },
    #         # "i_Linear": {
    #         #     "max_length": 250
    #         # },
    #         "predictionIntervals": {
    #             "confidenceLevel": 90  # confidence level of the prediction intervals (in %)
    #         },
    #         'extendedOutputConfiguration': {
    #             'returnExtendedImportances': True
    #             # flag that specifies if the importances of features are returned in the response
    #         }
    #     }

    start = time()
    model = api_client.prediction_build_model(df, build_configuration)
    end = time()
    print(f'It took {end - start} seconds to build model')
    if not indicator:
        df = model.prediction.reset_index()
        return df
    else:
        return model


api_client = authenticate()
data = pd.read_csv('./Data/final.csv', index_col=0)
nrow, ncol = data.shape

print("Number of rows: ", nrow)
print("Number of columns: ", ncol)
print(data.columns)

cols = data.columns

fig = plt.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=data.loc[:, cols[0]], y=data.loc[:, cols[1]], name=cols[1]), row=1, col=1)

for col in cols[2:ncol]:
    fig.add_trace(go.Scatter(x=data.loc[:, cols[0]], y=data.loc[:, col], name=col), row=2, col=1)

fig.update_layout(height=700, width=1000, title_text="Data visualization")

fig.show()

timestamp = "timestamp"
target_variable = "PV_obs"
predictor_candidates = ['GHI', 'DNI', 'DHI', 'SA', 'SZ', 'CO', 'Temp']

count_row = data.shape[0]
in_sample_end = int(np.floor(count_row * 2 / 3))
data_train = data[:in_sample_end]
data_train

prediction_model = build_model(data_train, api_client)
