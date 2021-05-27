import dash_bootstrap_components as dbc
import logging
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from app import app
import plotly.io as pio
import numpy as np
import json

import tim_client

# --------------------------------------------------------
with open('credentials.json') as f:
    credentials_json = json.load(f)  # loading the credentials from credentials.json

TIM_URL = 'https://timws.tangent.works/v4/api'  # URL to which the requests are sent

credentials = tim_client.Credentials(credentials_json['license_key'], credentials_json['email'],
                                     credentials_json['password'], tim_url=TIM_URL)
api_client = tim_client.ApiClient(credentials)

# --------------------------------------------------------
configuration_backtest = {
    'usage': {
        'predictionTo': {
            'baseUnit': 'Sample',
            # units that are used for specifying the prediction horizon length (one of 'Day', 'Hour', 'QuarterHour',
            # 'Sample')
            'offset': 38  # number of units we want to predict into the future (24 hours in this case)
        },
        'backtestLength': 6946
        # number of samples that are used for backtesting (note that these samples are excluded from model building
        # period)
    },
    "predictionIntervals": {
        "confidenceLevel": 90  # confidence level of the prediction intervals (in %)
    },
    'extendedOutputConfiguration': {
        'returnExtendedImportances': True
        # flag that specifies if the importances of features are returned in the response
    }
}

# --------------------------------------------------------
data = tim_client.load_dataset_from_csv_file('data.csv', sep=',')  # loading data from data.csv

# --------------------------------------------------------

backtest = api_client.prediction_build_model_predict(data,
                                                     configuration_backtest)  # running the RTInstantML forecasting
# using data and defined configuration
print(f'Backtest has type {type(backtest)}')
# -------------------------------------------------------------------------------------
layout = html.Div([
    dbc.Row(
        [
            dbc.Col(
                html.Label("Please Choose an asset to analyse"),
                width={"size": 6}
            )
        ]),
    dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id='group_dropdown',
                    options=[{'label': 'One', 'value': 1}],
                    value=[1],
                    multi=True,
                    clearable=False,
                    style={"color": '#222222'}),
                width=6
            )
        ]),
    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id='Figure_4'),
                width=6
            )
        ]),

])


# -------------------------------------------------------------------------------------
@app.callback(
    Output(component_id="Figure_4", component_property="figure"),
    [Input(component_id="group_dropdown", component_property="value")]
)
def update_graph(gp_dropdown):
    fig = plt.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)  # plot initialization

    fig.add_trace(go.Scatter(x=data.loc[:, "timestamp"], y=data.loc[:, "PV_obs"],
                             name="target", line=dict(color='black')), row=1, col=1)  # plotting the target variable

    fig.add_trace(go.Scatter(x=backtest.prediction.index,
                             y=backtest.prediction.loc[:, 'Prediction'],
                             name="production forecast",
                             line=dict(color='purple')), row=1, col=1)  # plotting production prediction

    fig.add_trace(go.Scatter(x=backtest.prediction_intervals_upper_values.index,
                             y=backtest.prediction_intervals_upper_values.loc[:, 'UpperValues'],
                             marker=dict(color="#444"),
                             line=dict(width=0),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=backtest.prediction_intervals_lower_values.index,
                             y=backtest.prediction_intervals_lower_values.loc[:, 'LowerValues'],
                             fill='tonexty',
                             line=dict(width=0),
                             showlegend=False), row=1, col=1)  # plotting confidence intervals

    fig.add_trace(go.Scatter(x=backtest.aggregated_predictions[1]['values'].index,
                             y=backtest.aggregated_predictions[1]['values'].loc[:, 'Prediction'],
                             name="in-sample MAE: " + str(
                                 round(backtest.aggregated_predictions[1]['accuracyMetrics']['MAE'], 2)),
                             line=dict(color='goldenrod')), row=1, col=1)  # plotting in-sample prediction

    fig.add_trace(go.Scatter(x=backtest.aggregated_predictions[3]['values'].index,
                             y=backtest.aggregated_predictions[3]['values'].loc[:, 'Prediction'],
                             name="out-of-sample MAE: " + str(
                                 round(backtest.aggregated_predictions[3]['accuracyMetrics']['MAE'], 2)),
                             line=dict(color='red')), row=1, col=1)  # plotting out-of-sample-sample prediction

    fig.add_trace(go.Scatter(x=data.loc[:, "timestamp"], y=data.loc[:, "GHI"],
                             name="Temperature", line=dict(color='forestgreen')), row=2,
                  col=1)  # plotting the predictor Temperature

    fig.update_layout(height=600, width=1000,
                      title_text="Backtesting, modelling difficulty: "
                                 + str(round(backtest.data_difficulty, 2)) + "%")  # update size and title of the plot

    return fig
