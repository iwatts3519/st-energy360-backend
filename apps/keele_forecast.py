import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from app import app
import helper_functions as hf
from keele_data import complete_df as keele_df
import pandas as pd
import numpy as np

# keele_df = pd.read_csv('./Data/Complete_Df.csv')
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)
keele_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)
# print('***** Keele Data Frame *****')
# print(keele_df)
# keele_df.to_csv('./Data/final.csv', index=False)
backtest = hf.build_model(keele_df, True)

# print('*** Backtest ***')
# print(f'1. Backtest has type {type(backtest)}')
# print(backtest)
# print(f'{type(backtest.aggregated_predictions)}')
# for i, item in enumerate(backtest.aggregated_predictions):
#     print(f'item {i} is {item}')

x, y = hf.get_missing_index(keele_df)
# -------------------
layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col([
                html.Br()

            ])
        ]),
    dbc.Row(
        [
            dbc.Col([
                html.H1('Keele Home Farm Solar Production 7 day Forecast')
            ], width={'size': 12, 'offset': 0})
        ]),
    # dbc.Row([
    #     dbc.Col([
    #         html.H1('Keele Home Farm Solar Production 7 day Forecast')
    #     ], style='textAlign: center', width={'size': 12})
    # ]),

    dbc.Row(
        [
            dbc.Col([
                dcc.Graph(id='Figure_Keele'),
                dcc.Graph(id="Importance")
            ], width={'size': 4, 'offset': 1}
            ),
            dbc.Col([
                html.H2(['Accuracy Scores']),
                dcc.Graph(id='Gauge')
            ], width={'size': 4, 'offset': 1})
        ]),
    dbc.Row([
        dbc.Col(
            dcc.RangeSlider(
                id='forecast-slider',
                min=0,
                max=y,
                step=1,
                value=[0, 24],
                marks={
                    0: '0H',
                    24: '24H',
                    48: '48H',
                    72: '72H',
                    96: '96H',
                    120: '120H',
                    144: '144H',
                    168: '168H'
                }),
            width={'size': 10, 'offset': 1}
        )

    ]),
    dbc.Row([
        html.Br()
    ])

], fluid=True)


# -------------------------------------------------------------------------------------
@app.callback(
    [Output(component_id="Figure_Keele", component_property="figure"),
     Output(component_id="Gauge", component_property="figure"),
     Output(component_id="Importance", component_property="figure"),
     Output(component_id="mae", component_property="children"),
     Output(component_id="mse", component_property="children"),
     Output(component_id="mape", component_property="children"),
     Output(component_id="rmse", component_property="children")],
    [Input('forecast-slider', 'value')]
)
def update_graph(value):
    dff = backtest.prediction.reset_index()[value[0]:value[1]]
    dff2 = backtest.prediction_intervals_upper_values.reset_index()[value[0]:value[1]]
    dff3 = backtest.prediction_intervals_lower_values.reset_index()[value[0]:value[1]]
    print(dff)
    print(dff2)
    print(dff3)
    simple_importances = backtest.predictors_importances['simpleImportances']  # get predictor importances
    simple_importances = sorted(simple_importances, key=lambda i: i['importance'], reverse=True)  # sort by importance

    si_df = pd.DataFrame(index=np.arange(len(simple_importances)), columns=['predictor name',
                                                                            'predictor importance (%)'])  # initialize predictor importances dataframe

    for (i, si) in enumerate(simple_importances):
        si_df.loc[i, 'predictor name'] = si['predictorName']  # get predictor name
        si_df.loc[i, 'predictor importance (%)'] = si['importance']  # get importance of the predictor

    fig = go.Figure()
    correct_metrics = ((value[1] - value[0]) // 24) + 8

    fig.add_trace(go.Scatter(x=dff.Timestamp,
                             y=dff.Prediction,
                             name="production forecast",
                             line=dict(color='purple'),
                             showlegend=True))  # plotting production prediction

    fig.add_trace(go.Scatter(x=dff2.Timestamp,
                             y=dff2.UpperValues,
                             name='upper',
                             line=dict(color='green'),
                             showlegend=True))
    fig.add_trace(go.Scatter(x=dff3.Timestamp,
                             y=dff3.LowerValues,
                             name='lower',
                             line=dict(color='red'),
                             showlegend=True))  # plotting confidence intervals

    fig.update_layout(title_text=f"Keele - Home Farm Forecast")
    mae = f'MAE: {round(backtest.aggregated_predictions[correct_metrics]["accuracyMetrics"]["MAE"], 4)}'
    mse = f'MSE: {round(backtest.aggregated_predictions[correct_metrics]["accuracyMetrics"]["MSE"], 4)}'
    mape = f'MAPE: {round(backtest.aggregated_predictions[correct_metrics]["accuracyMetrics"]["MAPE"], 2)}%'
    rmse = f'RMSE: {round(backtest.aggregated_predictions[correct_metrics]["accuracyMetrics"]["RMSE"], 4)}'
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(backtest.aggregated_predictions[correct_metrics]["accuracyMetrics"]["MAPE"], 2),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "MAPE"}))
    fig3 = go.Figure(go.Bar(x=si_df['predictor name'], y=si_df['predictor importance (%)']))  # plot the bar chart
    fig3.update_layout(height=400,  # update size, title and axis titles of the chart
                       width=600,
                       title_text="Importances of predictors",
                       xaxis_title="Predictor name",
                       yaxis_title="Predictor importance (%)")
    return fig, fig2, fig3, mae, mse, mape, rmse
