import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from app import app
import helper_functions as hf
# from keele_data import complete_df as keele_df
import pandas as pd
import numpy as np

keele_df = pd.read_csv('./Data/Complete_Df.csv')
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)
keele_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)

backtest = hf.build_model(keele_df, True, 168)

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

    dbc.Row(
        [
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Configuration Options"),
                    dbc.CardBody([

                    ]),
                    dbc.CardFooter("Choose Options for live update")

                ])
            ], width={'size': 2},
                className='align-self-start'
            ),

            dbc.Col([
                dcc.Graph(id='Figure_Keele'),
            ], width={'size': 4},
                className='align-self-start'
            ),
            dbc.Col([
                dcc.Graph(id='Gauge'),

                dcc.Graph(id="Importance")
            ], width={'size': 4, 'offset': 1},
                className='align-self-start'
            ),
        ]),
    dbc.Row(
        [
            dbc.Col([
                html.Br()

            ])
        ]),
    dbc.Row([
        dbc.Col(
            dcc.RangeSlider(
                id='forecast-slider',
                min=0,
                max=168,
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
     Output(component_id="Importance", component_property="figure")],
    [Input('forecast-slider', 'value')]
)
def update_graph(value):
    dff = backtest.prediction.reset_index()[value[0]:value[1]]
    dff2 = backtest.prediction_intervals_upper_values.reset_index()[value[0]:value[1]]
    dff3 = backtest.prediction_intervals_lower_values.reset_index()[value[0]:value[1]]
    simple_importances = backtest.predictors_importances['simpleImportances']  # get predictor importances
    simple_importances = sorted(simple_importances, key=lambda i: i['importance'], reverse=True)  # sort by importance
    si_df = pd.DataFrame(index=np.arange(len(simple_importances)), columns=['predictor name',
                                                                            'predictor importance (%)'])  # initialize predictor importances dataframe
    # print(dff3)
    # dff3['LowerValues'] = dff3.LowerValues.clip(lower=0)
    for (i, si) in enumerate(simple_importances):
        si_df.loc[i, 'predictor name'] = si['predictorName']  # get predictor name
        si_df.loc[i, 'predictor importance (%)'] = si['importance']  # get importance of the predictor

    fig = go.Figure()
    correct_metrics = ((value[1] - value[0]) // 24) + 6

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

    fig.update_layout(title_text=f"Keele - Home Farm Forecast", height=500)

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(backtest.aggregated_predictions[correct_metrics]["accuracyMetrics"]["RMSE"], 2),
        number={'valueformat': '%'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Chance of Error (RMSE)"}))
    fig2.update_traces(gauge_axis_range=[0, 1])
    fig2.update_layout(height=250)
    fig3 = go.Figure(go.Bar(x=si_df['predictor name'], y=si_df['predictor importance (%)']))  # plot the bar chart
    fig3.update_layout(title_text="Importances of predictors",
                       xaxis_title="Predictor name",
                       yaxis_title="Predictor importance (%)",
                       height=250)
    return fig, fig2, fig3
