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
import dash_table

# keele_df = pd.read_csv('./Data/Complete_Df.csv')
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)
keele_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
keele_df['timestamp'] = pd.to_datetime(keele_df['timestamp'], utc=True).dt.tz_localize(None)
pred_model = hf.build_model(keele_df, True, keele_df['PV_obs'].isna().sum())
start = keele_df['PV_obs'].isna().sum() - 168
print(f'start is {start}')
dff_table = pred_model.prediction.reset_index()[start:-1]
print(len(dff_table))
dff_table['Timestamp'] = pd.to_datetime(dff_table['Timestamp'], utc=True).dt.tz_localize(None)
dff_table['Timestamp'] = dff_table['Timestamp'].dt.strftime('%A %d %b %H:%M')
dff_table['Prediction'] = round(dff_table['Prediction'], 2)
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
            ])
        ], className='justify-center'),

    dbc.Row(
        [

            dbc.Col([
                dbc.Card([
                    dcc.Graph(id='Figure_Keele'),
                ]),
            ], width={'size': 5, 'offset': 1},
                className='align-self-start'),
            dbc.Col([
                dbc.Card([
                    dbc.CardDeck([
                        dbc.Card([
                            dcc.Graph(id='Gauge1')

                        ]),
                        dbc.Card([
                            dcc.Graph(id='Gauge2')

                        ]),
                        dbc.Card([
                            dcc.Graph(id='Gauge3')
                        ])
                    ]),
                    html.Hr(),
                    dcc.Graph(id="Importance")
                ])

            ], width={'size': 4, 'offset': 1},
                className='align-self-start')
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
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in dff_table.columns],
                data=dff_table.to_dict('records'),
            )
        ], width={'size': 2, 'offset': 5}
        )

    ])

], fluid=True)


# -------------------------------------------------------------------------------------
@app.callback(
    [Output(component_id="Figure_Keele", component_property="figure"),
     Output(component_id="Gauge1", component_property="figure"),
     Output(component_id="Gauge2", component_property="figure"),
     Output(component_id="Gauge3", component_property="figure"),
     Output(component_id="Importance", component_property="figure")],
    [Input('forecast-slider', 'value')]
)
def update_graph(fs_value):
    dff = pred_model.prediction.reset_index()[fs_value[0] + start:fs_value[1] + start]
    dff2 = pred_model.prediction_intervals_upper_values.reset_index()[fs_value[0] + start:fs_value[1] + start]
    dff3 = pred_model.prediction_intervals_lower_values.reset_index()[fs_value[0] + start:fs_value[1] + start]
    simple_importances = pred_model.predictors_importances['simpleImportances']  # get predictor importances
    simple_importances = sorted(simple_importances, key=lambda i: i['importance'], reverse=True)  # sort by importance
    si_df = pd.DataFrame(index=np.arange(len(simple_importances)), columns=['predictor name',
                                                                            'predictor importance (%)'])  # initialize predictor importances dataframe

    # print(dff3)
    # dff3['LowerValues'] = dff3.LowerValues.clip(lower=0)

    for (i, si) in enumerate(simple_importances):
        si_df.loc[i, 'predictor name'] = si['predictorName']  # get predictor name
        si_df.loc[i, 'predictor importance (%)'] = si['importance']  # get importance of the predictor

    fig = go.Figure()
    correct_metrics = ((fs_value[1] - fs_value[0]) // 24) + 6

    fig.add_trace(go.Scatter(x=dff.Timestamp,
                             y=dff.Prediction,
                             name="production forecast",
                             line=dict(color='purple'),
                             showlegend=True))  # plotting production prediction

    fig.add_trace(go.Scatter(x=dff2.Timestamp,
                             y=dff2.UpperValues,
                             name='upper',
                             marker=dict(color="#444"),
                             line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=dff3.Timestamp,
                             y=dff3.LowerValues,
                             name='lower',
                             fill='tonexty',
                             line=dict(width=0),
                             showlegend=False))  # plotting confidence intervals

    fig.update_layout(height=400, showlegend=False)

    fig2 = go.Figure(go.Indicator(
        mode="number",
        value=round(pred_model.aggregated_predictions[correct_metrics]["accuracyMetrics"]["MAE"], 2),
        title={'text': "MAE"}))
    fig2.update_layout(height=150)
    fig3 = go.Figure(go.Bar(x=si_df['predictor name'], y=si_df['predictor importance (%)']))  # plot the bar chart
    fig3.update_layout(title_text="Importances of predictors",
                       xaxis_title="Predictor name",
                       yaxis_title="Predictor importance (%)",
                       height=200)
    fig4 = go.Figure(go.Indicator(
        mode="number",
        value=round(pred_model.aggregated_predictions[correct_metrics]["accuracyMetrics"]["MSE"], 2),
        title={'text': "MSE"}))
    fig4.update_layout(height=150)
    fig5 = go.Figure(go.Indicator(
        mode="number",
        value=round(pred_model.aggregated_predictions[correct_metrics]["accuracyMetrics"]["RMSE"], 2),
        title={'text': "RMSE"}))
    fig5.update_layout(height=150)

    return fig, fig2, fig4, fig5, fig3
