import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_auth

# Connect to main app.py file
from app import app
from app import server

from apps import welcome, keele_forecast

auth = dash_auth.BasicAuth(app, {'iwatts': 'secret'})

app.layout = html.Div([

    dbc.Row([
        dbc.Col([
            dbc.NavbarSimple([

                dbc.NavItem([
                    dbc.Button(dcc.Link("Keele Forecast ", href='/apps/keele_forecast', style={'color': 'white'}),
                               className="lg mx-2",
                               color="primary")
                ])
            ],
                brand="Rafa Analytics",
                brand_href="/apps/welcome",
                fluid=True,
                dark=True,
                color="primary")
        ], width=12)
    ]),

    dcc.Location(id="url", refresh=False, pathname="/apps/welcome"),
    html.Div(id='page-content', children=[]),
    dbc.Row(
        dbc.Col(
            html.Div("(c) 2021 Reliable Insights & Keele University SEND -  Built by Dash on Flask",
                     style={"text-align": "center"}), className='footer')
    )
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/welcome':
        return welcome.layout
    if pathname == '/apps/keele_forecast':
        return keele_forecast.layout
    else:
        return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=False)
