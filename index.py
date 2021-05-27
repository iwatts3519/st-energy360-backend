import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_auth

# Connect to main app.py file
from app import app
from app import server

auth = dash_auth.BasicAuth(app, {'iwatts': 'secret'})

from apps import welcome, solar_forecast

app.layout = html.Div([

    dcc.Location(id="url", refresh=False, pathname="/apps/welcome"),

    dbc.Row([
        dbc.Col([
            dbc.NavbarSimple([
                dbc.NavItem([
                    dbc.Button(dbc.NavLink("Solar Forecast ", href='/apps/solar_forecast'), className="lg mx-2",
                               color="primary")
                ])
            ],
                brand="RAFA Analytics",
                brand_href="welcome.layout",
                fluid=True,
                dark=True,
                color="primary")
        ], width=12)
    ]),


    html.Div(id='page-content', children=[]),
    dbc.Row(
        dbc.Col(
            html.Div("(c) CAD Group 6 - Keele University -  Built by Dash on Flask",
                     style={"text-align": "center"}))
    )
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print("Function Running")
    if pathname == '/apps/welcome':
        print('welcome')
        return welcome.layout
    if pathname == '/apps/solar_forecast':
        print('forecast')
        return solar_forecast.layout
    else:
        return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=False)
