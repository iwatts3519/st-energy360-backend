import dash
import dash_bootstrap_components as dbc

# metatags needed for mobile responsive

external_scripts = ["https://code.jquery.com/jquery-1.12.4.min.js"]
app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=[dbc.themes.MINTY],
                suppress_callback_exceptions=True)
server = app.server
