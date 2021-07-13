import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# metatags needed for mobile responsive

external_scripts = ["https://code.jquery.com/jquery-1.12.4.min.js"]
app = dash.Dash(__name__,
                meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_scripts=external_scripts,
                external_stylesheets=[dbc.themes.MINTY],
                suppress_callback_exceptions=True)
load_figure_template('minty')
server = app.server
