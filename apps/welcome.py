import dash_bootstrap_components as dbc
import dash_html_components as html
from app import app

# -------------------------------------------------------------------------------------
layout = html.Div([
    dbc.Row(
        [
            dbc.Col([
                html.Br()

            ])
        ]),
    dbc.Row(
        [
            dbc.Col([
                html.H1("Customer Profile")
            ], width={'size': 6, 'offset': 1})
        ]),
    dbc.Row(
        [
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="https://via.placeholder.com/300x200.png?text=Profile+Image", top=True),
                    dbc.CardBody([
                        html.H2("Customer Name", className="card-title"),
                        html.P(
                            "Dexter, secundus nomens vix talem de salvus, peritus decor. Nunquam experientia habena.",
                            className='card-text')
                    ])
                ])
            ], width={'size': 2, 'offset': 1}),
            dbc.Col([
                html.P(
                    "Tnunquam experientia lumen. a falsis, elogium altus mortem. sunt fraticinidaes pugna germanus, "
                    "lotus deuses. torquis flavum aonides est. cum buxum studere, omnes liberies amor superbus, teres "
                    "axonaes. Pol, a bene index. Cum danista studere, omnes lamiaes desiderium magnum, varius musaes. "
                    "elogiums sunt rumors de clemens cedrium. altus, placidus scutums patienter amor de fidelis, lotus "
                    "danista. hafnia."),
                html.P(
                    "Tnunquam experientia lumen. a falsis, elogium altus mortem. sunt fraticinidaes pugna germanus, "
                    "lotus deuses. torquis flavum aonides est. cum buxum studere, omnes liberies amor superbus, teres "
                    "axonaes. Pol, a bene index. Cum danista studere, omnes lamiaes desiderium magnum, varius musaes. "
                    "elogiums sunt rumors de clemens cedrium. altus, placidus scutums patienter amor de fidelis, lotus "
                    "danista. hafnia."),
                html.P(
                    "Tnunquam experientia lumen. a falsis, elogium altus mortem. sunt fraticinidaes pugna germanus, "
                    "lotus deuses. torquis flavum aonides est. cum buxum studere, omnes liberies amor superbus, teres "
                    "axonaes. Pol, a bene index. Cum danista studere, omnes lamiaes desiderium magnum, varius musaes. "
                    "elogiums sunt rumors de clemens cedrium. altus, placidus scutums patienter amor de fidelis, lotus "
                    "danista. hafnia.")

            ], width={'size': 8})

        ]),
    dbc.Row(
        [
            dbc.Col([
                html.Br()

            ])
        ]),
    dbc.Row(
        [
            dbc.Col([
                html.H2("Assets")
            ], width={'size': 6, 'offset': 1})
        ]),

    dbc.Row(
        [
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="https://via.placeholder.com/150x100.png?text=Asset+Image",
                                top=True),
                    dbc.CardBody([
                        html.H6("Asset Name", className="card-title"),

                    ])
                ])
            ], width={'size': 1, 'offset': 1}),
            dbc.Col([
                html.P(
                    "Tnunquam experientia lumen. a falsis, elogium altus mortem. sunt fraticinidaes pugna germanus, "
                    "lotus deuses. torquis flavum aonides est. cum buxum studere, omnes liberies amor superbus, teres "
                    "axonaes. Pol, a bene index. Cum danista studere, omnes lamiaes desiderium magnum, varius musaes. "
                    "elogiums sunt rumors de clemens cedrium. altus, placidus scutums patienter amor de fidelis, lotus "
                    "danista. hafnia.")

            ], width={'size': 9})

        ]),
    dbc.Row(
        [
            dbc.Col([
                html.Br()

            ])
        ]),
    dbc.Row(
        [
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="https://via.placeholder.com/150x100.png?text=Asset+Image",
                                top=True),
                    dbc.CardBody([
                        html.H6("Asset Name", className="card-title"),

                    ])
                ])
            ], width={'size': 1, 'offset': 1}),
            dbc.Col([
                html.P(
                    "Tnunquam experientia lumen. a falsis, elogium altus mortem. sunt fraticinidaes pugna germanus, "
                    "lotus deuses. torquis flavum aonides est. cum buxum studere, omnes liberies amor superbus, teres "
                    "axonaes. Pol, a bene index. Cum danista studere, omnes lamiaes desiderium magnum, varius musaes. "
                    "elogiums sunt rumors de clemens cedrium. altus, placidus scutums patienter amor de fidelis, lotus "
                    "danista. hafnia.")

            ], width={'size': 9})

        ])

])
