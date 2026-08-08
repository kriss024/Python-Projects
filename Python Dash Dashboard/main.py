from datetime import date
import pandas as pd
from dash import Dash, html, dcc, Output, Input, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px


# stylesheet with the .dbc class from dash-bootstrap-templates library
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

hello_h1 = html.H1(
    children='Hello Dash',
    className="pb-4"
)

dash_label = html.Div(
    children='Dash: A web application framework for your data.',
    className="pb-4"
)

hello_div = html.Div(
    [
        hello_h1,
        dash_label
    ],
    className="pb-3 bg-light rounded"
)

dropdown = html.Div(
    [
        dbc.Label("Dropdown with Bootstrap theme"),
        dcc.Dropdown(["Apple", "Carrots", "Chips", "Cookies"], "Cookies", id='my-input')
    ],
    className="dbc py-3"
)

datepicker_single = html.Div(
    dcc.DatePickerSingle(date=date(2022, 8, 5), className="py-3", id='date-picker')
)

output_div = html.Div(
    className="py-3 rounded", id='my-output'
)

app.layout = dbc.Container(
    [hello_div, dropdown, datepicker_single, output_div],
    className="p-4"
)

@callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
    Input(component_id='date-picker', component_property='date')
)

def update_output_div(dropdown_value, selected_date):
    return f'Dropdown: {dropdown_value}, Date: {selected_date}'

if __name__ == '__main__':
    app.run(debug=True)
    