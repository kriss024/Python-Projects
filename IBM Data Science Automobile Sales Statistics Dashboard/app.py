import dash
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, no_update

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"

df = pd.read_csv(URL)
print("Data downloaded and read into a dataframe!")

# Other formatting Consts
vehicle_type_names = {
    "Supperminicar": "Super Mini Car",
    "Mediumfamilycar": "Medium Family Car",
    "Smallfamiliycar": "Small Family Car",
    "Sports": "Sports Car",
    "Executivecar": "Executive Car",
}
label_names = {
    "Automobile_Sales": "Automobile Sales",
    "Vehicle_Type": "Vehicle Type",
    "Advertising_Expenditure": "Advertising Expenditure",
    "unemployment_rate": "Unemployment Rate",
}
month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# Initialize the Dash app
app = Dash(__name__)
app.title = "Automobile Sales Statistics Dashboard"

# Build dropdown option lists
year_options = [{"label": str(y), "value": y} for y in sorted(df["Year"].unique())]

vehicle_options = [
    {"label": vehicle_type_names.get(v, v), "value": v}
    for v in sorted(df["Vehicle_Type"].unique())
]

stat_options = [
    {"label": label_names.get(k, k), "value": k}
    for k in ["Automobile_Sales", "Advertising_Expenditure", "unemployment_rate"]
]

# Define the layout
app.layout = html.Div(
    children=[
        html.H1(
            "Welcome to Automobile Sales Statistics Dashboard",
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}
        ),

        # Control panel
        html.Div([
            html.Div([
                html.Label("Select Year", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=year_options,
                    value=year_options[0]["value"],
                    clearable=False,
                    style={"marginBottom": "12px"}
                ),

                html.Label("Select Vehicle Type", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="vehicle-dropdown",
                    options=vehicle_options,
                    value=vehicle_options[0]["value"],
                    clearable=True,
                    placeholder="(All vehicle types if cleared)",
                    style={"marginBottom": "12px"}
                ),

                html.Label("Select Statistic", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="stat-dropdown",
                    options=stat_options,
                    value=stat_options[0]["value"],
                    clearable=False,
                    style={"marginBottom": "12px"}
                ),

                html.Label("Show Recession Report", style={"fontWeight": "bold"}),
                dcc.Checklist(
                    id="recession-checklist",
                    options=[{"label": "Recession only", "value": "recession"}],
                    value=[],
                    inline=True,
                    style={"marginBottom": "12px"}
                ),

            ], style={"width": "330px", "display": "inline-block", "verticalAlign": "top", "paddingRight": "30px"}),

            # Dynamic options panel
            html.Div(id="input-container",
                     style={"display": "inline-block", "verticalAlign": "top", "maxWidth": "520px"}),
        ], style={"width": "90%", "margin": "auto", "marginBottom": 30}),

        # Output panel
        html.Div(id="output-container", children=[
            dcc.Graph(id="main-graph")
        ], style={"width": "90%", "margin": "auto"}),
    ]
)

# Callback 1: dynamic controls based on stat
@app.callback(
    Output("input-container", "children"),
    Input("stat-dropdown", "value"),
)
def update_input_container(selected_stat):
    if selected_stat == "Automobile_Sales":
        return html.Div([
            html.H4("Automobile Sales options", style={"marginTop": 0}),
            html.P("Clear Vehicle Type to view aggregated totals."),
            dcc.Checklist(
                id="show-points-checklist",
                options=[{"label": "Show points", "value": "points"}],
                value=[],
                inline=True
            )
        ])
    elif selected_stat == "Advertising_Expenditure":
        return html.Div([
            html.H4("Advertising Expenditure options", style={"marginTop": 0}),
            html.P("You can enable rolling mean below."),
            dcc.Checklist(
                id="rolling-mean-checklist",
                options=[{"label": "3-month rolling mean", "value": "rm3"}],
                value=[],
                inline=True
            )
        ])
    elif selected_stat == "unemployment_rate":
        return html.Div([
            html.H4("Unemployment Rate", style={"marginTop": 0}),
            html.P("Vehicle Type is not used for this statistic.")
        ])
    return html.Div(["Select a statistic to see options."])

# Callback 2: update graph
@app.callback(
    Output("main-graph", "figure"),
    Input("year-dropdown", "value"),
    Input("vehicle-dropdown", "value"),
    Input("stat-dropdown", "value"),
    Input("recession-checklist", "value"),
)
def update_output(year, vehicle_type, stat, recession_filter):

    # Filter year
    dff = df[df["Year"] == year].copy()

    # Apply recession filter
    if "recession" in recession_filter:
        dff = dff[dff["Recession"] == 1]

    # Logic for statistics
    if stat == "unemployment_rate":
        plot_df = dff.groupby("Month", as_index=False)[stat].mean()
    else:
        if not vehicle_type:
            plot_df = dff.groupby("Month", as_index=False)[stat].sum()
        else:
            plot_df = dff[dff["Vehicle_Type"] == vehicle_type].groupby("Month", as_index=False)[stat].sum()

    plot_df["Month"] = pd.Categorical(plot_df["Month"], categories=month_order, ordered=True)
    plot_df = plot_df.sort_values("Month")

    y_label = label_names.get(stat, stat)

    fig = px.line(
        plot_df,
        x="Month",
        y=stat,
        markers=True,
        title=f"{y_label} — {year}"
              + (f" — {vehicle_type_names.get(vehicle_type, vehicle_type)}" if vehicle_type else " — All vehicle types")
              + (" — Recession Only" if "recession" in recession_filter else ""),
        labels={"Month": "Month", stat: y_label}
    )

    fig.update_layout(transition_duration=300, template="simple_white")
    return fig


if __name__ == "__main__":
    app.run(debug=True)