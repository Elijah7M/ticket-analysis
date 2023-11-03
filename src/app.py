import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
import dash
from dash import dash_table
import openpyxl
import matplotlib.pyplot as plt
import webbrowser

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import warnings
warnings.filterwarnings('ignore')

# Load the dataset - Replace 'sorted_events' with your actual dataset
df = pd.read_excel('Events.xlsx')

# Custom color mapping for seasons
custom_colors = {
    '2018-2019': 'darkslategrey',
    '2019-2020': 'paleturquoise',
    '2021-2022': 'aquamarine',
    '2022-2023': 'mediumspringgreen'
    # Add more seasons and colors as needed
}

# Create a Dash app
app = dash.Dash(__name__)
server = app.server

# Define the unique seasons for the dropdown options
season_options = [{'label': season, 'value': season} for season in df['Season'].unique()]

# Define the Y-values dropdown options
y_value_options = [
    {'label': 'Primary: Ticket Revenue', 'value': 'Primary: Ticket Revenue'},
    {'label': 'Primary: Tickets Sold', 'value': 'Primary: Tickets Sold'},
]

# Define the opponent filter options
opponent_options = [
    {'label': 'All Opponents', 'value': 'all_opponents'},
    {'label': 'Division Opponents', 'value': 'division_opponents'},
    {'label': 'Non-Division Opponents', 'value': 'non_division_opponents'},
]

# Define the day of the week filter options
day_of_week_options = [
    {'label': 'All Days of the Week', 'value': 'all_days'},
    {'label': 'Monday', 'value': 'Monday'},
    {'label': 'Tuesday', 'value': 'Tuesday'},
    {'label': 'Wednesday', 'value': 'Wednesday'},
    {'label': 'Thursday', 'value': 'Thursday'},
    {'label': 'Friday', 'value': 'Friday'},
    {'label': 'Saturday', 'value': 'Saturday'},
    {'label': 'Sunday', 'value': 'Sunday'},
]

app.layout = html.Div(style={'backgroundColor': 'transparent'}, children=[
    html.H1("Ticket Revenue and Sales Analysis by Season", style={'color': 'black'}),  # Change text color to black
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="season-dropdown",
                options=season_options,
                value=['all_seasons'],
                multi=True,
                placeholder="Select Season(s)",
                style={'width': '47%', 'display': 'inline-block', 'color': 'black'},  # Change text color to black
            ),
            
            dcc.RadioItems(
                id="ticket-revenue-or-sales",
                options=y_value_options,
                value='Primary: Ticket Revenue',
                style={'display': 'inline-block', 'color': 'black'},  # Change text color to black
            ),
        ]),
    ]),
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="opponent-dropdown",
                options=opponent_options,
                value='all_opponents',
                placeholder="Select Opponents",
                style={'width': '47%', 'display': 'inline-block', 'color': 'black'},  # Change text color to black
            ),
            
            dcc.Dropdown(
                id="day-of-week-dropdown",
                options=day_of_week_options,
                value='all_days',
                placeholder="Select Day of the Week",
                style={'width': '47%', 'display': 'inline-block', 'color': 'black'},  # Change text color to black
            ),
        ]),
    ]),
    
    dcc.Graph(id="seasonal-ticket-revenue"),
    
    html.Div([
        html.H3("Statistics per Event", style={'color': 'black'}),
        dash_table.DataTable(
            id='statistics-table',
            columns=[
                {'name': 'Data', 'id': 'Data'},
                {'name': 'Events', 'id': 'Events'},
                {'name': 'Average Attendance', 'id': 'Average Attendance'},
                {'name': 'Average Ticket Price', 'id': 'Average Ticket Price'},
                {'name': 'Attendance Difference', 'id': 'Attendance Difference'},
                {'name': 'Ticket Price Difference', 'id': 'Ticket Price Difference'}
            ],
            style_table={
                'width': '100%',
                'maxHeight': '300px',  # Set the maximum height of the table
                'overflowY': 'auto',   # Enable vertical scrolling when the content exceeds the maximum height
                'backgroundColor': 'lightgrey'
            },
            style_cell={'textAlign': 'left', 'color': 'black'},
            style_data_conditional=[{
                'backgroundColor': 'grey'
            }],
            style_as_list_view=True,
                )
    ], style={'flex': '1'}),  # Set the
])

@app.callback(
    [dash.dependencies.Output("seasonal-ticket-revenue", "figure"),
     dash.dependencies.Output('statistics-table', 'data')],
    [dash.dependencies.Input("season-dropdown", "value"),
     dash.dependencies.Input("ticket-revenue-or-sales", "value"),
     dash.dependencies.Input("opponent-dropdown", "value"),
     dash.dependencies.Input("day-of-week-dropdown", "value")]
)
def update_graph(selected_season, selected_y_value, selected_opponent, selected_day_of_week):
    if not selected_season or 'all_seasons' in selected_season:
        # If no season selected or 'All Seasons' is selected, show all seasons with custom colors
        season_data = df
    else:
        season_data = df[df["Season"].isin(selected_season)]
    
    if selected_opponent == 'division_opponents':
        opponents = ['3', '30', '31']
        season_data = season_data[season_data['Opponent'].isin(opponents)]
    elif selected_opponent == 'non_division_opponents':
        opponents = ['3', '30', '31']
        season_data = season_data[~season_data['Opponent'].isin(opponents)]
    
    if selected_day_of_week != 'all_days':
        season_data = season_data[season_data['Day of Week'] == selected_day_of_week]
    
    if not selected_season or 'all_seasons' in selected_season:
        # If no season selected or 'All Seasons' is selected, show all seasons with custom colors
        season_avg = season_data.groupby("Season")[selected_y_value].sum().reset_index()
        season_count = season_data.groupby("Season").size().reset_index()
        season_avg = season_avg.merge(season_count, on="Season")
        season_avg = season_avg.rename(columns={0: "Event Count"})
        fig = px.bar(season_avg, x="Season", y=selected_y_value, 
                     color="Season", color_discrete_map=custom_colors,
                     labels={selected_y_value: selected_y_value},
                     title=f"{selected_y_value} for All Seasons",
                     hover_data=["Event Count"])
    else:
        # If specific seasons are selected, show the selected seasons with custom colors
        season_count = season_data.groupby("Season").size().reset_index()
        season_data = season_data.merge(season_count, on="Season")
        season_data = season_data.rename(columns={0: "Event Count"})
        fig = px.scatter(season_data, x="Event Date", y=selected_y_value, hover_data=['Opponent', 'Day of Week', 'Event Time', 'Event Count'],
                         color="Season", color_discrete_map=custom_colors,
                         labels={selected_y_value: selected_y_value},
                         title=f"{selected_y_value} for Selected Seasons")

    # Update axis labels and add hover tooltip
    fig.update_xaxes(title_text="Event Date")
    fig.update_yaxes(title_text=selected_y_value)
    fig.update_layout(plot_bgcolor='darkgray', paper_bgcolor='lightgray')

    # Calculate mean and median for the entire dataset
    overall_mean = round(df[selected_y_value].mean())
    overall_median = round(df[selected_y_value].median())

    # Calculate average attendance and average ticket price for the entire dataset
    overall_avg_attendance = round(df['Primary: Tickets Sold'].mean())
    overall_avg_ticket_price = round(df['Primary: Ticket Revenue'].sum() / df['Primary: Tickets Sold'].sum(), 2)

    # Calculate mean and median for the filtered dataset
    filtered_mean = round(season_data[selected_y_value].mean())
    filtered_median = round(season_data[selected_y_value].median())

    # Calculate average attendance and average ticket price for the filtered dataset
    filtered_avg_attendance = round(season_data['Primary: Tickets Sold'].mean())
    filtered_avg_ticket_price = round(season_data['Primary: Ticket Revenue'].sum() / season_data['Primary: Tickets Sold'].sum(), 2)

    # Calculate attendance difference
    attendance_diff = filtered_avg_attendance - overall_avg_attendance

    # Calculate ticket price difference
    ticket_price_diff = round(filtered_avg_ticket_price - overall_avg_ticket_price, 2)

    # Update the statistics_data dictionary to include the difference columns
    statistics_data = [
        {'Data': 'Overall', 'Events': f'{len(df):,}', 'Average Attendance': f'{overall_avg_attendance:,}',
        'Average Ticket Price': f'${overall_avg_ticket_price}', 'Attendance Difference': 0, 'Ticket Price Difference': 0},
        {'Data': 'Filtered', 'Events': f'{len(season_data):,}', 'Average Attendance': f'{filtered_avg_attendance:,}',
        'Average Ticket Price': f'${filtered_avg_ticket_price}', 'Attendance Difference': f'{attendance_diff:,}',
        'Ticket Price Difference': f'${ticket_price_diff}'}
    ]

    return fig, statistics_data

@app.callback(
    Output('statistics-table', 'columns'),
    [Input('ticket-revenue-or-sales', 'value')]
)
def update_table_columns(selected_y_value):
    columns = [
        {'name': 'Data', 'id': 'Data'},
        {'name': 'Events', 'id': 'Events'}
    ]
    
    # Add the new columns
    columns.append({'name': 'Average Attendance', 'id': 'Average Attendance'})
    columns.append({'name': 'Average Ticket Price', 'id': 'Average Ticket Price'})
    columns.append({'name': 'Attendance Difference', 'id': 'Attendance Difference'})
    columns.append({'name': 'Ticket Price Difference', 'id': 'Ticket Price Difference'})
    
    return columns

if __name__ == "__main__":
    app.run_server(debug=True)