import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
import dash
from dash import dash_table
import openpyxl
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import warnings
warnings.filterwarnings('ignore')

# Load the dataset - Replace 'sorted_events' with your actual dataset
df = pd.read_excel('Events.xlsx')
df = df.drop(columns=['Unnamed: 0'])

data = df

# Select relevant features and target
categorical_features = data[['Day of Week', 'Event Time']]
quantitative_features = data['Median Days Before']
target_tickets_sold = data['Primary: Tickets Sold']
target_ticket_revenue = data['Primary: Ticket Revenue']

# Convert categorical features to one-hot encoding
categorical_features = pd.get_dummies(categorical_features, columns=['Day of Week', 'Event Time'])

# Combine the one-hot encoded categorical features with the quantitative feature
features = pd.concat([categorical_features, quantitative_features], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train_tickets_sold, y_test_tickets_sold = train_test_split(features, target_tickets_sold, test_size=0.3, random_state=42)
X_train, X_test, y_train_ticket_revenue, y_test_ticket_revenue = train_test_split(features, target_ticket_revenue, test_size=0.3, random_state=42)

# Create and train the Random Forest model for ticket sold prediction
rf_tickets_sold = RandomForestRegressor(n_estimators=100, random_state=42)
rf_tickets_sold.fit(X_train, y_train_tickets_sold)

# Predict on the test set
y_pred_tickets_sold = rf_tickets_sold.predict(X_test)

# Evaluate the model for ticket sold prediction
mse_tickets_sold = mean_squared_error(y_test_tickets_sold, y_pred_tickets_sold)
r2_tickets_sold = r2_score(y_test_tickets_sold, y_pred_tickets_sold)

# Create and train the Random Forest model for ticket revenue prediction
rf_ticket_revenue = RandomForestRegressor(n_estimators=100, random_state=42)
rf_ticket_revenue.fit(X_train, y_train_ticket_revenue)

# Predict on the test set
y_pred_ticket_revenue = rf_ticket_revenue.predict(X_test)

# Evaluate the model for ticket revenue prediction
mse_ticket_revenue = mean_squared_error(y_test_ticket_revenue, y_pred_ticket_revenue)
r2_ticket_revenue = r2_score(y_test_ticket_revenue, y_pred_ticket_revenue)

# Sample input data for predictions
input_data = pd.DataFrame({
    'Day of Week_Friday': [0], 'Day of Week_Monday': [0], 'Day of Week_Saturday': [1],
       'Day of Week_Sunday': [0], 'Day of Week_Thursday': [0], 'Day of Week_Tuesday': [0],
       'Day of Week_Wednesday': [0], 'Event Time_12:30:00': [0], 'Event Time_13:00:00': [0],
       'Event Time_15:00:00': [0], 'Event Time_16:00:00': [0], 'Event Time_18:00:00': [0],
       'Event Time_18:30:00': [0], 'Event Time_19:00:00': [0], 'Event Time_19:30:00': [1],
       'Median Days Before': [450]
})

# Use the trained model to make predictions
predicted_tickets_sold = rf_tickets_sold.predict(input_data)
predicted_ticket_revenue = rf_ticket_revenue.predict(input_data)

# Custom color mapping for seasons
custom_colors = {
    '2018-2019': 'darkslategrey',
    '2019-2020': 'lightseagreen',
    '2021-2022': 'aqua',
    '2022-2023': 'skyblue'
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

# Create a reusable function to generate day of week and event time dropdown options
def generate_dropdown_options(label_prefix, values, selected_value):
    return [
        {'label': value, 'value': value}
        for value in values
    ]

# Define the possible day of week and event time values
day_of_week_values = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
event_time_values = df['Event Time'].sort_values().unique()

# Create the day of week dropdown options
day_of_week_dropdown_options = generate_dropdown_options('Day of Week', day_of_week_values, 'all_days')

# Create the event time dropdown options
event_time_dropdown_options = generate_dropdown_options('Event Time', event_time_values, '12:30:00')

opponent_dropdown_options = generate_dropdown_options('Opponents', opponent_options, 'all_opponents')

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
                options=day_of_week_dropdown_options,
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
    ], style={'flex': '1'}),

    html.Div([
        html.H3("Top 5 Events", style={'color': 'black'}),
        dash_table.DataTable(
            id='top-5-rows-table',
            style_table={
                'width': '100%',
                'maxHeight': '300px',  # Set the maximum height of the table
                'overflowY': 'auto',   # Enable vertical scrolling when the content exceeds the maximum height
                'backgroundColor': 'lightgrey'
            },
            style_cell={'textAlign': 'left', 'color': 'black'},
        )
    ]),
    html.Div([
        html.H3("Future Predictions"),
        html.Div([
            dcc.Dropdown(
                id="prediction-day-of-week",
                options=day_of_week_dropdown_options,
                value='all_days',
                placeholder="Select Day of the Week",
                style={'width': '47%', 'display': 'inline-block', 'color': 'black'},
            ),
            dcc.Dropdown(
                id="prediction-event-time",
                options=event_time_dropdown_options,
                value='12:30:00',
                placeholder="Select Event Time",
                style={'width': '47%', 'display': 'inline-block', 'color': 'black'},
            ),
            dcc.Input(
                id="prediction-median-days-before",
                type="number",
                placeholder="Enter Median Days Before",
                style={'width': '47%', 'display': 'inline-block', 'color': 'black'},
            ),
            html.Button(
                "Predict",
                id="predict-button",
                style={'color': 'black', 'backgroundColor': 'lightgrey'}
            ),
        ]),
        html.Div([
            html.H3("Predicted Results", style={'color': 'black'}),
            html.P("Predicted Tickets Sold: ", style={'color': 'black'}),
            html.Div(id="predicted-tickets-sold", style={'color': 'black'}),
            html.P("Predicted Ticket Revenue: ", style={'color': 'black'}),
            html.Div(id="predicted-ticket-revenue", style={'color': 'black'}),
        ]),
    ]),
    
    # New section to compare predictions with filtered and overall median ticket revenue and sales
    html.Div([
        html.H3("Compare Predictions"),
        html.Div([
            html.P("Overall Median Tickets Sold: ", style={'color': 'black'}),
            html.Div(id="overall-median-tickets-sold", style={'color': 'black'}),
        ]),
        html.Div([
            html.P("Overall Median Ticket Revenue: ", style={'color': 'black'}),
            html.Div(id="overall-median-ticket-revenue", style={'color': 'black'}),
        ]),
    ])
])

@app.callback(
    [dash.dependencies.Output("seasonal-ticket-revenue", "figure"),
     dash.dependencies.Output('statistics-table', 'data'),
     dash.dependencies.Output('top-5-rows-table', 'data')],
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
        season_avg = season_avg.rename(columns={0: "Events"})
        fig = px.bar(season_avg, x="Season", y=selected_y_value,
                     color="Season", color_discrete_map=custom_colors,
                     labels={selected_y_value: selected_y_value},
                     title=f"{selected_y_value} for All Seasons",
                     hover_data=["Events"])
    else:
        # If specific seasons are selected, show the selected seasons with custom colors
        season_count = season_data.groupby("Season").size().reset_index()
        season_data = season_data.merge(season_count, on="Season")
        season_data = season_data.rename(columns={0: "Event Count"})
        fig = px.line(season_data, x="Event Date", y=selected_y_value, markers=True, hover_data=['Opponent', 'Day of Week', 'Event Time', 'Event Count', 'Median Days Before'],
                         color="Season", color_discrete_map=custom_colors,
                         labels={selected_y_value: selected_y_value},
                         title=f"{selected_y_value} for Selected Seasons")

    # Update axis labels and add hover tooltip
    fig.update_xaxes(title_text="Event Date")
    fig.update_yaxes(title_text=selected_y_value)
    fig.update_layout(plot_bgcolor='darkgray', paper_bgcolor='lightgray')

    # Calculate average attendance and average ticket price for the entire dataset
    overall_avg_attendance = round(df['Primary: Tickets Sold'].mean())
    overall_avg_ticket_price = round(df['Primary: Ticket Revenue'].sum() / df['Primary: Tickets Sold'].sum(), 2)

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

    top_5_rows = season_data.sort_values(by=selected_y_value, ascending=False).head().to_dict('records')

    return fig, statistics_data, top_5_rows

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

@app.callback(
    [dash.dependencies.Output("predicted-tickets-sold", "children"),
     dash.dependencies.Output("predicted-ticket-revenue", "children"),
     dash.dependencies.Output("overall-median-tickets-sold", "children"),
     dash.dependencies.Output("overall-median-ticket-revenue", "children")],
    [dash.dependencies.Input("predict-button", "n_clicks")],
    [dash.dependencies.State("prediction-day-of-week", "value"),
     dash.dependencies.State("prediction-event-time", "value"),
     dash.dependencies.State("prediction-median-days-before", "value")]
)
def predict_results(n_clicks, day_of_week, event_time, median_days_before):
    try:
    
        if n_clicks is None:
            return "N/A", "N/A", "N/A", "N/A",
        dy = median_days_before
        
        # Create a DataFrame with an index
        input_data = pd.DataFrame({
            'Day of Week_Friday': [0],  # Provide a list with a single value
            'Day of Week_Monday': [0],
            'Day of Week_Saturday': [0],
            'Day of Week_Sunday': [0],
            'Day of Week_Thursday': [0],
            'Day of Week_Tuesday': [0],
            'Day of Week_Wednesday': [0],
            'Event Time_12:30:00': [0],
            'Event Time_13:00:00': [0],
            'Event Time_15:00:00': [0],
            'Event Time_16:00:00': [0],
            'Event Time_18:00:00': [0],
            'Event Time_18:30:00': [0],
            'Event Time_19:00:00': [0],
            'Event Time_19:30:00': [0],
            'Median Days Before': [dy]
        })

        # Set the corresponding columns to 1 based on user input
        input_data.loc[0, f'Day of Week_{day_of_week}'] = 1
        input_data.loc[0, f'Event Time_{event_time}'] = 1
        
        # Use the trained Random Forest models to make predictions
        predicted_tickets_sold = rf_tickets_sold.predict(input_data)
        predicted_ticket_revenue = rf_ticket_revenue.predict(input_data)
        
        predicted_tickets_sold_str = "{:,.0f}".format(round(predicted_tickets_sold[0]))
        predicted_ticket_revenue_str = "${:,.2f}".format(round(predicted_ticket_revenue[0]))
    
        # Calculate overall median ticket revenue and sales
        overall_median_tickets_sold = df['Primary: Tickets Sold'].median()
        overall_median_ticket_revenue = df['Primary: Ticket Revenue'].median()

        return (
            str(predicted_tickets_sold_str),
            str(predicted_ticket_revenue_str),
            "{:,.0f}".format(round(overall_median_tickets_sold)),
            "${:,.2f}".format(round(overall_median_ticket_revenue))
        )
    except Exception as e:
        traceback.print_exc()
        return "Error", "Error", "Error", "Error"

if __name__ == "__main__":
    app.run_server(debug=True)