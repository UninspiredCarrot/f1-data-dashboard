import pandas as pd
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection
import fastf1
import base64
import io
from fastf1 import plotting
import matplotlib as mpl

# Set the backend to 'Agg' for non-interactive plotting
plt.switch_backend('Agg')

fastf1.Cache.enable_cache('cache')

def create_figure(selected_driver):
    # Load session data
    session = fastf1.get_session(2024, 'Suzuka', 'R')
    session.load()

    # Pick the fastest lap for the selected driver
    lap = session.laps.pick_driver(selected_driver).pick_fastest()
    telemetry = lap.get_telemetry()

    x = np.array(telemetry['X'].values)
    y = np.array(telemetry['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gear = telemetry['nGear'].to_numpy().astype(float)

    cmap = colormaps['Paired']
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to free memory
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

COMPOUND_COLORS = {
    'SOFT': '#D32F2F',        # Darker red
    'MEDIUM': '#FFB300',      # Darker yellow
    'HARD': '#757575',        # Gray for Hard
    'INTERMEDIATE': '#388E3C', # Darker green
    'WET': '#1976D2' }

def create_scatterplot(selected_driver):
    # Load session data for 2024 Suzuka GP
    race = fastf1.get_session(2024, 'Suzuka', 'R')
    race.load()

    # Filter the laps for the selected driver
    driver_laps = race.laps.pick_driver(selected_driver).pick_quicklaps().reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=driver_laps,
        x="LapNumber",
        y="LapTime",
        ax=ax,
        hue="Compound",
        palette=COMPOUND_COLORS,  # Use manually defined compound colors
        s=80,
        linewidth=0,
        legend='auto'
    )

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")
    ax.invert_yaxis()
    plt.suptitle(f"{selected_driver} Lap Times in the 2024 Japan GP (Suzuka)")

    plt.grid(color='w', which='major', axis='both')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

# Function to create the telemetry line graph
def create_speed_graph(selected_driver):
    year = 2024
    wknd = 'Suzuka'
    ses = 'R'

    # Load the session and select the desired data
    session = fastf1.get_session(year, wknd, ses)
    session.load()
    lap = session.laps.pick_driver(selected_driver).pick_fastest()

    # Get telemetry data
    x = lap.telemetry['X']              # values for x-axis
    y = lap.telemetry['Y']              # values for y-axis
    color = lap.telemetry['Speed']      # value to base color gradient on

    # Create segments for line collection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a plot with title and adjust some settings to make it look good
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
    fig.suptitle(f'{session.event.name} {year} - {selected_driver} - Speed', size=24, y=0.97)

    # Adjust margins and turn off axis
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis('off')

    # Create background track line
    ax.plot(lap.telemetry['X'], lap.telemetry['Y'],
            color='black', linestyle='-', linewidth=16, zorder=0)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=norm,
                        linestyle='-', linewidth=5)

    # Set the values used for colormapping
    lc.set_array(color)

    # Merge all line segments together
    line = ax.add_collection(lc)

    # Finally, create a color bar as a legend
    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=mpl.cm.plasma,
                                       orientation="horizontal")

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to free memory
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

# Create the Dash app
app = Dash(__name__)

# Load the session data to get the list of drivers
session = fastf1.get_session(2024, 'Suzuka', 'R')
session.load()
drivers = session.laps['Driver'].unique()

app.layout = html.Div(
    children=[
        html.H1(children="2024 Japan Grand Prix Data Dashboard"),
        html.P(children="Select a driver to visualize their lap telemetry."),
        
        # Dropdown for driver selection
        dcc.Dropdown(
            id='driver-dropdown',
            options=[{'label': driver, 'value': driver} for driver in drivers],
            value=drivers[0]  # Default value to the first driver
        ),
        
        html.Div(id='driver-graph-container'),
        html.Img(id='lap-graph'),
        html.Img(id='scatter-plot'),
        html.Img(id='speed-graph') 
    ]
)

@app.callback(
    Output('lap-graph', 'src'),
    Output('speed-graph', 'src'),  
    Output('scatter-plot', 'src'),  # Add scatter plot output
    Input('driver-dropdown', 'value')
)
def update_figures(selected_driver):
    image_src = create_figure(selected_driver)
    speed_graph_src = create_speed_graph(selected_driver)
    scatterplot_src = create_scatterplot(selected_driver)  # Generate the scatterplot
    return image_src, speed_graph_src, scatterplot_src  # Return all three images

if __name__ == "__main__":
    app.run_server(debug=True)
