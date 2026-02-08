#importing libraries
import sys
import os
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

#importing data
buildings = pd.read_csv('data/advanced_core/building_metadata.csv')

meter_jan = pd.read_csv('data/advanced_core/meter-readings-jan-2025.csv')

meter_feb = pd.read_csv('data/advanced_core/meter-readings-feb-2025.csv')

meter_march = pd.read_csv('data/advanced_core/meter-readings-march-2025.csv')

meter_april = pd.read_csv('data/advanced_core/meter-readings-april-2025.csv')

weather = pd.read_csv('data/advanced_core/weather_data_hourly_2025.csv')

#functions

def create_wind_rose(wind_speed, wind_direction, title="Wind Rose"):
    # Define wind direction bins (N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW)
    bins = np.arange(0, 360, 22.5)
    direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Define wind speed categories (in m/s)
    speed_bins = [0, 2, 5, 8, 11, 14, 17, 20, np.inf]
    speed_labels = ['Calm (<2)', 'Light (2-5)', 'Moderate (5-8)', 'Fresh (8-11)',
                   'Strong (11-14)', 'Gale (14-17)', 'Storm (17-20)', 'Violent (>20)']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(speed_bins)-1))
    
    # Ensure arrays are numpy arrays
    wind_speed = np.asarray(wind_speed)
    wind_direction = np.asarray(wind_direction)
    
    # Remove any NaN values
    mask = ~(np.isnan(wind_speed) | np.isnan(wind_direction))
    wind_speed = wind_speed[mask]
    wind_direction = wind_direction[mask]
    
    # mod to avoid negative angles and ensure within 0-360
    wind_direction = np.mod(wind_direction, 360)
    
    # Direction binning - using searchsorted for more precise control
    direction_bin_indices = np.digitize(wind_direction, bins) - 1
    direction_bin_indices[direction_bin_indices == -1] = len(direction_labels) - 1
    direction_bin_indices[direction_bin_indices == len(direction_labels)] = 0
    
    # Speed binning
    speed_bin_indices = np.digitize(wind_speed, speed_bins) - 1
    speed_bin_indices[speed_bin_indices == len(speed_labels)] = len(speed_labels) - 1
    
    # Create frequency matrix
    freq_matrix = np.zeros((len(direction_labels), len(speed_labels)))
    
    for dir_idx, speed_idx in zip(direction_bin_indices, speed_bin_indices):
        if 0 <= dir_idx < len(direction_labels) and 0 <= speed_idx < len(speed_labels):
            freq_matrix[dir_idx, speed_idx] += 1
    
    # Convert to percentages
    total_count = len(wind_direction)
    if total_count > 0:
        freq_matrix = freq_matrix / total_count * 100
    
    # Create the wind rose plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # plot set up
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_zero_location('N')  # North at top
    angles = np.linspace(0, 2*np.pi, len(direction_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]])) 
    
    # Plot each wind speed category
    bottom = np.zeros(len(direction_labels) + 1) 
    
    for i in range(len(speed_labels)):
        
        # Get values and close the circle
        values = freq_matrix[:, i]
        values = np.concatenate((values, [values[0]]))
        
        # The bottom array also needs to be closed
        bottom_closed = np.concatenate((bottom[:-1], [bottom[0]])) if len(bottom) > 1 else bottom
        
        ax.bar(angles, values, width=2*np.pi/len(direction_labels), 
               bottom=bottom_closed, color=colors[i], 
               edgecolor='white', 
               linewidth=0.5,
               label=speed_labels[i])
        
        bottom = bottom + values
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(direction_labels)
    
    # Set y limits
    if len(bottom) > 0:
        max_val = bottom.max()
        ax.set_ylim(0, max_val * 1.1)
        yticks = np.arange(0, max_val, max(1, max_val/10))
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{tick:.1f}%' for tick in yticks])
    
    # Add title and legend
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(title='Wind Speed (m/s)', bbox_to_anchor=(1.1, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, ax

def create_animated_wind_rose(weather_df, time_var='hour'):
    """Create an animated wind rose showing changes over time"""
    from matplotlib.animation import FuncAnimation
    
    # Sort by time
    weather_df = weather_df.sort_values(time_var)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Get unique time values
    time_values = sorted(weather_df[time_var].unique())
    
    # Initialize plot
    lines = []
    title = ax.set_title(f'Time: {time_values[0]}')
    
    def update(frame):
        """Update function for animation"""
        ax.clear()
        
        # Get data for this time
        time_val = time_values[frame]
        time_data = weather_df[weather_df[time_var] == time_val]
        
        if len(time_data) > 0:
            # Create wind rose for this time
            create_wind_rose(
                time_data['wind_speed_100m'],
                time_data['wind_direction_100m'],
                title=f'Wind Rose - Hour {time_val}'
            )
        
        return [ax]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(time_values), 
                       interval=500, blit=False)
    
    plt.tight_layout()
    return ani, fig




#convert date to datetime and extract hour
weather['date'] = pd.to_datetime(weather['date'])
weather['hour'] = weather['date'].dt.hour

#wind direction distribution
fig1, ax1 = create_animated_wind_rose(weather, time_var='hour')
plt.show()

#print results or findings
print("=== BUILDING METADATA ===")
print(f"\nAverage building area: {buildings['grossarea'].mean():.0f} sq ft")

#closest building to weather station
weather_station_location = (weather['longitude'].mean(), weather['latitude'].mean())
buildings['distance_to_weather_station'] = np.sqrt((buildings['longitude'] - weather_station_location[0])**2 + 
                                                   (buildings['latitude'] - weather_station_location[1])**2)
closest_by_attr = buildings.loc[buildings['distance_to_weather_station'].idxmin()]

print(f"\nbuilding closest to weather station: {closest_by_attr['buildingname']} also known as {closest_by_attr['alsoknownas']} at {closest_by_attr['longitude']}, {closest_by_attr['latitude']} ")










#ends script
sys.exit()
