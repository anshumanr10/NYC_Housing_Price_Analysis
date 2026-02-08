import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

def create_single_wind_rose(wind_speed, wind_direction, ax=None, title=""):
    """
    Create a single wind rose plot on given axes
    Modified from your working function
    """
    # Ensure arrays are numpy arrays
    wind_speed = np.asarray(wind_speed)
    wind_direction = np.asarray(wind_direction)
    
    # Remove any NaN values
    mask = ~(np.isnan(wind_speed) | np.isnan(wind_direction))
    wind_speed = wind_speed[mask]
    wind_direction = wind_direction[mask]
    
    if len(wind_speed) == 0:
        if ax:
            ax.set_title(f"{title} (No data)", fontsize=10)
        return ax
    
    # Define wind direction bins
    bins = np.arange(0, 360, 22.5)
    direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Define wind speed categories
    speed_bins = [0, 2, 5, 8, 11, 14, 17, 20, np.inf]
    speed_labels = ['Calm (<2)', 'Light (2-5)', 'Moderate (5-8)', 'Fresh (8-11)',
                   'Strong (11-14)', 'Gale (14-17)', 'Storm (17-20)', 'Violent (>20)']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(speed_bins)-1))
    
    # Handle wind direction properly
    wind_direction = np.mod(wind_direction, 360)
    
    # Direction binning
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
    freq_matrix = freq_matrix / total_count * 100 if total_count > 0 else freq_matrix
    
    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
    
    # Clear existing plot
    ax.clear()
    
    # Set up the plot
    ax.set_theta_direction(-1)  # clockwise
    ax.set_theta_zero_location('N')  # North at top
    
    # Create bars for each direction
    angles = np.linspace(0, 2*np.pi, len(direction_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the circle
    
    # Plot each wind speed category
    bottom = np.zeros(len(direction_labels) + 1)
    
    for i in range(len(speed_labels)):
        values = freq_matrix[:, i]
        values = np.concatenate((values, [values[0]]))  # Close the circle
        bottom_closed = np.concatenate((bottom[:-1], [bottom[0]])) if len(bottom) > 1 else bottom
        
        ax.bar(angles, values, width=2*np.pi/len(direction_labels), 
               bottom=bottom_closed, color=colors[i], 
               edgecolor='white', linewidth=0.5,
               label=speed_labels[i])
        
        bottom = bottom + values
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(direction_labels)
    
    if len(bottom) > 0:
        max_val = bottom.max()
        ax.set_ylim(0, max_val * 1.1)
        yticks = np.arange(0, max_val, max(1, max_val/10))
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{tick:.1f}%' for tick in yticks])
    
    ax.set_title(f'{title}\nN={total_count}', fontsize=12, fontweight='bold', pad=20)
    
    # Add legend only if it's the first frame (to avoid duplication)
    #if not hasattr(ax, 'legend_added'):
    ax.legend(title='Wind Speed (m/s)', loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.legend_added = True
    
    return ax

def create_animated_wind_rose(weather_df, time_var='hour', save_path=None, fps=2):
    """
    Create an animated wind rose showing changes over time
    
    Parameters:
    -----------
    weather_df : pandas DataFrame
        Weather data with wind_speed_10m, wind_direction_10m, and time_var columns
    time_var : str
        Column name for time variable (e.g., 'hour', 'month', 'season')
    save_path : str, optional
        Path to save the animation (e.g., 'wind_rose_animation.gif')
    fps : int
        Frames per second for the animation
    
    Returns:
    --------
    ani : FuncAnimation object
    fig : matplotlib Figure object
    """
    
    # Sort by time
    weather_df = weather_df.sort_values(time_var)
    
    # Get unique time values
    time_values = sorted(weather_df[time_var].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(projection='polar'))
    
    # Set initial title
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    # Initialize with empty data for first frame
    def init():
        ax.clear()
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_title('Loading animation...', fontsize=14, fontweight='bold')
        return ax,
    
    def update(frame):
        """Update function for animation"""
        ax.clear()
        
        # Get data for this time
        time_val = time_values[frame]
        time_data = weather_df[weather_df[time_var] == time_val]
        
        if len(time_data) > 0:
            # Create wind rose for this time
            create_single_wind_rose(
                time_data['wind_speed_100m'].values,
                time_data['wind_direction_100m'].values,
                ax=ax,
                title=f'{time_var.capitalize()}: {time_val}'
            )
        
        return ax,
    
    # Create animation
    print(f"Creating animation with {len(time_values)} frames...")
    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(time_values),
        init_func=init,
        interval=1000/fps,  # milliseconds per frame
        blit=False,
        repeat=True,
        repeat_delay=2000  # 2 second delay before repeating
    )
    
    plt.tight_layout()
    
    # Save animation if path is provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        # Save as GIF
        if save_path.endswith('.gif'):
            ani.save(save_path, writer=PillowWriter(fps=fps))
        # Save as MP4 (requires ffmpeg)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        print("Animation saved!")
    
    return ani, fig

def display_animation(ani):
    """Display the animation in Jupyter notebook"""
    try:
        from IPython.display import HTML
        return HTML(ani.to_jshtml())
    except:
        print("Animation created. Use plt.show() to view or save to file.")
        return ani

# Example usage:
if __name__ == "__main__":
    # Load your data
    weather = pd.read_csv('data/advanced_core/weather_data_hourly_2025.csv')
    
    # Convert date and extract hour
    weather['date'] = pd.to_datetime(weather['date'])
    weather['hour'] = weather['date'].dt.hour
    weather['month'] = weather['date'].dt.month
    weather['season'] = weather['date'].dt.month % 12 // 3 + 1  # 1:Winter, 2:Spring, etc.
    
    print(f"Dataset has {len(weather)} records")
    print(f"Time range: {weather['date'].min()} to {weather['date'].max()}")
    print(f"Hours available: {sorted(weather['hour'].unique())}")
    
    # Option A: Animate by hour (24 frames)
    print("\n=== Creating animation by hour ===")
    ani_hour, fig_hour = create_animated_wind_rose(
        weather_df=weather,
        time_var='hour',  # Animate through hours 0-23
        save_path='wind_rose_by_hour.gif',  # Optional: save to file
        fps=6  # 12 frames per second
    )
    
    # Display in Jupyter notebook
    # display_animation(ani_hour)
    
    # Or show in matplotlib window
    plt.show()
    
    # Option B: Animate by month (12 frames - faster)
    print("\n=== Creating animation by month ===")
    ani_month, fig_month = create_animated_wind_rose(
        weather_df=weather,
        time_var='month',
        save_path='wind_rose_by_month.gif',
        fps=3
    )
    
    plt.show()