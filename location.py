#importing libraries
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#importing data
buildings = pd.read_csv('data/advanced_core/building_metadata.csv')

meter_jan = pd.read_csv('data/advanced_core/meter-readings-jan-2025.csv')

meter_feb = pd.read_csv('data/advanced_core/meter-readings-feb-2025.csv')

meter_march = pd.read_csv('data/advanced_core/meter-readings-march-2025.csv')

meter_april = pd.read_csv('data/advanced_core/meter-readings-april-2025.csv')

weather = pd.read_csv('data/advanced_core/weather_data_hourly_2025.csv')

#building size distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(buildings['grossarea'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of Building Gross Area')
plt.xlabel('Gross Area (sq ft)')
plt.ylabel('Number of Buildings')
#plt.show()

#Geographic distribution of buildings
plt.figure(figsize=(10, 8))
plt.scatter(buildings['longitude'], buildings['latitude'], 
            s=buildings['grossarea']/500, alpha=0.6, 
            c=buildings['floorsaboveground'], cmap='viridis')
plt.colorbar(label='Floors Above Ground')
plt.title('Geographic Distribution of Buildings (Size = Area, Color = Floors)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#weather data overview
meter_data = pd.concat([meter_jan, meter_feb, meter_march, meter_april])

#seperate into utility type




#print results
print("=== BUILDING METADATA ===")
print(f"\nAverage building area: {buildings['grossarea'].mean():.0f} sq ft")




#ends script
sys.exit()
