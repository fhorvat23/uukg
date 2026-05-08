"""Visualize the spatial and temporal distribution of the USTP dataset"""

# taxi
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
"""

Taxi

"""

###########################################################
#
file_path = '../Processed_data/CHI/CHI_taxi.csv'
taxi_data = pd.read_csv(file_path)

# Create a Folium heat map Travel starting point
taxi_start_m = folium.Map(location=[taxi_data['start_lat'].mean(), taxi_data['start_lng'].mean()], zoom_start=10, tiles="cartodbpositron")
taxi_end_m = folium.Map(location=[taxi_data['end_lat'].mean(), taxi_data['end_lng'].mean()], zoom_start=10, tiles="cartodbpositron")
#
# HeatMap(taxi_data[['start_lat', 'start_lng']].values).add_to(taxi_start_m)
# HeatMap(taxi_data[['end_lat', 'end_lng']].values).add_to(taxi_end_m)
#
# # Save Heatmap
# taxi_start_m.save('taxi_start_m.html')
# taxi_end_m.save('taxi_end_m.html')
#
#
# Extract time information and format
date_format = "%Y-%m-%d %H:%M:%S"
taxi_data['start_time'] = pd.to_datetime(taxi_data['start_time'], format=date_format)

# Plot frequency distribution over time dimension (grouped by hour)

start_data_by_day = taxi_data['start_time'].groupby(taxi_data['start_time'].dt.to_period('D')).count()

# Create an image with a specified size (e.g. 12 inches wide and 6 inches tall)
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(start_data_by_day.index.to_timestamp(), start_data_by_day)

# Set titles and axis labels
plt.title('Frequency distribution of CHI taxi over time', fontsize=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Frequency', fontsize=24)

# Modify the x-axis scale to only show each month
xtick_locator = mdates.MonthLocator()
xtick_formatter = mdates.DateFormatter('%b %Y')
ax.xaxis.set_major_locator(xtick_locator)
ax.xaxis.set_major_formatter(xtick_formatter)
plt.xticks(rotation=45)

# Adjust the bounds to ensure the full abscissa is displayed
plt.tight_layout()

# Save frequency distribution plot
plt.savefig('time_taxi_distribution.png')

###########################################################

"""

Bike


"""

# # Read CSV file
file_path = '../Processed_data/CHI/CHI_bike_road.csv'
bike_data = pd.read_csv(file_path)

# Create Folium heat map travel starting point
bike_start_m = folium.Map(location=[bike_data['start_lat'].mean(), bike_data['start_lng'].mean()], zoom_start=10, tiles="cartodbpositron")
bike_end_m = folium.Map(location=[bike_data['end_lat'].mean(), bike_data['end_lng'].mean()], zoom_start=10, tiles="cartodbpositron")

# HeatMap(bike_data[['start_lat', 'start_lng']].values).add_to(bike_start_m)
# HeatMap(bike_data[['end_lat', 'end_lng']].values).add_to(bike_end_m)
#
# # Save heat map
# bike_start_m.save('bike_start_m.html')
# bike_end_m.save('bike_end_m.html')
#
#
# # Extract time information and format it
date_format = "%Y-%m-%d %H:%M:%S"
bike_data['start_time'] = pd.to_datetime(bike_data['start_time'], format=date_format)

# # Plot the frequency distribution in the time dimension (grouped by hour)

start_data_by_day = bike_data['start_time'].groupby(bike_data['start_time'].dt.to_period('D')).count()

# Create an image with a specified size (e.g. 12 inches wide and 6 inches tall)
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(start_data_by_day.index.to_timestamp(), start_data_by_day)

# Set titles and axis labels
plt.title('Frequency distribution of CHI bike over time', fontsize=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Frequency', fontsize=24)

# Modify the x-axis scale to only show each month
xtick_locator = mdates.MonthLocator()
xtick_formatter = mdates.DateFormatter('%b %Y')
ax.xaxis.set_major_locator(xtick_locator)
ax.xaxis.set_major_formatter(xtick_formatter)
plt.xticks(rotation=45)

# Adjust the bounds to ensure the full abscissa is displayed
plt.tight_layout()

# Save frequency distribution plot
plt.savefig('time_bike_distribution.png')

"""

Human


"""

# # Read CSV file
file_path = '../Processed_data/CHI/CHI_human.csv'
human_data = pd.read_csv(file_path)

# Create Folium heat map travel starting point
human_start_m = folium.Map(location=[human_data['start_lat'].mean(), human_data['start_lng'].mean()], zoom_start=10, tiles="cartodbpositron")
human_end_m = folium.Map(location=[human_data['end_lat'].mean(), human_data['end_lng'].mean()], zoom_start=10, tiles="cartodbpositron")

# HeatMap(human_data[['start_lat', 'start_lng']].values).add_to(human_start_m)
# HeatMap(human_data[['end_lat', 'end_lng']].values).add_to(human_end_m)
#
# # Save heat map
# human_start_m.save('human_start_m.html')
# human_end_m.save('human_end_m.html')
#
#
# # Extract time information and format it
date_format = "%Y-%m-%d %H:%M:%S"
human_data['start_time'] = pd.to_datetime(human_data['start_time'], format=date_format)

# Plot frequency distribution over time (grouped by hour)

start_data_by_day = human_data['start_time'].groupby(human_data['start_time'].dt.to_period('D')).count()

# Create an image with a specified size (e.g. 12 inches wide and 6 inches tall)
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(start_data_by_day.index.to_timestamp(), start_data_by_day)

# Set titles and axis labels
plt.title('Frequency distribution of CHI human over time', fontsize=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Frequency', fontsize=24)

# Modify the x-axis scale to only show each month
xtick_locator = mdates.MonthLocator()
xtick_formatter = mdates.DateFormatter('%b %Y')
ax.xaxis.set_major_locator(xtick_locator)
ax.xaxis.set_major_formatter(xtick_formatter)
plt.xticks(rotation=45)

# Adjust the bounds to ensure the full abscissa is displayed
plt.tight_layout()

# Save frequency distribution plot
plt.savefig('time_human_distribution.png')

# """
#
# crime
#
#
# """
#
# Read CSV file
file_path = '../Processed_data/CHI/CHI_crime.csv'

crime_data = pd.read_csv(file_path)
crime_data['time'] = pd.to_datetime(crime_data['time'], format="%m/%d/%Y %I:%M:%S %p")
crime_data['time'] = crime_data['time'].dt.strftime("%Y-%m-%d %H:%M:%S")

# # Create Folium heat map travel starting point
# crime_m = folium.Map(location=[crime_data['lat'].mean(), crime_data['lng'].mean()], zoom_start=10, tiles="cartodbpositron")
#
# HeatMap(crime_data[['lat', 'lng']].values).add_to(crime_m)
#
# # Save heat map
# crime_m.save('crime_m.html')
#
#
# # Extract time information and format it
date_format = "%Y-%m-%d %H:%M:%S"
crime_data['time'] = pd.to_datetime(crime_data['time'], format=date_format)

# Plot frequency distribution over time (grouped by hour)

start_data_by_day = crime_data['time'].groupby(crime_data['time'].dt.to_period('D')).count()

# Create an image with a specified size (e.g. 12 inches wide and 6 inches tall)
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(start_data_by_day.index.to_timestamp(), start_data_by_day)

# Set titles and axis labels
plt.title('Frequency distribution of CHI crime over time', fontsize=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Frequency', fontsize=24)

# Modify the x-axis scale to only show each month
xtick_locator = mdates.MonthLocator()
xtick_formatter = mdates.DateFormatter('%b %Y')
ax.xaxis.set_major_locator(xtick_locator)
ax.xaxis.set_major_formatter(xtick_formatter)
plt.xticks(rotation=45)

# Adjust the bounds to ensure the full abscissa is displayed
plt.tight_layout()

# Save frequency distribution plot
plt.savefig('time_crime_distribution.png')

"""

311 service


"""

# Read CSV file
file_path = '../Processed_data/CHI/CHI_311_service.csv'
service_data = pd.read_csv(file_path)

# # Create Folium heat map travel starting point
# service_m = folium.Map(location=[service_data['lat'].mean(), service_data['lng'].mean()], zoom_start=10, tiles="cartodbpositron")
#
# HeatMap(service_data[['lat', 'lng']].values).add_to(service_m)
#
# # Save heat map
# service_m.save('service_m.html')
#
#
# Extract time information and format it
date_format = "%Y-%m-%d %H:%M:%S"
service_data['time'] = pd.to_datetime(service_data['time'], format=date_format)

# Plot frequency distribution over time (grouped by hour)

start_data_by_day = service_data['time'].groupby(service_data['time'].dt.to_period('D')).count()

# Create an image with a specified size (e.g. 12 inches wide and 6 inches tall)
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(start_data_by_day.index.to_timestamp(), start_data_by_day)

# Set titles and axis labels
plt.title('Frequency distribution of CHI 311 service over time', fontsize=24)
plt.xlabel('Date', fontsize=24)
plt.ylabel('Frequency', fontsize=24)

# Modify the x-axis scale to only show each month
xtick_locator = mdates.MonthLocator()
xtick_formatter = mdates.DateFormatter('%b %Y')
ax.xaxis.set_major_locator(xtick_locator)
ax.xaxis.set_major_formatter(xtick_formatter)
plt.xticks(rotation=45)

# Adjust the bounds to ensure the full abscissa is displayed
plt.tight_layout()

# Save frequency distribution plot
plt.savefig('time_service_distribution.png')