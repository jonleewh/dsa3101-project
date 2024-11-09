import pandas as pd

# Daily Weather Data
daily_weather_data = pd.read_csv("Group B/data/daily_weather_data.csv")
daily_weather_data = daily_weather_data[["Year", "Month", "Day", "Mean Temperature (°C)", "Daily Rainfall Total (mm)"]]
daily_weather_data["Mean Temperature (°C)"] = pd.to_numeric(daily_weather_data["Mean Temperature (°C)"], errors='coerce')
daily_weather_data["Daily Rainfall Total (mm)"] = pd.to_numeric(daily_weather_data["Daily Rainfall Total (mm)"], errors='coerce')
monthly_avg = daily_weather_data.groupby(['Year', 'Month']).agg(
    avg_temperature = ('Mean Temperature (°C)', 'mean'),
    avg_rainfall = ('Daily Rainfall Total (mm)', 'mean')).reset_index()

monthly_avg.to_csv("Group B/data/weather_data_monthly_avg.csv")

# Hourly Weather Data
hourly_weather_data = pd.read_excel("Group B/data/hourly_weather_data.xlsx")

# data cleaning
hourly_weather_data['Temp'] = hourly_weather_data['Temp'].str.replace('\xa0°C', '')
hourly_weather_data['Rain'] = hourly_weather_data['Weather'].str.contains('rain', case=False, na=False).astype(int)
hourly_weather_data.drop(['Barometer', 'Visibility', 'Weather', 'Wind', 'Humidity'], axis=1, inplace=True)
hourly_weather_data['Date'] = pd.to_datetime(hourly_weather_data['Date'])
hourly_weather_data['Time'] = pd.to_datetime(hourly_weather_data['Time'], format='%H:%M:%S').dt.time
hourly_weather_data = hourly_weather_data[
    (hourly_weather_data['Time'] >= pd.to_datetime("09:00:00", format='%H:%M:%S').time()) &
    (hourly_weather_data['Time'] <= pd.to_datetime("22:00:00", format='%H:%M:%S').time())
]

# Information about each dates
dates_data = pd.read_csv("Group B/data/singapore_holidays_nov2023_oct2024.csv")
dates_data['year'] = dates_data['Month'].apply(lambda x: 2023 if x in [11, 12] else 2024)
dates_data['Date'] = pd.to_datetime(dates_data[['year', 'Month', 'Day']])
dates_data['Date'] = dates_data['Date'].dt.strftime('%Y-%m-%d')
columns = ['Date'] + [col for col in dates_data.columns if col != 'Date']
dates_data['Date'] = pd.to_datetime(dates_data['Date'])
dates_data = dates_data[columns].drop(['Month', 'Day', 'year'], axis=1)

# combine with hourly weather data
hourly_weather_data = pd.merge(hourly_weather_data, dates_data, on='Date', how='left')
hourly_weather_data.to_csv("Group B/data/weather_data_hour.csv")
