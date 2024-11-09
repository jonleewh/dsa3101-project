import pandas as pd

# Daily Weather Data
daily_weather_data = pd.read_csv("Group B/data/daily_weather_data.csv")
temperature_data = pd.DataFrame(daily_weather_data[["Year", "Month", "Day", "Mean Temperature (°C)"]])
# monthly_avg = temperature_data.groupby(['Month'])['Mean Temperature (°C)'].mean()

#print(monthly_avg)

# Hourly Weather Data
hourly_weather_data = pd.read_excel("Group B/data/hourly_weather_data.xlsx")

# data cleaning
hourly_weather_data['Temp'] = hourly_weather_data['Temp'].str.replace('\xa0°C', '')
hourly_weather_data['Wind'] = hourly_weather_data['Wind'].replace("No wind", "0 km/h").str.replace(' km/h', '').astype(float)
hourly_weather_data['Humidity'] = (hourly_weather_data['Humidity'] * 100).astype(int)
hourly_weather_data['Rain'] = hourly_weather_data['Weather'].str.contains('rain', case=False, na=False).astype(int)
hourly_weather_data.drop(['Barometer', 'Visibility', 'Weather'], axis=1, inplace=True)
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
print(hourly_weather_data)

hourly_weather_data.to_csv("Group B/data/weather_data_hour.csv")
