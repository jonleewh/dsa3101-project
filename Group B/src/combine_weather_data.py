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
rain_keywords = ['rain', 'showers', 'thunderstorms', 'sprinkles', 'thundershowers']
hourly_weather_data['rain'] = hourly_weather_data['Weather'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in rain_keywords) else 0)
hourly_weather_data.drop(['Barometer', 'Visibility', 'Weather', 'Wind', 'Humidity'], axis=1, inplace=True)
hourly_weather_data['Date'] = pd.to_datetime(hourly_weather_data['Date'])
hourly_weather_data['Time'] = pd.to_datetime(hourly_weather_data['Time'], format='%H:%M:%S').dt.time
hourly_weather_data = hourly_weather_data[
    (hourly_weather_data['Time'] >= pd.to_datetime("10:00:00", format='%H:%M:%S').time()) &
    (hourly_weather_data['Time'] <= pd.to_datetime("21:00:00", format='%H:%M:%S').time())
]

# Information about each dates
dates_data = pd.read_csv("Group B/data/singapore_holidays_oct2023_sep2024.csv")
dates_data['year'] = dates_data['Month'].apply(lambda x: 2023 if x in [10, 11, 12] else 2024)
dates_data['Date'] = pd.to_datetime(dates_data[['year', 'Month', 'Day']])
dates_data['Date'] = dates_data['Date'].dt.strftime('%Y-%m-%d')
columns = ['Date'] + [col for col in dates_data.columns if col != 'Date']
dates_data['Date'] = pd.to_datetime(dates_data['Date'])
dates_data = dates_data[columns].drop(['Month', 'Day', 'year'], axis=1)
dates_data['Weekend'] = dates_data['Day of the Week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# combine with hourly weather data
hourly_weather_data = pd.merge(hourly_weather_data, dates_data, on='Date', how='left')
hourly_weather_data['Date'] = pd.to_datetime(hourly_weather_data['Date'])
hourly_weather_data['Time'] = pd.to_datetime(hourly_weather_data['Time'], format='%H:%M:%S').dt.time
hourly_weather_data['datetime'] = pd.to_datetime(hourly_weather_data['Date'].dt.date.astype(str)
                                                 + " " + hourly_weather_data['Time'].astype(str))
def categorise_day(row): # create the 'type_of_day' column
    if row['School Holiday'] == 1 and row['Public Holiday'] == 1:
        return 1  # school holiday and public holiday
    elif row['School Holiday'] == 1 and row['Public Holiday'] == 0:
        return 2  # school holiday and not a public holiday
    elif row['School Holiday'] == 0 and row['Public Holiday'] == 1:
        return 3  # not a school holiday but public holiday
    elif row['Weekend'] == 0 and row['School Holiday'] == 0 and row['Public Holiday'] == 0:
        return 4  # weekday with no school/public holiday
    elif row['Weekend'] == 1 and row['School Holiday'] == 0 and row['Public Holiday'] == 0:
        return 5  # weekend but no school/public holiday
hourly_weather_data['type_of_day'] = hourly_weather_data.apply(categorise_day, axis = 1)

new_hourly_weather_data = pd.DataFrame()

for date in hourly_weather_data['Date'].unique():
    # Extract the subset of the data for that day
    day_data = hourly_weather_data[hourly_weather_data['Date'] == date]
    
    # Generate the minute-level timestamps for 10am to 9pm
    day_start = pd.to_datetime(f"{date} 10:00:00")
    day_end = pd.to_datetime(f"{date} 21:00:00")
    minute_range = pd.date_range(start=day_start, end=day_end, freq='min')
    
    # Repeat the weather data for every minute within the range
    temp_df = pd.DataFrame({
        'datetime': minute_range,
        'Date': [date] * len(minute_range),
        'Time': minute_range.time,
        'Temp': day_data['Temp'].iloc[0],  # Assuming the weather conditions are the same throughout the day
        'rain': day_data['rain'].iloc[0],
        'Day of the Week': day_data['Day of the Week'].iloc[0],
        'School Holiday': day_data['School Holiday'].iloc[0],
        'Public Holiday': day_data['Public Holiday'].iloc[0],
        'Weekend': day_data['Weekend'].iloc[0],
        'type_of_day': day_data['type_of_day'].iloc[0]
    })
    
    # Append the generated data for the day
    new_hourly_weather_data = pd.concat([new_hourly_weather_data, temp_df])

new_hourly_weather_data.to_csv("Group B/data/weather_data_hour.csv")

filtered_hourly_weather_data = {}
for day_type in new_hourly_weather_data['type_of_day'].unique():
    filtered_hourly_weather_data[f"weather_data_hour_{day_type}"] = new_hourly_weather_data[new_hourly_weather_data['type_of_day'] == day_type]
for day_type, df in filtered_hourly_weather_data.items():
    df.to_csv(f"Group B/data/{day_type}.csv", index=False)
