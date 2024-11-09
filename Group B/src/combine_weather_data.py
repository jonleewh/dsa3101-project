import pandas as pd

# Daily Weather Data
daily_weather_data = pd.read_csv("Group B/data/daily_weather_data.csv")

temperature_data = pd.DataFrame(daily_weather_data[["Year", "Month", "Day", "Mean Temperature (°C)"]])
monthly_avg = temperature_data.groupby(['Month'])['Mean Temperature (°C)'].mean()

#print(monthly_avg)

# Hourly Weather Data
#hourly_weather_data = pd.read_csv("Group B/data/hourly_weather_data.csv")

