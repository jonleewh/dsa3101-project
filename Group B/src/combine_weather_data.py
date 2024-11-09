import pandas as pd

weather_data = pd.read_csv("Group B/data/weather_data.csv")

temperature_data = pd.DataFrame(weather_data[["Year", "Month", "Day", "Mean Temperature (°C)"]])
monthly_avg = temperature_data.groupby(['Month'])['Mean Temperature (°C)'].mean()

print(monthly_avg)
