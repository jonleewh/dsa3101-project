import pandas as pd
from datetime import datetime

weather_files = [
    '../data/DAILYDATA_S60_202301.csv', '../data/DAILYDATA_S60_202302.csv', 
    '../data/DAILYDATA_S60_202303.csv', '../data/DAILYDATA_S60_202304.csv',
    '../data/DAILYDATA_S60_202305.csv', '../data/DAILYDATA_S60_202306.csv',
    '../data/DAILYDATA_S60_202307.csv', '../data/DAILYDATA_S60_202308.csv',
    '../data/DAILYDATA_S60_202309.csv', '../data/DAILYDATA_S60_202310.csv',
    '../data/DAILYDATA_S60_202311.csv', '../data/DAILYDATA_S60_202312.csv',
]

weather_df = pd.concat([pd.read_csv(file) for file in weather_files])

# weather_df.to_csv('../data/2023_daily_weather.csv')

holidays_df = pd.read_csv('../data/PublicHolidaysfor2023.csv')

holidays_df['date'] = pd.to_datetime(holidays_df['date'])

public_holidays = set(holidays_df['date'].dt.date)

weather_df['Date'] = pd.to_datetime(weather_df[['Year', 'Month', 'Day']])

weather_df['Public Holiday'] = weather_df['Date'].dt.date.isin(public_holidays).astype(int)

# weather_df.to_csv('../data/2023_daily_weather_with_holidays.csv', index=False)

file_paths = [
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_jan.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_feb.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_mar.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_apr.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_may.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_jun.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_jul.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_aug.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_sep.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_oct.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_nov.csv',
    '../../Group B/data/USS daily waiting time/download_universal-studios-singapore_dec.csv'
]

def calculate_daily_avg_wait_time(file_path):
    df = pd.read_csv(file_path)
    
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    
    df['Date'] = df['Date/Time'].dt.date
    
    daily_avg_wait_time = df.groupby('Date')['Wait Time'].mean().reset_index()
    
    return daily_avg_wait_time

all_avg_wait_times = pd.DataFrame()

for file_path in file_paths:
    daily_avg_wait_time = calculate_daily_avg_wait_time(file_path)
    all_avg_wait_times = pd.concat([all_avg_wait_times, daily_avg_wait_time])

weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.date

weather_df = pd.merge(weather_df, all_avg_wait_times, on='Date', how='left')

# weather_df.to_csv('../data/2023_daily_weather_with_wait_times.csv', index=False)

def categorize_rain(row):
    daily_rainfall = pd.to_numeric(row['Daily Rainfall Total (mm)'], errors='coerce')
    
    if daily_rainfall <= 2.4:
        return 0  # Sunny
    else:
        return 1  # Rainy
    
def categorize_shower(row):
    daily_rainfall = pd.to_numeric(row['Daily Rainfall Total (mm)'], errors='coerce')
    highest_60_min_rainfall = pd.to_numeric(row['Highest 60 min Rainfall (mm)'], errors='coerce')
    
    if daily_rainfall > 2.4 and highest_60_min_rainfall <= 4:
        return 1  # Shower
    else:
        return 0  # Sunny


weather_df['Rain Condition'] = weather_df.apply(categorize_rain, axis=1)
weather_df['Shower Condition'] = weather_df.apply(categorize_shower, axis=1)

weather_df['Day'] = pd.to_datetime(weather_df['Date']).dt.day_name()

weather_df = weather_df.dropna(subset=['Wait Time'])

columns_to_keep = [
    'Date','Day', 'Wait Time','Public Holiday', 'Rain Condition',
    'Shower Condition', 'Daily Rainfall Total (mm)', 
    'Highest 30 min Rainfall (mm)', 'Highest 60 min Rainfall (mm)', 
    'Highest 120 min Rainfall (mm)', 'Mean Temperature (°C)', 
    'Maximum Temperature (°C)', 'Minimum Temperature (°C)'
]

weather_df = weather_df[columns_to_keep]

weather_df = weather_df.drop(16) # manually dropped 2023-01-17 as all entries are 7, clearly a data errer
weather_df = weather_df.drop(352) # manually dropped 2023-12-19 as only one entry at 19:55, loss of data

# weather_df.to_csv('../data/2023_daily_weather_with_wait_times_and_conditions.csv', index=False)
