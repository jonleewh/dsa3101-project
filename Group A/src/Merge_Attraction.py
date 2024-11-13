import pandas as pd
from datetime import datetime

RIDE = '' # key in your ride name

attraction_files = [
    f'{RIDE}_jan.csv', f'{RIDE}_feb.csv', f'{RIDE}_mar.csv', 
    f'{RIDE}_apr.csv', f'{RIDE}_may.csv', f'{RIDE}_jun.csv', 
    f'{RIDE}_jul.csv', f'{RIDE}_aug.csv', f'{RIDE}_sep.csv', 
    f'{RIDE}_oct.csv', f'{RIDE}_nov.csv', f'{RIDE}_dec.csv'
]

attraction_df = pd.concat([pd.read_csv(file) for file in attraction_files])

attraction_df['Date/Time'] = pd.to_datetime(attraction_df['Date/Time'])

attraction_df = attraction_df[attraction_df['Date/Time'].dt.year == 2023]

attraction_df['Wait Time'] = pd.to_numeric(attraction_df['Wait Time'], errors='coerce')

attraction_df = attraction_df[attraction_df['Wait Time'] != 0]

attraction_df['Time'] = attraction_df['Date/Time'].dt.time

attraction_df_filtered = attraction_df[(attraction_df['Time'] >= pd.to_datetime('10:00:00').time()) & 
                                       (attraction_df['Time'] <= pd.to_datetime('19:00:00').time())]

attraction_df_filtered = attraction_df_filtered.drop(columns=['Time'])

school_holidays_df = pd.read_csv('../data/SchoolHolidaysfor2023.csv')
public_holidays_df = pd.read_csv('../data/PublicHolidaysfor2023.csv')

school_holidays_df['date'] = pd.to_datetime(school_holidays_df['date'])
public_holidays_df['date'] = pd.to_datetime(public_holidays_df['date'])

school_holidays_set = set(school_holidays_df['date'].dt.date)
public_holidays_set = set(public_holidays_df['date'].dt.date)

def classify_day(date):
    date_only = date.date()

    if date_only in public_holidays_set:
        return 3
    elif date.weekday() >= 5:
        return 2
    else:
        if date_only in school_holidays_set:
            return 1
        else:
            return 0

attraction_df_filtered['DayType'] = attraction_df_filtered['Date/Time'].apply(classify_day)

output_file_path = f'{RIDE}_cleaned.csv'
attraction_df_filtered.to_csv(output_file_path, index=False)
