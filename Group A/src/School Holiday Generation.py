import csv
from datetime import datetime, timedelta
import calendar
import os

holiday_ranges = [
    ('2023-03-11', '2023-03-19'),  # Sat 11 Mar to Sun 19 Mar
    ('2023-05-27', '2023-06-25'),  # Sat 27 May to Sun 25 Jun
    ('2023-09-02', '2023-09-10'),  # Sat 2 Sep to Sun 10 Sep
    ('2023-11-18', '2023-12-31'),  # Sat 18 Nov to Sun 31 Dec
]

individual_dates = [
    '2023-07-03',  # Mon 3 Jul
    '2023-08-10',  # Thu 10 Aug
    '2023-09-01',  # Fri 1 Sep
    '2023-10-06',  # Fri 6 Oct
]

def generate_dates(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=1)
    current_date = start
    dates = []
    while current_date <= end:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += delta
    return dates

data = []

for start_date, end_date in holiday_ranges:
    dates_in_range = generate_dates(start_date, end_date)
    for date in dates_in_range:
        day_of_week = calendar.day_name[datetime.strptime(date, '%Y-%m-%d').weekday()]
        data.append((date, day_of_week))

for date in individual_dates:
    day_of_week = calendar.day_name[datetime.strptime(date, '%Y-%m-%d').weekday()]
    data.append((date, day_of_week))

file_path = '../data/SchoolHolidaysfor2023.csv'

os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Date', 'Day'])
    
    writer.writerows(data)


