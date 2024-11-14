import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

random.seed(3101)
np.random.seed(3101)

weather_df = pd.read_csv('../data/2023_daily_weather_with_wait_times_and_conditions.csv')

weather_df['Maximum Temperature (°C)'].fillna(weather_df['Maximum Temperature (°C)'].median(), inplace=True)
weather_df['Daily Rainfall Total (mm)'].fillna(weather_df['Daily Rainfall Total (mm)'].median(), inplace=True)
weather_df['Public Holiday'].fillna(0, inplace=True)

base_price = 62

avg_wait_time = weather_df['Wait Time'].mean()
avg_max_temp = weather_df['Maximum Temperature (°C)'].mean()
avg_rainfall = weather_df['Daily Rainfall Total (mm)'].mean()

heavy_rain_threshold = 35
high_temp_threshold = avg_max_temp

def scale_column(df, col_name):
    median = df[col_name].median()
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    shifted_values = df[col_name] - median
    scale_factor = 20 / (max_val - min_val)
    scaled_values = shifted_values * scale_factor
    return scaled_values

weather_df['Scaled Maximum Temperature'] = scale_column(weather_df, 'Maximum Temperature (°C)')

X = weather_df[['Maximum Temperature (°C)', 'Daily Rainfall Total (mm)', 'Public Holiday']]
y = weather_df['Wait Time']

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_

print(f"Intercept: {intercept:.2f}")
print(f"Coefficients: Max Temp: {coefficients[0]:.2f}, Rainfall: {coefficients[1]:.2f}, Public Holiday: {coefficients[2]:.2f}")

num_days = 100
num_public_holidays = 10

random_max_temps = np.random.normal(loc=avg_max_temp, scale=5, size=num_days)
random_rainfall = np.random.normal(loc=avg_rainfall, scale=5, size=num_days)
public_holiday_flags = np.array([1]*num_public_holidays + [0]*(num_days - num_public_holidays))

simulated_data = pd.DataFrame({
    'Date': pd.date_range(start="2023-11-01", periods=num_days, freq='D'),
    'Maximum Temperature (°C)': random_max_temps,
    'Daily Rainfall Total (mm)': random_rainfall,
    'Public Holiday': public_holiday_flags
})

simulated_data['Predicted Wait Time'] = model.predict(simulated_data[['Maximum Temperature (°C)', 'Daily Rainfall Total (mm)', 'Public Holiday']])

simulated_data['Scaled Maximum Temperature'] = scale_column(simulated_data, 'Maximum Temperature (°C)')

def scale_predicted_wait_time(df, col_name, median_wait_time):
    median_val = df[col_name].median()
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    shifted_values = df[col_name] - median_wait_time
    scale_factor = 20 / (max_val - min_val)
    scaled_values = shifted_values * scale_factor
    return scaled_values

simulated_data['Scaled Predicted Wait Time'] = scale_predicted_wait_time(simulated_data, 'Predicted Wait Time', median_wait_time=avg_wait_time)

def scale_satisfaction_score(df, col_name):
    median_val = df[col_name].median()
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    
    shifted_values = df[col_name] - median_val

    scale_factor = 10 / (max_val - min_val)  
    scaled_values = shifted_values * scale_factor
    
    return scaled_values

def calculate_satisfaction(row):
    satisfaction_score = 100
    satisfaction_score -= row['Scaled Predicted Wait Time']
    satisfaction_score -= row['Scaled Maximum Temperature']
    
    if row['Daily Rainfall Total (mm)'] > heavy_rain_threshold:
        satisfaction_score -= 2.5
    
    return satisfaction_score

def calculate_dynamic_price(satisfaction_score):
    price = 62 + satisfaction_score
    rounded_price = np.ceil(price * 2) / 2
    return rounded_price

simulated_data['Satisfaction Score'] = simulated_data.apply(calculate_satisfaction, axis=1)
simulated_data['Scaled Satisfaction Score'] = scale_satisfaction_score(simulated_data, 'Satisfaction Score')
simulated_data['Dynamic Price'] = simulated_data['Scaled Satisfaction Score'].apply(calculate_dynamic_price)

simulated_revenue_base = simulated_data.shape[0] * base_price /100
simulated_revenue_dynamic = simulated_data['Dynamic Price'].sum() /100

print(f"Simulated Revenue per Person with Base Price: ${simulated_revenue_base:.2f}")
print(f"Simulated Revenue per Person with Dynamic Pricing: ${simulated_revenue_dynamic:.2f}")

simulated_revenue_increase = (simulated_revenue_dynamic - simulated_revenue_base) / simulated_revenue_base * 100
print(f"Simulated Revenue Increase with Dynamic Pricing: {simulated_revenue_increase:.2f}%")

min_price = int(simulated_data['Dynamic Price'].min())
max_price = int(simulated_data['Dynamic Price'].max())

price_bins = np.arange(min_price, max_price + 2, 1)

price_hist, bin_edges = np.histogram(simulated_data['Dynamic Price'], bins=price_bins)

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], price_hist, width=1, edgecolor='black', align='edge', color='skyblue')
plt.axvline(x=62, color='red', linestyle='--', label='Fixed Price (62$)')
plt.text(62 + 0.2, max(price_hist) * 0.8, 'Fixed Price', color='red')
plt.xlabel('Price ($)')
plt.ylabel('Number of Days')
plt.title('Distribution of Dynamic Price Over 100 Days')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

percentiles = [0, 25, 50, 75, 100]
percentile_values = np.percentile(simulated_data['Dynamic Price'], percentiles)

sorted_prices = np.sort(simulated_data['Dynamic Price'])
cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices) * 100

plt.figure(figsize=(10, 6))
plt.plot(cdf, sorted_prices, marker='.', color='blue', label='Cumulative Distribution')

percentile_colors = ['red', 'green', 'orange', 'purple', 'brown']

for i, val in enumerate(percentile_values):
    plt.axhline(val, color=percentile_colors[i], linestyle='--', label=f'{percentiles[i]}th Percentile')

plt.axhline(62, color='black', linestyle='-', label="Fixed Price ($62)")
plt.title('Cumulative Distribution of Dynamic Price')
plt.ylabel('Dynamic Price ($)')
plt.xlabel('Cumulative Percentage of Days (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')
plt.show()


