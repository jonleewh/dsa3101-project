import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Load and preprocess data (already done in previous steps)
data = pd.read_csv('Weather data.csv')
data.replace({'-': np.nan}, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.median(), inplace=True)
data.drop(columns=['Station'], inplace=True)  # Drop irrelevant columns

# Convert to numeric and handle missing values
data['Year'] = data['Year'].astype(int)
data['Month'] = data['Month'].astype(int)
data['Day'] = data['Day'].astype(int)

# Satisfaction Score Calculation based on weather conditions
def calculate_satisfaction(row):
    satisfaction_score = 5
    # Categorize the day (sunny, rainy, or shower)
    if row['Highest 60 min Rainfall (mm)'] > row['Highest 30 min Rainfall (mm)']:
        day_type = 'rainy'
    elif row['Highest 30 min Rainfall (mm)'] == row['Highest 60 min Rainfall (mm)']:
        day_type = 'shower'
    else:
        day_type = 'sunny'
    
    # Sunny days: Cooler days are better for satisfaction
    if day_type == 'sunny':
        satisfaction_score -= row['Mean Temperature (°C)'] * 0.6

    # Rainy days: Rainy days have lower satisfaction, more rain means worse satisfaction
    elif day_type == 'rainy':
        satisfaction_score -= row['Daily Rainfall Total (mm)'] * 0.6
        satisfaction_score -= row['Highest 30 min Rainfall (mm)'] * 0.3

    # Shower days: Mildly affected by rain, and temperature matters too
    elif day_type == 'shower':
        satisfaction_score -= row['Highest 30 min Rainfall (mm)'] * 0.4
        satisfaction_score += row['Mean Temperature (°C)'] * 0.2

    # Wind Speed: If max wind exceeds thresholds, it reduces satisfaction
    if row['Max Wind Speed (km/h)'] > 60:
        satisfaction_score -= 5  # Very high winds, all outdoor rides closed
    elif row['Max Wind Speed (km/h)'] > 40:
        satisfaction_score -= 3  # High winds, outdoor rides affected

    # Clamp the satisfaction score to a reasonable range (0-100)
    satisfaction_score = max(min(satisfaction_score, 100), 0)

    return satisfaction_score

# Apply the function to the dataframe
data['Satisfaction Score'] = data.apply(calculate_satisfaction, axis=1)

# Dynamic Pricing Calculation
def dynamic_pricing(satisfaction_score):
    """
    Adjust the ticket price based on the satisfaction score.
    Higher satisfaction: Higher price.
    Lower satisfaction: Lower price.
    The price is centered around 62 SGD.
    """
    base_price = 62  # Base ticket price in SGD (fixed)
    
    # Increase the price adjustment factor to ensure a broader price range
    price_adjustment_factor = (satisfaction_score - 6.5) * 0.5  # 0.5 SGD per point of satisfaction    
    final_price = base_price + price_adjustment_factor
    
    # Ensure that the price doesn't go below 30 SGD or above a reasonable upper limit (e.g., 100 SGD)
    final_price = max(min(final_price, 100), 30)
    
    return final_price


# Simulating the dynamic pricing system for 100 days
days_to_simulate = 100
simulated_revenue = []
simulated_satisfaction = []
simulated_prices = []

# Simulating guest behavior: Assuming guests are more likely to visit if the price is low and satisfaction is high
def simulate_guests(price, satisfaction_score):
    base_visits = 1000  # Assume a base number of visitors for a day
    price_sensitivity = 0.5  # Price sensitivity factor
    satisfaction_sensitivity = 0.1  # Satisfaction sensitivity factor

    # Simulate the number of visitors based on price and satisfaction
    adjusted_visits = base_visits * (1 - price_sensitivity * (price - 62)) * (1 + satisfaction_sensitivity * (satisfaction_score - 50))
    return max(adjusted_visits, 0)

# Simulating for each day over 6 months
for i in range(days_to_simulate):
    # Select a random day's data (simulate random weather)
    day_data = data.iloc[random.randint(0, len(data) - 1)]
    
    # Get satisfaction score and dynamic price for the day
    satisfaction_score = day_data['Satisfaction Score']
    price = dynamic_pricing(satisfaction_score)
    
    # Simulate guest behavior (number of visitors)
    num_visitors = simulate_guests(price, satisfaction_score)
    
    # Calculate daily revenue
    if price >= 62:
        revenue = (price - 62) * num_visitors
    else:
        revenue = 62
    
    # Record the results for analysis
    simulated_revenue.append(revenue)
    simulated_satisfaction.append(satisfaction_score)
    simulated_prices.append(price)

# Create a DataFrame with the simulation results
simulation_results = pd.DataFrame({
    'Day': range(1, days_to_simulate + 1),
    'Satisfaction Score': simulated_satisfaction,
    'Dynamic Price (SGD)': simulated_prices,
    'Possible Extra Revenue (SGD)': simulated_revenue
})

# Step 6: Analyze Results
print(simulation_results.head())

# Plot the Results
plt.figure(figsize=(12, 6))

# Plot Revenue vs Satisfaction
plt.subplot(2, 1, 1)
plt.plot(simulation_results['Day'], simulation_results['Possible Extra Revenue (SGD)'], label='Possible Extra Revenue (SGD)', color='blue')
plt.xlabel('Day')
plt.ylabel('Possible Extra Revenue (SGD)')
plt.title('Revenue Over 6 Months')
plt.legend()

# Plot Dynamic Prices vs Satisfaction
plt.subplot(2, 1, 2)
plt.plot(simulation_results['Day'], simulation_results['Dynamic Price (SGD)'], label='Dynamic Price (SGD)', color='red')
plt.xlabel('Day')
plt.ylabel('Price (SGD)')
plt.title('Dynamic Pricing Over 6 Months')
plt.legend()

plt.tight_layout()
plt.show()

