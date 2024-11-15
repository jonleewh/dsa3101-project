import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
import glob
import sqlite3

np.random.seed(2024)
os.chdir('Group B/src')

################################################################
## Read in Nodes & Weather Data + Data Manipulation using SQL ##
################################################################
nodes = pd.read_csv('../data/theme_park_nodes.csv')
nodes.fillna(0, inplace=True)
nodes.drop(["actual_wait_time", "expected_wait_time", "crowd_level"], axis=1, inplace = True)
columns_to_convert = ["duration", "cleanliness", "affordability", "capacity", "staff"]
for column in columns_to_convert:
    nodes[column] = pd.to_numeric(nodes[column], errors='coerce')

weather_data_hour_1 = pd.read_csv('../data/weather_data_hour_1.csv')
weather_data_hour_2 = pd.read_csv('../data/weather_data_hour_2.csv')
weather_data_hour_3 = pd.read_csv('../data/weather_data_hour_3.csv')
weather_data_hour_4 = pd.read_csv('../data/weather_data_hour_4.csv')
weather_data_hour_5 = pd.read_csv('../data/weather_data_hour_5.csv')

weather_data_hour_1["type_of_day"] = weather_data_hour_1["type_of_day"].astype(int)
weather_data_hour_2["type_of_day"] = weather_data_hour_2["type_of_day"].astype(int)
weather_data_hour_3["type_of_day"] = weather_data_hour_3["type_of_day"].astype(int)
weather_data_hour_4["type_of_day"] = weather_data_hour_4["type_of_day"].astype(int)
weather_data_hour_5["type_of_day"] = weather_data_hour_5["type_of_day"].astype(int)

weather_data_hour_1['datetime'] = pd.to_datetime(weather_data_hour_1['datetime'])
weather_data_hour_2['datetime'] = pd.to_datetime(weather_data_hour_2['datetime'])
weather_data_hour_3['datetime'] = pd.to_datetime(weather_data_hour_3['datetime'])
weather_data_hour_4['datetime'] = pd.to_datetime(weather_data_hour_4['datetime'])
weather_data_hour_5['datetime'] = pd.to_datetime(weather_data_hour_5['datetime'])

# to reduce the computational cost, we consider every hour
weather_data_hour_1 = weather_data_hour_1[weather_data_hour_1['datetime'].dt.minute % 60 == 0]
weather_data_hour_2 = weather_data_hour_2[weather_data_hour_2['datetime'].dt.minute % 60 == 0]
weather_data_hour_3 = weather_data_hour_3[weather_data_hour_3['datetime'].dt.minute % 60 == 0]
weather_data_hour_4 = weather_data_hour_4[weather_data_hour_4['datetime'].dt.minute % 60 == 0]
weather_data_hour_5 = weather_data_hour_5[weather_data_hour_5['datetime'].dt.minute % 60 == 0]

####################################
## Read in Simulation Output Data ##
####################################

# Get a list of all CSV files in a directory
simulation_output_both_holiday = glob.glob('../data/simulation_output_both_holiday/*.csv')
simulation_output_public_holiday = glob.glob('../data/simulation_output_public_holiday/*.csv')
simulation_output_school_holiday = glob.glob('../data/simulation_output_school_holiday/*.csv')
simulation_output_weekday_non_holiday = glob.glob('../data/simulation_output_weekday_non_holiday/*.csv')
simulation_output_weekend_non_holiday = glob.glob('../data/simulation_output_weekend_non_holiday/*.csv')

# Create an empty dataframe to store the combined data
combined_df_both_holiday = pd.DataFrame()
combined_df_public_holiday = pd.DataFrame()
combined_df_school_holiday = pd.DataFrame()
combined_df_weekday_non_holiday = pd.DataFrame()
combined_df_weekend_non_holiday = pd.DataFrame()

# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in simulation_output_both_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_both_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.minute == 0]
    df['type_of_day'] = 1
    combined_df_both_holiday = pd.concat([combined_df_both_holiday, df])

for csv_file in simulation_output_public_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_public_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.minute == 0]
    df['type_of_day'] = 2
    combined_df_public_holiday = pd.concat([combined_df_public_holiday, df])

for csv_file in simulation_output_school_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_school_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.minute == 0]
    df['type_of_day'] = 3
    combined_df_school_holiday = pd.concat([combined_df_school_holiday, df])

for csv_file in simulation_output_weekday_non_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_weekday_non_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.minute == 0]
    df['type_of_day'] = 4
    combined_df_weekday_non_holiday = pd.concat([combined_df_weekday_non_holiday, df])

for csv_file in simulation_output_weekend_non_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_weekend_non_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.minute == 0]
    df['type_of_day'] = 5
    combined_df_weekend_non_holiday = pd.concat([combined_df_weekend_non_holiday, df])

combined_df_both_holiday = pd.merge(combined_df_both_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_public_holiday = pd.merge(combined_df_public_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_school_holiday = pd.merge(combined_df_school_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_weekday_non_holiday = pd.merge(combined_df_weekday_non_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_weekend_non_holiday = pd.merge(combined_df_weekend_non_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])

combined_df_both_holiday = pd.merge(weather_data_hour_1, combined_df_both_holiday, on = 'type_of_day', how='left').drop(columns = ["datetime", "Day of the Week", "School Holiday", "Public Holiday", "Weekend"])
combined_df_public_holiday = pd.merge(weather_data_hour_2, combined_df_public_holiday, on = 'type_of_day', how='left').drop(columns = ["datetime", "Day of the Week", "School Holiday", "Public Holiday", "Weekend"])
combined_df_school_holiday = pd.merge(weather_data_hour_3, combined_df_school_holiday, on = 'type_of_day', how='left').drop(columns = ["datetime", "Day of the Week", "School Holiday", "Public Holiday", "Weekend"])
combined_df_weekday_non_holiday = pd.merge(weather_data_hour_4, combined_df_weekday_non_holiday, on = 'type_of_day', how='left').drop(columns = ["datetime", "Day of the Week", "School Holiday", "Public Holiday", "Weekend"])
combined_df_weekend_non_holiday = pd.merge(weather_data_hour_5, combined_df_weekend_non_holiday, on = 'type_of_day', how='left').drop(columns = ["datetime", "Day of the Week", "School Holiday", "Public Holiday", "Weekend"])

combined_dfs = [combined_df_both_holiday, combined_df_public_holiday, combined_df_school_holiday,
               combined_df_weekday_non_holiday, combined_df_weekend_non_holiday]


######################################
## Calculating Actual Waiting Times ##
######################################

# normalise data
def waiting_time(type, duration, fast_pass_crowd_level, regular_crowd_level, popularity,
                 capacity, rain, staff, outdoor): # calculate expected waiting time for a ride.
    """
    Parameters:
    # type: type of ride
    # duration: duration of the ride (in minutes)
    # crowd_level: Number of people currently in the queue.
    # capacity: Capacity of the attraction (people per cycle).
    # staff: Number of staff available (for F&B and retail).
    # popularity: Popularity of the attraction.
    # outdoor: Whether the attraction is outdoor (influenced by weather).
    """
    capacity = 1 if capacity == 0 else capacity # to ensure capacity is not zero, avoid division by 0
    
    if type == 'Food Cart': # assumed to be outdoors
        if rain == 1:
            waiting_time = 0 # because there will be no customers who will be out buying F&B from the food carts
        else:
            waiting_time = round(max(0,
                                     (popularity / 100) * (fast_pass_crowd_level + regular_crowd_level) / capacity * duration
                                     + 10 / staff
                                     + np.random.normal(-1, 5)), 0)
    
    elif type == 'Retail' or type == "Dining Outlet": # assumed to be indoors
        waiting_time = round(max(0,
                                 (popularity / 100) * (fast_pass_crowd_level + regular_crowd_level) / capacity * duration
                                 + 10 / staff
                                 + 15 * rain # if it is raining, we assume an additional 15 minutes of waiting time added
                                 + np.random.normal(-1, 5)), 0) # we introduce noise into the data
    
    else: # for rides, we assume that there will be sufficient staff to operate the rides
        if rain == 1 and outdoor == 1:
            waiting_time = 0
        else:
            waiting_time = round(max(0,
                                     (popularity / 100) * (fast_pass_crowd_level + regular_crowd_level) / capacity * duration
                                     + 15 * rain # if it is raining, we assume an additional 15 minutes of waiting time added
                                     + np.random.normal(-1, 5)), 0) # we introduce noise into the data
    
    return waiting_time

combined_data = pd.DataFrame()

wait_time_key_X_features = ['duration', 'fast_pass_crowd_level', 'regular_crowd_level',
                            'popularity', 'capacity', 'rain', 'staff', 'outdoor']
wait_time_X_combined = pd.DataFrame()
wait_time_y_synthetic = pd.DataFrame() # generate wait times based on X

for combined_df in combined_dfs:
    combined_data = pd.concat([combined_data, combined_df])
    wait_time_X = combined_df[['attraction', 'type_of_day', 'type', 'duration',
                               'fast_pass_crowd_level', 'regular_crowd_level',
                               'popularity', 'capacity', 'rain', 'staff', 'outdoor']]
    wait_time_X_combined = pd.concat([wait_time_X_combined, wait_time_X])

for entry in wait_time_X_combined.itertuples():
    actual_waiting_time = waiting_time(entry.type, entry.duration, entry.fast_pass_crowd_level, entry.regular_crowd_level,
                                       entry.popularity, entry.capacity, entry.rain, entry.staff, entry.outdoor)
    wait_time_y_synthetic = pd.concat([wait_time_y_synthetic,
                                       pd.DataFrame({"waiting_time": [actual_waiting_time]})], ignore_index=True)

combined_data = combined_data.reset_index(drop=True)  # Reset index to ensure alignment
wait_time_y_synthetic = wait_time_y_synthetic.reset_index(drop = True)
combined_data['waiting_time'] = wait_time_y_synthetic['waiting_time']

# Train a Random Forest regressor to evaluate importance of each feature
wait_time_rf_model = RandomForestRegressor(n_estimators = 50)
wait_time_rf_model.fit(wait_time_X_combined[wait_time_key_X_features], wait_time_y_synthetic)
wait_time_importances = wait_time_rf_model.feature_importances_

# Plot a graph of the importance of each variable
wait_time_feature_importance = pd.DataFrame({'Feature': wait_time_key_X_features, 'Importance': wait_time_importances}).sort_values(by='Importance', ascending = False) # Sort the DataFrame by importance for a better plot
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data = wait_time_feature_importance, palette='viridis')
plt.suptitle('Importance of each feature in determining Waiting Time', fontsize = 18)
plt.title("determined using Random Forest Model", fontsize = 12)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Apply weights based on feature importances
wait_time_X_importance = wait_time_X_combined[wait_time_key_X_features] * wait_time_importances # Each column in X is scaled by its corresponding importance from the random forest model

# Train the linear regression model with the coefficients accounting for the importance
wait_time_ml_model = LinearRegression()
wait_time_ml_model.fit(wait_time_X_importance, wait_time_y_synthetic)


####################################
## Calculating Satisfaction Score ##
####################################
# Satisfaction score refers to the score for a single NODE, not the visitors.
def satisfaction_score(fast_pass_crowd_level, regular_crowd_level, affordability,
                       cleanliness, capacity, waiting_time, temperature, rain):
    # we input a first guess of the coefficients here first
    # score is between 0 and 100, but we assume that there are no extremes.
    # it is impossible for a customer to be 100% satisfied or 100% dissatisfied.
    satisfaction_score = max(5, min(95, (50 # average satisfaction score
                                            - 5 * (fast_pass_crowd_level + regular_crowd_level) # higher crowd level results in lower satisfaction score
                                            + 4 * affordability # more affordable items results in higher satisfaction score
                                            + 2 * cleanliness # better cleanliness results in higher satisfaction score
                                            + 2 * capacity # higher capacity results in higher satisfaction score
                                            - 5 * waiting_time # longer wait time results in lower satisfaction score
                                            - 5 * temperature # higher temperatures results in lower satisfaction score
                                            - 5 * rain # rain results in lower satisfaction score
                                            )))
    return satisfaction_score

satisfaction_score_X_key_features = ['fast_pass_crowd_level', 'regular_crowd_level',
                                     'affordability', 'cleanliness', 'capacity',
                                     "waiting_time", "Temp", 'rain']
satisfaction_score_X_combined = pd.DataFrame()
satisfaction_score_y_synthetic = pd.DataFrame() # generate satisfaction scores based on X

combined_data["popularity"] = (combined_data["popularity"] - combined_data["popularity"].min()) / (combined_data["popularity"].max() - combined_data["popularity"].min())
combined_data["staff"] = (combined_data["staff"] - combined_data["staff"].min()) / (combined_data["staff"].max() - combined_data["staff"].min())
combined_data["affordability"] = (combined_data["affordability"] - combined_data["affordability"].min()) / (combined_data["affordability"].max() - combined_data["affordability"].min())
combined_data["cleanliness"] = (combined_data["cleanliness"] - combined_data["cleanliness"].min()) / (combined_data["cleanliness"].max() - combined_data["cleanliness"].min())
combined_data["Temp"] = combined_data["Temp"] - 30 # for temperature, we normalise the values where 30 degrees is zero, anything above has positive value, anything below has negative value

# Normalisation & Standardisation of Data
for combined_df in combined_dfs:
    satisfaction_score_X = combined_data[['attraction', 'type_of_day', 'type', 
                                          'fast_pass_crowd_level', 'regular_crowd_level',
                                          'affordability', 'cleanliness', 'capacity',
                                          "waiting_time", "Temp", 'rain']]
    satisfaction_score_X_combined = pd.concat([satisfaction_score_X_combined, satisfaction_score_X])

for entry in satisfaction_score_X_combined.itertuples():
    actual_satisfaction_score = satisfaction_score(entry.fast_pass_crowd_level, entry.regular_crowd_level, entry.affordability,
                                                   entry.cleanliness, entry.capacity, entry.waiting_time, entry.Temp, entry.rain)
    satisfaction_score_y_synthetic = pd.concat([satisfaction_score_y_synthetic,
                                                pd.DataFrame({"satisfaction_score": [actual_satisfaction_score]})], ignore_index=True)

# Train a Random Forest regressor to evaluate importance of each feature
satisfaction_score_rf_model = RandomForestRegressor(n_estimators = 10)
satisfaction_score_rf_model.fit(satisfaction_score_X_combined[satisfaction_score_X_key_features],
                                satisfaction_score_y_synthetic)
satisfaction_score_importances = satisfaction_score_rf_model.feature_importances_

# Plot a graph of the importance of each variable
satisfaction_score_feature_importance = pd.DataFrame({'Feature': satisfaction_score_X_key_features,
                                                      'Importance': satisfaction_score_importances}
                                                     ).sort_values(by='Importance', ascending = False) # Sort the DataFrame by importance for a better plot

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data = satisfaction_score_feature_importance, palette='viridis')
plt.suptitle('Importance of each feature in determining Satisfaction Score', fontsize = 18)
plt.title("determined using Random Forest Model", fontsize = 12)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Apply weights based on feature importances
satisfaction_score_X_importance = satisfaction_score_X_combined[satisfaction_score_X_key_features] * satisfaction_score_importances # Each column in X is scaled by its corresponding importance from the random forest model

# Train the linear regression model with the coefficients accounting for the importance
satisfaction_score_ml_model = LinearRegression()
satisfaction_score_ml_model.fit(satisfaction_score_X_importance, satisfaction_score_y_synthetic)

########################
## Seasonal Variation ##
########################
percentage_changes = np.linspace(-0.75, 0.75, 100)

# Get the mean values of each feature to use as the baseline
mean_regular_crowd_level = satisfaction_score_X_combined['regular_crowd_level'].mean()
mean_fast_pass_crowd_level = satisfaction_score_X_combined['fast_pass_crowd_level'].mean()
mean_affordability = satisfaction_score_X_combined['affordability'].mean()
mean_cleanliness = satisfaction_score_X_combined['cleanliness'].mean()
mean_capacity = satisfaction_score_X_combined['capacity'].mean()
mean_waiting_time = satisfaction_score_X_combined['waiting_time'].mean()
mean_temperature = satisfaction_score_X_combined['Temp'].mean() + 30
rain_probability = satisfaction_score_X_combined['rain'].mean()

# Initialize lists to store satisfaction scores for each feature
satisfaction_scores_crowd = []
satisfaction_scores_affordability = []
satisfaction_scores_cleanliness = []
satisfaction_scores_capacity = []

# Calculate the satisfaction score for each percentage change in the feature
for change in percentage_changes:
    
    # Vary crowd_level while keeping others constant
    regular_crowd_level = mean_regular_crowd_level * (1 + change)
    fast_pass_crowd_level = mean_fast_pass_crowd_level * (1 + change)
    score_crowd = satisfaction_score(regular_crowd_level, fast_pass_crowd_level, mean_affordability,
                                     mean_cleanliness, mean_capacity, mean_waiting_time, mean_temperature, rain_probability)
    satisfaction_scores_crowd.append(score_crowd)

    # Vary cleanliness while keeping others constant
    cleanliness = mean_cleanliness * (1 + change)
    score_cleanliness = satisfaction_score(mean_regular_crowd_level, mean_fast_pass_crowd_level, mean_affordability,
                                           cleanliness, mean_capacity, mean_waiting_time, mean_temperature, rain_probability)
    satisfaction_scores_cleanliness.append(score_cleanliness)
    
    # Vary capacity while keeping others constant
    capacity = mean_capacity * (1 + change)
    score_capacity = satisfaction_score(mean_regular_crowd_level, mean_fast_pass_crowd_level, mean_affordability,
                                        mean_cleanliness, capacity, mean_waiting_time, mean_temperature, rain_probability)
    satisfaction_scores_capacity.append(score_capacity)

# Plotting the results
plt.figure(figsize=(15, 5))

# Plot for crowd_level
plt.subplot(1, 3, 1)
# for all the graphs, have multiple lines that shows the percentage changes for each type of day
plt.plot(percentage_changes * 100, satisfaction_scores_crowd, label='Crowd Level', color='blue')
plt.axhline(y = satisfaction_score(mean_regular_crowd_level, mean_fast_pass_crowd_level, mean_affordability,
                                   mean_cleanliness, mean_capacity, mean_waiting_time, mean_temperature, rain_probability),
            color = 'gray', linestyle = '--', label = 'Baseline Score')
plt.xlabel('% Change in Crowd Level')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Crowd Level on Satisfaction Score')
plt.legend()
plt.ylim(0, 100)

# Plot for cleanliness
plt.subplot(1, 3, 2)
plt.plot(percentage_changes * 100, satisfaction_scores_cleanliness, label='Cleanliness', color='red')
plt.axhline(y = satisfaction_score(mean_regular_crowd_level, mean_fast_pass_crowd_level, mean_affordability,
                                   mean_cleanliness, mean_capacity, mean_waiting_time, mean_temperature, rain_probability),
            color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Cleanliness')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Cleanliness on Satisfaction Score')
plt.legend()
plt.ylim(0, 100)

# Plot for capacity
plt.subplot(1, 3, 3)
plt.plot(percentage_changes * 100, satisfaction_scores_capacity, label='Capacity', color='purple')
plt.axhline(y = satisfaction_score(mean_regular_crowd_level, mean_fast_pass_crowd_level, mean_affordability,
                                   mean_cleanliness, mean_capacity, mean_waiting_time, mean_temperature, rain_probability),
            color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Capacity')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Capacity on Satisfaction Score')
plt.legend()
plt.ylim(0, 100)

plt.tight_layout()
plt.show()
