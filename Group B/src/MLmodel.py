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
weather_data_hour_1 = weather_data_hour_1[weather_data_hour_1['datetime'].dt.minute % 30 == 0]
weather_data_hour_2['datetime'] = pd.to_datetime(weather_data_hour_2['datetime'])
weather_data_hour_2 = weather_data_hour_2[weather_data_hour_2['datetime'].dt.minute % 30 == 0]
weather_data_hour_3['datetime'] = pd.to_datetime(weather_data_hour_3['datetime'])
weather_data_hour_3 = weather_data_hour_3[weather_data_hour_3['datetime'].dt.minute % 30 == 0]
weather_data_hour_4['datetime'] = pd.to_datetime(weather_data_hour_4['datetime'])
weather_data_hour_4 = weather_data_hour_4[weather_data_hour_4['datetime'].dt.minute % 30 == 0]
weather_data_hour_5['datetime'] = pd.to_datetime(weather_data_hour_5['datetime'])
weather_data_hour_5 = weather_data_hour_5[weather_data_hour_5['datetime'].dt.minute % 30 == 0]


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
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time
    df['type_of_day'] = 1
    combined_df_both_holiday = pd.concat([combined_df_both_holiday, df])

for csv_file in simulation_output_public_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_public_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time
    df['type_of_day'] = 2
    combined_df_public_holiday = pd.concat([combined_df_public_holiday, df])

for csv_file in simulation_output_school_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_school_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time
    df['type_of_day'] = 3
    combined_df_school_holiday = pd.concat([combined_df_school_holiday, df])

for csv_file in simulation_output_weekday_non_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_weekday_non_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time
    df['type_of_day'] = 4
    combined_df_weekday_non_holiday = pd.concat([combined_df_weekday_non_holiday, df])

for csv_file in simulation_output_weekend_non_holiday:
    df = pd.read_csv(csv_file)
    df.drop(columns = ["fast_pass_waiting_time", "regular_waiting_time"], inplace=True)
    df["attraction"] = csv_file.replace("../data/simulation_output_weekend_non_holiday/", "").replace("_df.csv", "")
    df = df[['attraction'] + [col for col in df.columns if col != 'attraction']]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.time
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


"""
for every visitor, we need to know, what attraction types do they prefer and have to visit?
Have a "checklist" of what attraction types to visit
maximise satisfaction and minimise wait time
run through all possible paths that are present from jamie's csv file and change parameters (but cannot change waiting time)
might have to come up with a ML model to change parameters (e.g. defining a range of values for each parameter)
maximise satisfaction overall

Other things we need to do:
optimise guest flow and resource allocation (staff variable)
seasonal variation (after Group A gives us the weather index, etc)
"""

"""
2 ML models
Y1 = average satisfaction score of ALL visitors passing through the ONE node (using a combination of the factors) --> apply Random Forest separately
Y2 = average waiting time of ALL visitors passing through the ONE node (using another combination of the factors) --> apply Random Forest separately
X1, X2, .... = factors

# Factors that affect satisfaction score of a SINGLE node: [MAX]
(focus on what factors we can change)
Need to justify that we used survey data
# crowd level -- we can't change this!!!
menu variety
cleanliness
# accumulated waiting time -- we can't change this!!!
# weather -- we can't change this!!!
# ride quality (only for rides, whether you want to take the ride again) -- we can't change this!!! --> try to link to re-rideability from the survey

Business Suggestions:
*** All visitors benefit from an enhanced satisfaction score, itinerary doesn't really affect the satisfaction score?
*** For allocation of resources, we put the resources from the low demand attraction and transfer them to the high demand attraction
*** Depending on the most important factor
*** How will this impact revenue? e.g. Cleanliness --> increase satisfaction --> revenue likely to increase (explain this in the wiki)

# Factors that affect waiting time of a SINGLE node: [MIN]
(focus on what factors we can change)
ride duration
crowd level -- higher priority for this
staff
weather (indirect factor) --> Need to code which node is indoor/outdoor!

Supervised learning requires an output!
Run ML to determine the most important factor
If any variable is particularly important, how to improve the model?
"""

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

wait_time_key_X_features = ['type', 'duration', 'fast_pass_crowd_level', 'regular_crowd_level',
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
    print(wait_time_y_synthetic)

combined_data = combined_data.reset_index(drop=True)  # Reset index to ensure alignment
wait_time_y_synthetic = wait_time_y_synthetic.reset_index(drop=True)
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
wait_time_X_importance = wait_time_X[wait_time_key_X_features] * wait_time_importances # Each column in X is scaled by its corresponding importance from the random forest model

# Train the linear regression model with the coefficients accounting for the importance
wait_time_ml_model = LinearRegression()
wait_time_ml_model.fit(wait_time_X_importance, wait_time_y_synthetic)
print(wait_time_ml_model)


####################################
## Calculating Satisfaction Score ##
####################################
# Calculate satisfaction/desirability score based on crowd level, wait time
# Satisfaction score refers to the score for a single NODE, not the visitors.
# Suggestion: track how long a guest spends waiting, maximise satisfaction score AND minimise wait time.
# link to popularity
def satisfaction_score(fast_pass_crowd_level, regular_crowd_level, affordability,
                       cleanliness, capacity, waiting_time, temperature, rain):
    # we input a first guess of the coefficients here first
    # score is between 0 and 100, but we assume that there are no extremes.
    # it is impossible for a customer to be 100% satisfied or 100% dissatisfied.
    satisfaction_score_lr = max(5,
                                min(95,
                                    (50 # average satisfaction score
                                     - 5 * (fast_pass_crowd_level + regular_crowd_level) # higher crowd level results in lower satisfaction score
                                     + 4 * affordability # more affordable items results in higher satisfaction score
                                     + 2 * cleanliness # better cleanliness results in higher satisfaction score
                                     + 2 * capacity # higher capacity results in higher satisfaction score
                                     - 5 * waiting_time # longer wait time results in lower satisfaction score
                                     - 5 * temperature # higher temperatures results in lower satisfaction score
                                     - 5 * rain # rain results in lower satisfaction score
                                     )
                                    )
                                )
    return satisfaction_score

satisfaction_score_X_key_features = ['crowd_level', 'affordability', 'cleanliness', 'capacity']
satisfaction_score_X_combined = pd.DataFrame()
satisfaction_score_y_synthetic = pd.DataFrame() # generate satisfaction scores based on X

combined_data["popularity"] = (combined_data["popularity"] - combined_data["popularity"].min()) / (combined_data["popularity"].max() - combined_data["popularity"].min())
combined_data["staff"] = (combined_data["staff"] - combined_data["staff"].min()) / (combined_data["staff"].max() - combined_data["staff"].min())
combined_data["affordability"] = (combined_data["affordability"] - combined_data["affordability"].min()) / (combined_data["affordability"].max() - combined_data["affordability"].min())
combined_data["cleanliness"] = (combined_data["cleanliness"] - combined_data["cleanliness"].min()) / (combined_data["cleanliness"].max() - combined_data["cleanliness"].min())
combined_data["temperature"] = combined_data["temperature"] - 30 # for temperature, we normalise the values where 30 degrees is zero, anything above has positive value, anything below has negative value

# Normalisation & Standardisation of Data
for combined_df in combined_dfs:
    satisfaction_score_X = combined_data[['attraction', 'type_of_day', 'type', 
                                          'fast_pass_crowd_level', 'regular_crowd_level',
                                          'affordability', 'cleanliness', 'capacity',
                                          "waiting_time", "Temp", 'rain']]
    satisfaction_score_X_combined = pd.concat([satisfaction_score_X_combined, satisfaction_score_X])

for entry in satisfaction_score_X_combined.itertuples():
    actual_satisfaction_score = satisfaction_score(entry.fast_pass_crowd_level, entry.regular_crowd_level, entry.affordability,
                                                   entry.cleanliness, entry.capacity, entry.waiting_time, entry.temperature, entry.rain)
    satisfaction_score_y_synthetic = pd.concat([satisfaction_score_y_synthetic,
                                                pd.DataFrame({"satisfaction_score": [actual_satisfaction_score]})], ignore_index=True)
    print(satisfaction_score_y_synthetic)

# Train a Random Forest regressor to evaluate importance of each feature
satisfaction_score_rf_model = RandomForestRegressor(n_estimators = 10)
satisfaction_score_rf_model.fit(satisfaction_score_X_combined[satisfaction_score_X_key_features],
                                satisfaction_score_y_synthetic)
satisfaction_score_importances = satisfaction_score_rf_model.feature_importances_

# Plot a graph of the importance of each variable
satisfaction_score_feature_importance = pd.DataFrame({'Feature': satisfaction_score_X_key_features,
                                                      'Importance': satisfaction_score_importances}.sort_values(by='Importance', ascending = False)) # Sort the DataFrame by importance for a better plot

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
print(satisfaction_score_ml_model)



# split this into the 5 different types of days:
# weekday and non-holiday
# weekend and non-holiday
# school holiday
# public holiday
# both school and public holiday


########################
## Seasonal Variation ##
########################

# Define the percentage change range from -50% to +50%
percentage_changes = np.linspace(-0.5, 0.5, 100)

# Get the mean values of each feature to use as the baseline
mean_crowd_level = satisfaction_score_X_combined['crowd_level'].mean()
mean_affordability = satisfaction_score_X_combined['affordability'].mean()
mean_cleanliness = satisfaction_score_X_combined['cleanliness'].mean()
mean_capacity = satisfaction_score_X_combined['capacity'].mean()

# Initialize lists to store satisfaction scores for each feature
# for weekdays and non-holidays
satisfaction_scores_crowd = []
satisfaction_scores_affordability = []
satisfaction_scores_cleanliness = []
satisfaction_scores_capacity = []

# Calculate the satisfaction score for each percentage change in the feature
for change in percentage_changes:
    # Vary crowd_level while keeping others constant
    crowd_level = mean_crowd_level * (1 + change)
    score_crowd = satisfaction_score(crowd_level, mean_affordability, mean_cleanliness, mean_capacity)
    satisfaction_scores_crowd.append(score_crowd)

    # Vary affordability while keeping others constant
    affordability = mean_affordability * (1 + change)
    score_affordability = satisfaction_score(mean_crowd_level, affordability, mean_cleanliness, mean_capacity)
    satisfaction_scores_affordability.append(score_affordability)

    # Vary cleanliness while keeping others constant
    cleanliness = mean_cleanliness * (1 + change)
    score_cleanliness = satisfaction_score(mean_crowd_level, mean_affordability, cleanliness, mean_capacity)
    satisfaction_scores_cleanliness.append(score_cleanliness)
    
    # Vary capacity while keeping others constant
    capacity = mean_capacity * (1 + change)
    score_capacity = satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness, capacity)
    satisfaction_scores_capacity.append(score_capacity)

# Plotting the results
plt.figure(figsize=(10, 10))

# Plot for crowd_level
plt.subplot(2, 2, 1)
# for all the graphs, have multiple lines that shows the percentage changes for each type of day
plt.plot(percentage_changes * 100, satisfaction_scores_crowd, label='Crowd Level', color='blue')
plt.axhline(y = satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness, mean_capacity),
            color = 'gray', linestyle = '--', label = 'Baseline Score')
plt.xlabel('% Change in Crowd Level')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Crowd Level on Satisfaction Score')
plt.legend()
plt.ylim(0, 60)

# Plot for affordability
plt.subplot(2, 2, 2)
plt.plot(percentage_changes * 100, satisfaction_scores_affordability, label='Affordability', color='green')
plt.axhline(y=satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness, mean_capacity),
            color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Affordability')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Affordability on Satisfaction Score')
plt.legend()
plt.ylim(0, 60)

# Plot for cleanliness
plt.subplot(2, 2, 3)
plt.plot(percentage_changes * 100, satisfaction_scores_cleanliness, label='Cleanliness', color='red')
plt.axhline(y=satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness,mean_capacity), color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Cleanliness')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Cleanliness on Satisfaction Score')
plt.legend()
plt.ylim(0, 60)

# Plot for capacity
plt.subplot(2, 2, 4)
plt.plot(percentage_changes * 100, satisfaction_scores_capacity, label='Capacity', color='purple')
plt.axhline(y=satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness,mean_capacity), color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Capacity')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Capacity on Satisfaction Score')
plt.legend()
plt.ylim(0, 60)

plt.tight_layout()
plt.show()


# for seasonal variations, try to use things like Halloween
# an example is to have an increased percentage of staff
# (e.g. increase staff by 20%, what's the change in satisfaction? Consider the costs of doing this also)
# look at the DIFFERENCES in outcomes by increasing a variable by a certain percentage
# make a graph of this for business suggestions
# satisfaction score will never be 100 because some people will always have issues
# we can just choose the parameters with higher importance
# --> if we have 3 variables, we need to consider all possible combinations!? 3C1 + 3C2 + 3C3
# tweak some parameters for the dynamic queue --> e.g. staff deployment

# input: csv file
# variables: to be decided


"""
let's import data from the csv file instead of hard coding it
if the Ride doesn't exist in the csv file, keep it but
- current columns in csv file: index, name of attraction, duration
- add columns like usage, crowd level, cleanliness etc
- cleanliness taken from the survey (justify this)
- crowd level from dynamic queue
- menu variety from survey
"""

"""
Assumptions:
Roads are not congested, although the ideal is to reduce congestion.
Everyone walks at the same speed (this means that it takes the same time for anyone to get from point A to point B).

"""
