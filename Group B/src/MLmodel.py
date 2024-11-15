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

combined_dfs = [combined_df_both_holiday, combined_df_public_holiday, combined_df_school_holiday,
               combined_df_weekday_non_holiday, combined_df_weekend_non_holiday]

combined_df_both_holiday = pd.merge(combined_df_both_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_public_holiday = pd.merge(combined_df_public_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_school_holiday = pd.merge(combined_df_school_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_weekday_non_holiday = pd.merge(combined_df_weekday_non_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])
combined_df_weekend_non_holiday = pd.merge(combined_df_weekend_non_holiday, nodes, left_on = 'attraction', right_on = 'name', how='left').drop(columns = ["name"])

print(combined_df_both_holiday)
print(combined_df_public_holiday)
print(combined_df_school_holiday)
print(combined_df_weekday_non_holiday)
print(combined_df_weekend_non_holiday)

# conn = sqlite3.connect("weather_holiday_database.db")
# combined_df_both_holiday.to_sql("combined_df_both_holiday", conn, if_exists="replace", index=False)
# combined_df_public_holiday.to_sql("combined_df_public_holiday", conn, if_exists="replace", index=False)
# combined_df_school_holiday.to_sql("combined_df_school_holiday", conn, if_exists="replace", index=False)
# combined_df_weekday_non_holiday.to_sql("combined_df_weekday_non_holiday", conn, if_exists="replace", index=False)
# combined_df_weekend_non_holiday.to_sql("combined_df_weekend_non_holiday", conn, if_exists="replace", index=False)
# weather_data_hour_1.to_sql("weather_data_hour_1", conn, if_exists="replace", index=False)
# weather_data_hour_2.to_sql("weather_data_hour_2", conn, if_exists="replace", index=False)
# weather_data_hour_3.to_sql("weather_data_hour_3", conn, if_exists="replace", index=False)
# weather_data_hour_4.to_sql("weather_data_hour_4", conn, if_exists="replace", index=False)
# weather_data_hour_5.to_sql("weather_data_hour_5", conn, if_exists="replace", index=False)

# print("Tables in database:")
# cursor = conn.cursor()

# # Join with weather_data_hour_1
# cursor.execute("DROP TABLE IF EXISTS combined_df_both_holiday_updated;")
# cursor.execute(
#     """
#     CREATE TABLE combined_df_both_holiday_updated AS
#     SELECT a.*, b.* 
#     FROM combined_df_both_holiday AS a
#     LEFT JOIN weather_data_hour_1 AS b ON a.type_of_day = b.type_of_day;
#     """)

# # Join with weather_data_hour_2
# cursor.execute("DROP TABLE IF EXISTS combined_df_public_holiday_updated;")
# cursor.execute(
#     """
#     CREATE TABLE combined_df_public_holiday_updated AS
#     SELECT a.*, b.* 
#     FROM combined_df_public_holiday AS a
#     LEFT JOIN weather_data_hour_2 AS b ON a.type_of_day = b.type_of_day;
#     """)

# # Join with weather_data_hour_3
# cursor.execute("DROP TABLE IF EXISTS combined_df_school_holiday_updated;")
# cursor.execute(
#     """
#     CREATE TABLE combined_df_school_holiday_updated AS
#     SELECT a.*, b.* 
#     FROM combined_df_school_holiday AS a
#     LEFT JOIN weather_data_hour_3 AS b ON a.type_of_day = b.type_of_day;
#     """)

# # Join with weather_data_hour_4
# cursor.execute("DROP TABLE IF EXISTS combined_df_weekday_non_holiday_updated;")
# cursor.execute(
#     """
#     CREATE TABLE combined_df_weekday_non_holiday_updated AS
#     SELECT a.*, b.* 
#     FROM combined_df_weekday_non_holiday AS a
#     LEFT JOIN weather_data_hour_4 AS b ON a.type_of_day = b.type_of_day;
#     """)

# # Join with weather_data_hour_5
# cursor.execute("DROP TABLE IF EXISTS combined_df_weekend_non_holiday_updated;")
# cursor.execute(
#     """
#     CREATE TABLE combined_df_weekend_non_holiday_updated AS
#     SELECT a.*, b.* 
#     FROM combined_df_weekend_non_holiday AS a
#     LEFT JOIN weather_data_hour_5 AS b ON a.type_of_day = b.type_of_day;
#     """)

# print("Tables in database:")
# query1 = "SELECT * FROM combined_df_both_holiday_updated"
# new_df1 = pd.read_sql_query(query1, conn)

# query2 = "SELECT * FROM combined_df_public_holiday_updated"
# new_df2 = pd.read_sql_query(query2, conn)

# query3 = "SELECT * FROM combined_df_school_holiday_updated"
# new_df3 = pd.read_sql_query(query3, conn)

# query4 = "SELECT * FROM combined_df_weekday_non_holiday_updated"
# new_df4 = pd.read_sql_query(query4, conn)

# query5 = "SELECT * FROM combined_df_weekend_non_holiday_updated"
# new_df5 = pd.read_sql_query(query5, conn)

# print("done")

# conn.close()

# print(new_df1.head())

# Original Python code
combined_df_both_holiday = pd.merge(weather_data_hour_1, combined_df_both_holiday, on = 'type_of_day', how='left')
print("done 1")
combined_df_public_holiday = pd.merge(weather_data_hour_2, combined_df_public_holiday, on = 'type_of_day', how='left')
print("done 2")
combined_df_school_holiday = pd.merge(weather_data_hour_3, combined_df_school_holiday, on = 'type_of_day', how='left')
print("done 3")
combined_df_weekday_non_holiday = pd.merge(weather_data_hour_4, combined_df_weekday_non_holiday, on = 'type_of_day', how='left')
print("done 4")
combined_df_weekend_non_holiday = pd.merge(weather_data_hour_5, combined_df_weekend_non_holiday, on = 'type_of_day', how='left')
print("done 5")

print(combined_df_both_holiday)
print(combined_df_public_holiday)
print(combined_df_school_holiday)
print(combined_df_weekday_non_holiday)
print(combined_df_weekend_non_holiday)


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
def waiting_time(type, duration, crowd_level, capacity, temperature, rain, staff, outdoor): # calculate expected waiting time for a ride.
    """
    Parameters:
    - type: the type of the ride.
    - duration: Duration of the ride (in minutes).
    - crowd_level: Number of people currently in the queue.
    - capacity: Capacity of the attraction (people per cycle).
    - staff: Number of staff available (for F&B and retail).
    - popularity: Popularity of the attraction.
    - outdoor: Whether the attraction is outdoor (influenced by weather).
    
    Return expected waiting time in minutes.
    """
    if type == 'Food Cart': # assumed to be outdoors
        if rain == 1:
            waiting_time = 0 # because there will be no customers who will be out buying F&B from the food carts
        else:
            waiting_time = min(0,
                               crowd_level / (capacity + 1) * duration
                               + 10 / staff
                               + np.random.normal(-1, 5))
    
    elif type == 'Retail' or type == "Dining Outlet": # assumed to be indoors
        waiting_time = min(0, 
                           crowd_level / (capacity + 1) * duration
                           + 10 / staff
                           + 15 * rain # if it is raining, we assume an additional 15 minutes of waiting time added
                           + np.random.normal(-5, 15)) # we introduce noise into the data
    
    else: # for rides, we assume that there will be sufficient staff to operate the rides
        if rain == 1 and outdoor == 1:
            waiting_time = 0
        else:
            waiting_time = min(0,
                               crowd_level / (capacity + 1) * duration
                               + 15 * rain # if it is raining, we assume an additional 15 minutes of waiting time added
                               + np.random.normal(-5, 15) # we introduce noise into the data
                               )
    return waiting_time


for combined_df in combined_dfs:
    wait_time_X = combined_df[['attraction', 'type', 'duration', 'crowd_level', 'capacity', 'staff', 'popularity', 'outdoor']]
wait_time_key_X_features = ['duration', 'crowd_level', 'capacity', 'staff', 'popularity', 'outdoor']
wait_time_y_synthetic = pd.DataFrame() # generate wait times based on X
for entry in wait_time_X.itertuples():
    actual_waiting_time = waiting_time(entry.duration, entry.crowd_level, entry.capacity, entry.staff, entry.popularity, entry.outdoor)
    wait_time_y_synthetic = pd.concat([wait_time_y_synthetic, pd.DataFrame({"waiting_time": [actual_waiting_time]})], ignore_index=True)
print(wait_time_y_synthetic)

# Train a Random Forest regressor to evaluate importance of each feature
wait_time_rf_model = RandomForestRegressor(n_estimators = 10)
wait_time_rf_model.fit(wait_time_X[wait_time_key_X_features], wait_time_y_synthetic)
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


####################################
## Calculating Satisfaction Score ##
####################################
# Arguments are the properties of the node. Can put in the object
nodes["popularity"] = (nodes["popularity"] - nodes["popularity"].min()) / (nodes["popularity"].max() - nodes["popularity"].min())
nodes["staff"] = (nodes["staff"] - nodes["staff"].min()) / (nodes["staff"].max() - nodes["staff"].min())
nodes["affordability"] = (nodes["affordability"] - nodes["affordability"].min()) / (nodes["affordability"].max() - nodes["affordability"].min())
nodes["cleanliness"] = (nodes["cleanliness"] - nodes["cleanliness"].min()) / (nodes["cleanliness"].max() - nodes["cleanliness"].min())

# Calculate satisfaction/desirability score based on crowd level, wait time
# Satisfaction score refers to the score for a single NODE, not the visitors.
# Suggestion: track how long a guest spends waiting, maximise satisfaction score AND minimise wait time.
# link to popularity
def satisfaction_score(crowd_level, affordability, cleanliness, capacity, actual_wait_time, temperature, rain, ride_quality):
    # Using a logarithmic transformation for diminishing returns
    # we input a first guess of the coefficients here first
    # score is between 0 and 100, but we assume that there are no extremes.
    # it is impossible for a customer to be 100% satisfied or 100% dissatisfied.
    satisfaction_score_lr = max(5,
                                min(95,
                                    (50 # average satisfaction score
                                     - 5 * crowd_level # higher crowd level results in lower satisfaction score
                                     + 4 * affordability # more affordable items results in higher satisfaction score
                                     + 2 * cleanliness # better cleanliness results in higher satisfaction score
                                     + 2 * capacity # higher capacity results in higher satisfaction score
                                     - 5 * actual_wait_time # longer wait time results in lower satisfaction score
                                     - 5 * temperature # higher temperatures results in lower satisfaction score
                                     # for temperature, try to normalise the values where 30 degrees is zero, anything above has positive value, anything below has negative value
                                     - 5 * rain # rain results in lower satisfaction score
                                     + 4 * ride_quality # better ride quality results in higher satisfaction score
                                     + np.random.normal(-10, 15) # we introduce noise into the data
                                     )
                                    )
                                )
    return satisfaction_score

 # split this into the 5 different types of days:
 # weekday and non-holiday
 # weekend and non-holiday
 # school holiday
 # public holiday
 # both school and public holiday
satisfaction_score_X = nodes[['name', 'crowd_level', 'affordability', 'cleanliness', 'capacity']]
satisfaction_score_X_key_features = ['crowd_level', 'affordability', 'cleanliness', 'capacity']
satisfaction_score_y = pd.DataFrame() # generate satisfaction scores based on X
for entry in satisfaction_score_X.itertuples():
    actual_satisfaction_score = satisfaction_score(entry.crowd_level, entry.affordability, entry.cleanliness, entry.capacity)
    satisfaction_score_y = pd.concat([satisfaction_score_y, pd.DataFrame({"waiting_time": [actual_satisfaction_score]})], ignore_index=True)
print(satisfaction_score_X)
print(satisfaction_score_y)

# Train a Random Forest regressor to evaluate importance of each feature
satisfaction_score_rf_model = RandomForestRegressor(n_estimators = 10)
satisfaction_score_rf_model.fit(satisfaction_score_X[satisfaction_score_X_key_features],
                                satisfaction_score_y)
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
satisfaction_score_X_importance = satisfaction_score_X[satisfaction_score_X_key_features] * satisfaction_score_importances # Each column in X is scaled by its corresponding importance from the random forest model

# Train the linear regression model with the coefficients accounting for the importance
satisfaction_score_ml_model = LinearRegression()
satisfaction_score_ml_model.fit(satisfaction_score_X_importance, satisfaction_score_y)
print(satisfaction_score_ml_model)


########################
## Seasonal Variation ##
########################

# Define the percentage change range from -50% to +50%
percentage_changes = np.linspace(-0.5, 0.5, 100)

# Get the mean values of each feature to use as the baseline
mean_crowd_level = satisfaction_score_X['crowd_level'].mean()
mean_affordability = satisfaction_score_X['affordability'].mean()
mean_cleanliness = satisfaction_score_X['cleanliness'].mean()
mean_capacity = satisfaction_score_X['capacity'].mean()

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
def seasonal_variation():
    return


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

# Simulate park experience over a day (time range: 10am to 7pm)
for hour in range(10, 20):
    print(f"\n--- Time: {hour}:00 ---")
    for node in G.nodes(data=True):
        base_wait = node[1]["base_wait"]
        base_crowd = node[1]["base_crowd"]
        
        # Update wait time and crowd level dynamically
        wait_time, crowd_level = dynamic_crowd_wait(hour, base_wait, base_crowd)
        
        # Calculate satisfaction score
        satisfaction = calculate_satisfaction(wait_time, crowd_level, popularity)
        
        # Print status
        print(f"{node[0]} - Wait Time: {wait_time:.1f} mins, Crowd Level: {crowd_level:.1f}, Satisfaction: {satisfaction:.1f}")
"""

##########################
## Optimising Itinerary ##
##########################
def optimized_itinerary(start, attractions_list, current_hour):
    total_time = 0
    itinerary = [start]
    current_location = start
    
    for attraction in attractions_list:
        # Calculate dynamic wait time and crowd level for attraction
        wait_time, crowd_level = waiting_time(current_hour,
                                              G.nodes[attraction]["base_wait"],
                                              G.nodes[attraction]["base_crowd"])
        
        # Satisfaction score
        satisfaction = satisfaction_score(wait_time, crowd_level, G.nodes[attraction]["popularity"])
        
        # Pathfinding to minimize travel time between attractions
        path = find_shortest_path(G, current_location, attraction)
        travel_time = sum(G.edges[path[i], path[i+1]]["distance"] for i in range(len(path)-1))
        
        # Update itinerary and time
        # how to track wait time and minimise it?
        # A possible way is based on the guest's threshold waiting time vs the expected waiting time
        # If the waiting time is suddenly reduced, will it allow the customer to return to the node?
        # After every period of time, the guest can re-evaluate the optimised itinerary
        # Consider both the global time and attraction-specific time (e.g. different Ride durations)
        total_time += wait_time + travel_time
        itinerary.append((attraction, wait_time, satisfaction))
        current_location = attraction
        
        # Simulate moving to the next hour
        current_hour = (current_hour + 1) % 24
    
    return itinerary, total_time
