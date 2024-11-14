# migrate this to SQL eventually

import pandas as pd
import os

os.chdir('Group B/src')
print("Current working directory:", os.getcwd())
nodes_df = pd.read_csv("../data/theme_park_nodes_initial.csv")
facilities_df = pd.read_csv("../data/activity_counts_and_popularity_percentages.csv")

# for rides
nodes_rides = nodes_df.loc[nodes_df["type"] == "Ride"]
nodes_rides["index"] = "A"
nodes_rides["cleanliness"] = 84
df_rides = pd.merge(nodes_rides, facilities_df, left_on = 'index', right_on = 'letter', how='left')

# data manipulation for df_rides
df_rides = df_rides.drop(columns=['letter'])

# if there are NA values


# for seasonal attractions
nodes_seasonal = nodes_df.loc[nodes_df["type"] == "Seasonal"]
nodes_seasonal["index"] = "J"
nodes_seasonal["cleanliness"] = 84
df_seasonal = pd.merge(nodes_seasonal, facilities_df, left_on = 'index', right_on = 'letter', how='left')

# data manipulation for df_seasonal
df_seasonal = df_seasonal.drop(columns=['letter'])


# for Dining Outlets (i.e. restaurants)
nodes_fnb = nodes_df.loc[nodes_df["type"] == "Dining Outlet"]
nodes_fnb["index"] = "C"
nodes_fnb["cleanliness"] = 84
nodes_fnb["duration"] = 45 # on average, we can expect people to stay in a restaurant for around 45 minutes
df_fnb = pd.merge(nodes_fnb, facilities_df, left_on = 'index', right_on = 'letter', how='left')
df_fnb["affordability"] = 42 # based on our survey

# data manipulation for df_fnb
df_fnb = df_fnb.drop(columns=['letter'])


# for Food Carts
nodes_foodcarts = nodes_df.loc[nodes_df["type"] == "Food Cart"]
nodes_foodcarts["index"] = "C"
nodes_foodcarts["cleanliness"] = 84
nodes_fnb["duration"] = 3 # on average, we can expect food and drinks to be served in around 3 minutes
df_foodcarts = pd.merge(nodes_foodcarts, facilities_df, left_on = 'index', right_on = 'letter', how='left')
df_foodcarts["affordability"] = 42 # based on our survey
df_foodcarts["capacity"] = 1
df_foodcarts["duration"] = 3

# data manipulation for df_fnb
df_foodcarts = df_foodcarts.drop(columns=['letter'])


# for Retail
nodes_retail = nodes_df.loc[nodes_df["type"] == "Retail"]
nodes_retail["index"] = "D"
nodes_retail["cleanliness"] = 84
nodes_retail["duration"] = 15 # on average, we can expect people to stay in retail shops for around 15 minutes
df_retail = pd.merge(nodes_retail, facilities_df, left_on = 'index', right_on = 'letter', how='left')
df_retail["affordability"] = 42 # based on our survey

# data manipulation for df_retail
df_retail = df_retail.drop(columns=['letter'])


# for Restrooms
nodes_restroom = nodes_df.loc[nodes_df["type"] == "Restroom"]
nodes_restroom["index"] = "E"
nodes_restroom["duration"] = 10 # on average, we can expect people to use the restrooms for around 10 minutes
nodes_restroom["capacity"] = 5
df_restroom = pd.merge(nodes_restroom, facilities_df, left_on = 'index', right_on = 'letter', how='left')

# data manipulation for df_restroom
df_restroom = df_restroom.drop(columns=['letter'])

# for Entrance
nodes_entrance = nodes_df.loc[nodes_df["type"] == "Entrance"]
nodes_entrance["index"] = None
df_entrance = pd.merge(nodes_entrance, facilities_df, left_on = 'index', right_on = 'letter', how='left')

# data manipulation for df_entrance
df_entrance = df_entrance.drop(columns=['letter'])

df_combined = pd.concat([df_rides, df_seasonal, df_fnb, df_foodcarts, df_retail, df_restroom, df_entrance])
df_combined.to_csv("../data/theme_park_nodes.csv", index = False)
