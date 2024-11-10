# migrate this to SQL eventually

import pandas as pd
import os

os.chdir('Group B/src')
print("Current working directory:", os.getcwd())
nodes_df = pd.read_csv("../data/theme_park_nodes_initial.csv")
facilities_df = pd.read_csv("../data/facilities_information.csv")

# for rides
nodes_rides = nodes_df.loc[nodes_df["type"] == "Ride"]
facilities_rides = facilities_df.loc[facilities_df["type"] == "Attraction"]
df_rides = pd.merge(nodes_rides, facilities_rides, on='name', how='left').sort_values(by = ["name"])

# data manipulation for df_rides
df_rides = df_rides.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_rides = df_rides.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_rides = df_rides.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})


# for seasonal attractions
nodes_seasonal = nodes_df.loc[nodes_df["type"] == "Seasonal"]
facilities_seasonal = facilities_df.loc[facilities_df["type"] == "Adhoc"]
df_seasonal = pd.merge(nodes_seasonal, facilities_seasonal,
                       left_on = 'duration', right_on = 'duration (in min)', how = 'left')

# data manipulation for df_seasonal
df_seasonal = df_seasonal.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_seasonal = df_seasonal.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_seasonal = df_seasonal.drop(columns=['name_y']).rename(columns={'name_x': 'name'})
df_seasonal = df_seasonal.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})


# for Dining Outlets (i.e. restaurants)
nodes_fnb = nodes_df.loc[nodes_df["type"] == "Dining Outlet"]
facilities_fnb = facilities_df.loc[facilities_df["index"] == "C"]
facilities_fnb = facilities_fnb[facilities_fnb['name'].str.contains("Restaurant")].head(1)
facilities_fnb["name"] = facilities_fnb["name"].str.replace('Restaurant 1', 'Dining Outlet', regex=True)
df_fnb = pd.merge(nodes_fnb, facilities_fnb,
                  left_on = 'type', right_on = 'name', how = 'left')
df_fnb["affordability"] = 42

# data manipulation for df_fnb
df_fnb = df_fnb.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_fnb = df_fnb.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_fnb = df_fnb.drop(columns=['name_y']).rename(columns={'name_x': 'name'})
df_fnb = df_fnb.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})


# for Food Carts
nodes_foodcarts = nodes_df.loc[nodes_df["type"] == "Food Cart"]
facilities_foodcarts = facilities_df.loc[facilities_df["index"] == "C"]
facilities_foodcarts = facilities_foodcarts[facilities_foodcarts['name'].str.contains("Drink Stall")].head(1)
facilities_foodcarts["name"] = facilities_foodcarts["name"].str.replace('Drink Stall 1', 'Food Cart', regex=True)
df_foodcarts = pd.merge(nodes_foodcarts, facilities_foodcarts,
                  left_on = 'type', right_on = 'name', how = 'left')
df_foodcarts["affordability"] = 42
df_foodcarts["capacity"] = 1
df_foodcarts["duration"] = 3

# data manipulation for df_fnb
df_foodcarts = df_foodcarts.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_foodcarts = df_foodcarts.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_foodcarts = df_foodcarts.drop(columns=['name_y']).rename(columns={'name_x': 'name'})
df_foodcarts = df_foodcarts.drop(columns=['duration (in min)'])


# for Retail
nodes_retail = nodes_df.loc[nodes_df["type"] == "Retail"]
facilities_retail = facilities_df.loc[facilities_df["index"] == "D"]
facilities_retail["name"] = facilities_retail["name"].head(1).str.replace('Souvenir Shop 1', 'Retail', regex=True)
df_retail = pd.merge(nodes_retail, facilities_retail,
                  left_on = 'type', right_on = 'name', how = 'left')
df_retail["affordability"] = 42

# data manipulation for df_retail
df_retail = df_retail.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_retail = df_retail.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_retail = df_retail.drop(columns=['name_y']).rename(columns={'name_x': 'name'})
df_retail = df_retail.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})


# for Restrooms
nodes_restroom = nodes_df.loc[nodes_df["type"] == "Restroom"]
facilities_restroom = facilities_df.loc[facilities_df["index"] == "E"]
facilities_restroom["name"] = facilities_restroom["name"].head(1).str.replace(' 1', '', regex=True)
df_restroom = pd.merge(nodes_restroom, facilities_restroom,
                       left_on = 'type', right_on = 'name', how = 'left')

# data manipulation for df_restroom
df_restroom = df_restroom.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_restroom = df_restroom.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_restroom = df_restroom.drop(columns=['name_y']).rename(columns={'name_x': 'name'})
df_restroom = df_restroom.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})


# for Entrance
df_entrance = nodes_df.loc[nodes_df["type"] == "Entrance"]
df_entrance = pd.merge(df_entrance, facilities_restroom,
                       left_on = 'type', right_on = 'name', how = 'left')

# data manipulation for df_entrance
df_entrance = df_entrance.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_entrance = df_entrance.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_entrance = df_entrance.drop(columns=['name_y']).rename(columns={'name_x': 'name'})
df_entrance = df_entrance.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})


df = pd.concat([df_rides, df_seasonal, df_fnb, df_foodcarts, df_retail, df_restroom, df_entrance], ignore_index = True)
df.to_csv("../data/theme_park_nodes.csv", index=False)
