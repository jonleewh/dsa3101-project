# migrate this to SQL eventually

import pandas as pd
import os

os.chdir('Group B/src')
print("Current working directory:", os.getcwd())
nodes_df = pd.read_csv("../data/theme_park_nodes.csv")
facilities_df = pd.read_csv("../data/facilities_information.csv")

# for rides
nodes_rides = nodes_df.loc[nodes_df["type"] == "Ride"]
facilities_rides = facilities_df.loc[facilities_df["type"] == "Attraction"]
df_rides = pd.merge(nodes_rides, facilities_rides, on='name', how='left').sort_values(by = ["name"])

"""
A = roller coaster
B = water-based rides
G = child-friendly rides
H = Haunted House
I = Simulators with 3D/4D experiences
"""
for i in range(len(df_rides)):
    if df_rides.loc[i, 'index'] == 'A':
        df_rides.loc[i, 'popularity'] = 80
    elif df_rides.loc[i, 'index'] == 'B':
        df_rides.loc[i, 'popularity'] = 75
    elif df_rides.loc[i, 'index'] == 'G':
        df_rides.loc[i, 'popularity'] = 59
    elif df_rides.loc[i, 'index'] == 'H':
        df_rides.loc[i, 'popularity'] = 65
    elif df_rides.loc[i, 'index'] == 'I':
        df_rides.loc[i, 'popularity'] = 80

# data manipulation for df_rides
df_rides = df_rides.drop(columns=['type_y']).rename(columns={'type_x': 'type'})
df_rides = df_rides.drop(columns=['capacity_x']).rename(columns={'capacity_y': 'capacity'})
df_rides = df_rides.drop(columns=['duration']).rename(columns={'duration (in min)': 'duration'})
print(df_rides)
df_rides.to_csv("../data/combined_data_1.csv", index=False)


# for seasonal attractions
nodes_seasonal = nodes_df.loc[nodes_df["type"] == "Seasonal"]
facilities_seasonal = facilities_df.loc[facilities_df["type"] == "Adhoc"]
df_seasonal = pd.merge(nodes_seasonal, facilities_seasonal, how='outer').sort_values(by = ["name"])
df_seasonal = df_seasonal[df_seasonal['type'] != 'Adhoc']
df_seasonal["index"] = "J"

# for Dining Outlets (i.e. restaurants)
nodes_fnb = nodes_df.loc[nodes_df["type"] == "Dining Outlet"]
facilities_fnb = facilities_df.loc[facilities_df["index"] == "C"]
df_fnb = pd.merge(nodes_fnb, facilities_fnb, how='outer').sort_values(by = ["name"])
df_fnb = df_fnb[df_fnb['index'] != 'C']
df_fnb["index"] = "C"
df_fnb["affordability"] = 42

# for Food Carts
nodes_foodcarts = nodes_df.loc[nodes_df["type"] == "Food Cart"]
facilities_foodcarts = facilities_df.loc[facilities_df["index"] == "C"]
df_foodcarts = pd.merge(nodes_foodcarts, facilities_foodcarts, how='outer').sort_values(by = ["name"])
df_foodcarts = df_foodcarts[df_foodcarts['index'] != 'C']
df_foodcarts["index"] = "C"
df_foodcarts["affordability"] = 42

# for Retail
nodes_retail = nodes_df.loc[nodes_df["type"] == "Retail"]
facilities_retail = facilities_df.loc[facilities_df["index"] == "D"]
df_retail = pd.merge(nodes_retail, facilities_retail, how='outer').sort_values(by = ["name"])
df_retail = df_retail[df_retail['index'] != 'D']
df_retail["index"] = "D"

# for Restrooms
nodes_restroom = nodes_df.loc[nodes_df["type"] == "Restroom"]
facilities_restroom = facilities_df.loc[facilities_df["index"] == "E"]
df_restroom = pd.merge(nodes_restroom, facilities_restroom, how='outer').sort_values(by = ["name"])
df_restroom = df_restroom[df_restroom['index'] != 'E']
df_restroom["index"] = "E"
df_restroom["popularity"] = 79

df = pd.concat([df_rides, df_seasonal, df_fnb, df_foodcarts, df_retail, df_restroom], ignore_index=True)

df.to_csv("../data/combined_data.csv", index=False)
