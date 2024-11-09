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
print(df_rides)

# for seasonal attractions
nodes_seasonal = nodes_df.loc[nodes_df["type"] == "Seasonal"]
facilities_seasonal = facilities_df.loc[facilities_df["type"] == "Adhoc"]
df_seasonal = pd.merge(nodes_seasonal, facilities_df, how='left').sort_values(by = ["name"])
print(df_seasonal)

# for F&B (to be cleaned up)
nodes_fnb = nodes_df.loc[nodes_df["type"] == "Seasonal"]
facilities_fnb = facilities_df.loc[facilities_df["type"] == "Adhoc"]
df_fnb = pd.merge(nodes_fnb, facilities_fnb, how='left').sort_values(by = ["name"])
print(df_fnb)

# for retail (to be cleaned up)
nodes_retail = nodes_df.loc[nodes_df["type"] == "Seasonal"]
facilities_retail = facilities_df.loc[facilities_df["type"] == "Adhoc"]
df_retail = pd.merge(nodes_retail, facilities_retail, how='left').sort_values(by = ["name"])
print(df_retail)

df_rides.to_csv("../data/combined_data.csv", index=False)
df_seasonal.to_csv("../data/combined_data_1.csv", index=False)
