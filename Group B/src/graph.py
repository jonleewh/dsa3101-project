import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
import random
import datetime
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

np.random.seed(2024)
os.chdir('Group B/src')

"""
# walkthrough: change parameters of nodes and edges so that the satisfactory score is maximised and time spent in total is minimised
# constrained range of values
# capacity and staff available at the venue
# peak hours, special events, seasonal variations

look at theme park as a whole and what's happening at any point in time
generate each attraction's information every 5 minutes
graph --> physical space
dynamic queue --> time
combine both --> cohesive picture of how the theme park looks over time
separate the two graphs, have another graph where we can think of how to add/remove nodes
and how to adjust parameters like duration, capacity, etc --> run ML iteration
"""

############################################
## Read in Graph Data + Data Manipulation ##
############################################
nodes = pd.read_csv('../data/theme_park_nodes.csv')
nodes.fillna(0, inplace=True)
columns_to_convert = ["duration", "crowd_level", "cleanliness", "affordability", "capacity",
                      "actual_wait_time", "expected_wait_time", "staff"]
for column in columns_to_convert:
    nodes[column] = pd.to_numeric(nodes[column], errors='coerce')
edges = pd.read_csv('../data/theme_park_edges.csv')


######################
## Attraction Class ##
######################
class Attraction:
    def __init__(self, name, node_type, zone, crowd_level, duration, actual_waiting_time):
        self.name = name
        self.node_type = node_type
        self.zone = zone
        self.crowd_level = crowd_level # number of people in the queue
        self.duration = duration # duration
        self.actual_waiting_time = actual_waiting_time
    
    # Method to update crowd level
    def update_crowd_level(self, new_crowd):
        self.crowd_level = new_crowd
        print(f"{self.name} now has a crowd level of {self.crowd_level}")
    
    # Method to get details of the attraction
    def get_info(self):
        print(f"{self.name} is a {self.node_type} in {self.zone} with a current crowd level of {self.crowd_level}")

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
nodes["capacity"] = (nodes["capacity"] - nodes["capacity"].min()) / (nodes["capacity"].max() - nodes["capacity"].min())
nodes["crowd_level"] = (nodes["crowd_level"] - nodes["crowd_level"].min()) / (nodes["crowd_level"].max() - nodes["crowd_level"].min())
nodes["popularity"] = (nodes["popularity"] - nodes["popularity"].min()) / (nodes["popularity"].max() - nodes["popularity"].min())
nodes["staff"] = (nodes["staff"] - nodes["staff"].min()) / (nodes["staff"].max() - nodes["staff"].min())
nodes["duration"] = (nodes["duration"] - nodes["duration"].min()) / (nodes["duration"].max() - nodes["duration"].min())

def waiting_time(duration, crowd_level, capacity, staff, popularity, outdoor): # get the expected waiting time from the csv file
    # update this every 5 min
    # loop over the csv file to collect the data we want
    waiting_time = max(0,
                       1.5 * (crowd_level / (capacity + 1) * duration) # crowd level / capacity is the waiting time before someone who just joins the queue will be served
                       + 0.5 / (staff + 1) # the more staff we have, the lower the waiting time (for restaurants). If it is not a restaurant, ignore this (to be coded in)
                       + 0.8 * popularity
                       # include outdoor and rain
                       + np.random.normal(-5, 15)) # we introduce noise into the data
    return waiting_time

wait_time_X = nodes[['name', 'duration', 'crowd_level', 'capacity', 'staff', 'popularity', 'outdoor']]
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
nodes["affordability"] = (nodes["affordability"] - nodes["affordability"].min()) / (nodes["affordability"].max() - nodes["affordability"].min())
nodes["cleanliness"] = (nodes["cleanliness"] - nodes["cleanliness"].min()) / (nodes["cleanliness"].max() - nodes["cleanliness"].min())

# Calculate satisfaction/desirability score based on crowd level, wait time
# Suggestion: track how long a guest spends waiting, maximise satisfaction score AND minimise wait time.
# link to popularity
def satisfaction_score(crowd_level, affordability, cleanliness):
    # Using a logarithmic transformation for diminishing returns
    satisfaction_score_lr = (
        - 5 * crowd_level # higher crowd level results in lower satisfaction score
        + 4 * affordability # more affordable items results in higher satisfaction score
        + 2 * cleanliness # better cleanliness results in higher satisfaction score
        # + 5 / np.log1p(actual_wait_time) # longer wait time results in lower satisfaction score
        # + 3 / np.log1p(weather) # bad weather results in lower satisfaction score (heat and rain)
        # + 4 * np.log1p(ride_quality) # better ride quality results in higher satisfaction score
    ) # we input a first guess of the coefficients here first
    # logistic regression to get a value between 0 and 100?
    satisfaction_score = 1 / (1 + np.exp(-satisfaction_score_lr)) * 100
    return satisfaction_score
# Satisfaction score refers to the score for a single NODE, not the visitors.
# but it should be based on the visitor!

satisfaction_score_X = nodes[['name', 'crowd_level', 'affordability', 'cleanliness']]
satisfaction_score_X_key_features = ['crowd_level', 'affordability', 'cleanliness']
satisfaction_score_y = pd.DataFrame() # generate satisfaction scores based on X
for entry in satisfaction_score_X.itertuples():
    actual_satisfaction_score = satisfaction_score(entry.crowd_level, entry.affordability, entry.cleanliness)
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

# Initialize lists to store satisfaction scores for each feature
satisfaction_scores_crowd = []
satisfaction_scores_affordability = []
satisfaction_scores_cleanliness = []

# Calculate the satisfaction score for each percentage change in the feature
for change in percentage_changes:
    # Vary crowd_level while keeping others constant
    crowd_level = mean_crowd_level * (1 + change)
    score_crowd = satisfaction_score(crowd_level, mean_affordability, mean_cleanliness)
    satisfaction_scores_crowd.append(score_crowd)

    # Vary affordability while keeping others constant
    affordability = mean_affordability * (1 + change)
    score_affordability = satisfaction_score(mean_crowd_level, affordability, mean_cleanliness)
    satisfaction_scores_affordability.append(score_affordability)

    # Vary cleanliness while keeping others constant
    cleanliness = mean_cleanliness * (1 + change)
    score_cleanliness = satisfaction_score(mean_crowd_level, mean_affordability, cleanliness)
    satisfaction_scores_cleanliness.append(score_cleanliness)

# Plotting the results
plt.figure(figsize=(15, 5))

# Plot for crowd_level
plt.subplot(1, 3, 1)
plt.plot(percentage_changes * 100, satisfaction_scores_crowd, label='Crowd Level', color='blue')
plt.axhline(y=satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness), color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Crowd Level')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Crowd Level on Satisfaction Score')
plt.legend()
plt.ylim(0, 60)

# Plot for affordability
plt.subplot(1, 3, 2)
plt.plot(percentage_changes * 100, satisfaction_scores_affordability, label='Affordability', color='green')
plt.axhline(y=satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness), color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Affordability')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Affordability on Satisfaction Score')
plt.legend()
plt.ylim(0, 60)

# Plot for cleanliness
plt.subplot(1, 3, 3)
plt.plot(percentage_changes * 100, satisfaction_scores_cleanliness, label='Cleanliness', color='red')
plt.axhline(y=satisfaction_score(mean_crowd_level, mean_affordability, mean_cleanliness), color='gray', linestyle='--', label='Baseline Score')
plt.xlabel('% Change in Cleanliness')
plt.ylabel('Satisfaction Score')
plt.title('Effect of Cleanliness on Satisfaction Score')
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

###################################
## Creating Graph Data Structure ##
###################################

G = nx.Graph()

for node in nodes.itertuples():
    G.add_node(node.name, type = node.type, zone = node.zone, duration = node.duration, crowd_level = node.crowd_level, 
              cleanliness = node.cleanliness, affordability = node.affordability, capacity = node.capacity, 
              actual_wait_time = node.actual_wait_time, expected_wait_time = node.expected_wait_time, staff = node.staff )

for edge in edges.itertuples():
    G.add_edge(edge.source, edge.target, distance = edge.distance)

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


###############################################
## Shortest-Path Optimisation using Dijkstra ##
###############################################
def find_shortest_path(graph, start, end):
    undirected_graph = graph.to_undirected()
    path_nodes = nx.dijkstra_path(undirected_graph, start, end, weight = "distance")
    path_distance = nx.shortest_path_length(G, source = start, target = end, weight = "distance")
    return path_nodes, path_distance

path_data = []

for i in range(len(nodes)):
    for j in range(len(nodes)):
        path_nodes, path_distance = find_shortest_path(G, nodes['name'].loc[i], nodes['name'].loc[j])
        path_data.append({"source": nodes.loc[i, 'name'], "target": nodes.loc[j, 'name'],
                          "nodes": path_nodes, "distance": path_distance})

path_df = pd.DataFrame(path_data)
path_df.to_csv("../data/theme_park_paths.csv", index = False)


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


###############################
## Visualisation with Plotly ##
###############################

pio.renderers.default = "plotly_mimetype"
# Create a blank figure
fig = go.Figure()

pos_nodes = {
    "Mel's Mixtape": (6.5, 4.2),
    "Margo, Edith and Agnes Meet-and-Greet": (6.4, 2.4),
    "Illuminations Minion Monsters": (4.7, 2.6),
    "Restroom 1": (6.8, 4.1),
    "Starbucks": (4.5, 2.4),
    "Mel's Drive-In": (6.7, 4.4),
    "KT's Grill": (6.3, 3),
    "Star Snacks": (6.1, 3.3),
    "Pops! Popcorn Delight": (6, 3.6),
    "Pops Popcorn": (6, 3.6),
    "That's a Wrap!": (5, 3.4),
    "Universal Studios Store": (4.85, 3),
    "Candylicious": (5.2, 3.6),
    "Hello Kitty Studio": (6.8, 3.6),
    "Hello Kitty": (6.8, 3.6),
    "Minion Mart": (6.8, 3.1),
    "UNIVRS": (6.6, 2.25),
    "Lights Camera Action Hosted by Steven Spielberg": (7.4, 4.1),
    "Sesame Street Spaghetti Space Chase": (7.4, 3.9),
    "Rhythm Truck": (8.25, 4.1),
    "Restroom 2": (8.4, 3.9),
    "Loui's NY Pizza Parlor": (9, 4.1),
    "Big Bird's Emporium": (7.4, 4.4),
    "TRANSFORMERS The Ride: The Ultimate 3D Battle": (8.8, 4.8),
    "Accelerator": (8.75, 5.25),
    "Battlestar Galactica: Human": (8.7, 6.1),
    "Battlestar Galactica: HUMAN": (8.7, 6.1),
    "Battlestar Galactica: Cylon": (8.7, 5.65),
    "Battlestar Galactica: CYLON": (8.7, 5.65),
    "Restroom 3": (9.5, 5.1),
    "StarBot Cafe": (9, 5),
    "Galactic Treats": (8, 5),
    "Frozen Fuel": (7.7, 5),
    "The Sci-Fi Emporium": (8.8,4.5),
    "Planet Yen": (8,4.75),
    "Transformers Supply Vault": (8.8,4.3),
    "Revenge of the Mummy": (8.1, 7.2),
    "Treasure Hunters": (7, 7),
    "Restroom 4": (7, 6.8),
    "Oasis Spice Cafe": (6.8, 6.5),
    "Cairo Market": (7.9, 6.5),
    "Pharaoh's Dessert Oasis": (7.7, 6.3),
    "Carter's Curiosities": (7.5, 7),
    "Jurassic Park Rapids Adventure": (7, 7.9),
    "Dino Soarin'": (5.55, 8.25),
    "Canopy Flyer": (5.4, 7.85),
    "WaterWorld": (4.5, 7.65),
    "Restroom 5": (5.1, 7.8),
    "Restroom 6": (5.3, 7),
    "Discovery Food Court": (5, 8.2),
    "Fossil Fuels": (6, 7),
    "Mariner's Market": (5.2, 7.6),
    "Jungle Bites": (4.8, 8),
    "The Dino-Store": (5.8, 7),
    "Puss In Boots Giant Journey": (3.3, 8),
    "Magic Potion Spin": (3.1, 7.5),
    "Shrek 4D Adventure": (3.2, 6.8),
    "Enchanted Airways": (3.7, 6.1),
    "Donkey Live": (3, 7.95),
    "Fortune Favours The Furry": (3.1, 7.75),
    "Restroom 7": (4, 8),
    "Friar's Good Food": (4.1, 7.6),
    "Goldilocks": (4.2, 7.2),
    "Fairy Godmother's Potion Shop": (2.8, 7),
}

edge_x = []
edge_y = []

for edge in G.edges(data=True):
    x0, y0 = pos_nodes[edge[0]]
    x1, y1 = pos_nodes[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

# Edge traces
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    hoverinfo="text",                  
    mode="lines",
    line=dict(width=5)
)

node_x = []
node_y = []
node_text = []
customdata = []  # Store the detailed tooltip information separately
zone_colors = {
    "Far Far Away": "purple",
    "The Lost World": "green",
    "Hollywood": "peachpuff",
    "Sci-Fi City": "blue",
    "Ancient Egypt": "yellow",
    "New York": "pink",
    "Madagascar": "red"
}

# Create node colors list based on zones using a loop
node_colors = []
for node, data in G.nodes(data=True):
    zone = data.get("zone", "N/A")  # Get the zone of the node
    color = zone_colors.get(zone, "gray")  # Default to gray if zone not in zone_colors
    node_colors.append(color)


for node, data in G.nodes(data=True):
    x, y = pos_nodes[node]
    node_x.append(x)
    node_y.append(y)
    
    # Basic name for node label
    node_text.append(node)
    
    # Detailed hover information
    if data.get('type') in ['Seasonal', 'Ride']:
        tooltip_text = f"Name: {node}<br>Type: {data.get('type', 'N/A')}"
        tooltip_text += f"<br>Zone: {data.get('zone', 'N/A')}"
        tooltip_text += f"<br>Duration: {data.get('duration', 'N/A')}"
        tooltip_text += f"<br>Popularity: {data.get('popularity', 'N/A')}"
        tooltip_text += f"<br>Crowd Level: {data.get('crowd_level', 'N/A')}"
        tooltip_text += f"<br>Capacity: {data.get('capacity', 'N/A')}"
        tooltip_text += f"<br>Actual wait time: {data.get('actual_wait_time', 'N/A')}"
        tooltip_text += f"<br>Expected wait time: {data.get('expected_wait_time', 'N/A')}"
        tooltip_text += f"<br>Staff: {data.get('staff', 'N/A')}"
        customdata.append(tooltip_text)
    elif data.get('type') in ['Restroom']:
        tooltip_text = f"Name: {node}<br>Type: {data.get('type', 'N/A')}"
        tooltip_text += f"<br>Zone: {data.get('zone', 'N/A')}"
        tooltip_text += f"<br>Cleanliness: {data.get('cleanliness', 'N/A')}"
        tooltip_text += f"<br>Usage: {data.get('usage', 'N/A')}"
        customdata.append(tooltip_text)
    elif data.get('type') in ['Dining Outlet', 'Food Cart']:
        tooltip_text = f"Name: {node}<br>Type: {data.get('type', 'N/A')}"
        tooltip_text += f"<br>Zone: {data.get('zone', 'N/A')}"
        tooltip_text += f"<br>Affordability: {data.get('affordability', 'N/A')}"
        tooltip_text += f"<br>Capacity: {data.get('capacity', 'N/A')}"
        tooltip_text += f"<br>Actual wait time: {data.get('actual_wait_time', 'N/A')}"
        tooltip_text += f"<br>Expected wait time: {data.get('expected_wait_time', 'N/A')}"
        tooltip_text += f"<br>Staff: {data.get('staff', 'N/A')}"
        customdata.append(tooltip_text)
    elif data.get('type') in ['Retail']:
        tooltip_text = f"Name: {node}<br>Type: {data.get('type', 'N/A')}"
        tooltip_text += f"<br>Zone: {data.get('zone', 'N/A')}"
        tooltip_text += f"<br>Popularity: {data.get('popularity', 'N/A')}"
        tooltip_text += f"<br>Crowd Level: {data.get('crowd_level', 'N/A')}"
        tooltip_text += f"<br>Capacity: {data.get('capacity', 'N/A')}"
        tooltip_text += f"<br>Actual wait time: {data.get('actual_wait_time', 'N/A')}"
        tooltip_text += f"<br>Expected wait time: {data.get('expected_wait_time', 'N/A')}"
        tooltip_text += f"<br>Staff: {data.get('staff', 'N/A')}"
        customdata.append(tooltip_text)


# Update the Scatter trace
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=node_text,            # Show node names directly on nodes
    customdata=customdata,     # Store detailed information for hover
    hovertemplate="%{customdata}",  # Use customdata for the hover text
    textposition="middle center",
    textfont=dict(
        size=10,               
        color="black"          
    ),
    marker=dict(
        size=30,               
        color=node_colors,
        line=dict(width=2)
    )
)


# Plotting the graph
fig = go.Figure(data=[edge_trace, node_trace])

ussmap = Image.open("../data/uss_map.jpg")
fig.add_layout_image(
    dict(
        source=ussmap,  # Replace with your map file path or URL
        xref="x",
        yref="y",
        x=0,
        y=12,  # Adjust y to position the image correctly
        sizex=12,  # Adjust sizex to scale the image width
        sizey=12,  # Adjust sizey to scale the image height
        opacity=0.8,
        layer="below"  # Ensures the image is behind nodes and edges
    )
)

# Set up layout to remove axes (for a cleaner view)
fig.update_xaxes(range=[0, 12])  # Match these to sizex
fig.update_yaxes(range=[0, 12])  # Match these to sizey

fig.update_layout(
    width=1600,  # Adjust width to fit the map proportions
    height=1600,  # Adjust height to fit the map proportions
    xaxis_showgrid=False, yaxis_showgrid=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False
)

# Save the figure to an HTML file
html_file = "graph_output.html"
pio.write_html(fig, file=html_file, auto_open=False)

# Open the HTML file in the default web browser
webbrowser.open_new_tab(html_file)
