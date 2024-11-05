import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
import random
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


G = nx.Graph()

class Attraction:
    def __init__(self, name, node_type, zone, crowd_level, duration):
        self.name = name
        self.node_type = node_type
        self.zone = zone
        self.crowd_level = crowd_level # number of people in the queue
        self.duration = duration # duration
    
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

# add nodes, ensure properties are in the format (key = value)
# we can use the waiting time as a proxy for the crowd level using the csv file
# replace with the csv file data rather than generating it by ourselves
def dynamic_crowd_wait(time_of_day, expected_waiting_time): # get the expected waiting time from the csv file
    # update this every 5 min
    # loop over the csv file to collect the data we want
    actual_wait_time = expected_waiting_time + np.random(0, 10) # may need to change the random function
    return actual_wait_time

# loop over the csv file to collect the data we want
# code to read csv file
for row in csv_file: # need to change the csv_file
    actual_wait_time = dynamic_crowd_wait(
        # extract the time of the day
        # extract the expected_waiting_time
    )
    crowd_level = actual_wait_time * 5


# Calculate satisfaction/desirability score based on crowd level, wait time, and popularity
# Suggestion: track how long a guest spends waiting, maximise satisfaction score AND minimise wait time.
def calculate_satisfaction(actual_wait_time, crowd_level, popularity, menu_variety):
    # Satisfaction score example: higher with low crowd/wait and high popularity
    # may want to use multiple LR here
    satisfaction_score = (10 - 0.5 * actual_wait_time - 0.3 * crowd_level
                          + 0.2 * popularity + 0.2 * menu_variety) # we input a first guess of the coefficients here first
    return satisfaction_score

X = None # import csv file
y = None # generate satisfaction score based on X

# Train a Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X, y)

# Check feature importances
importances = rf_model.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, importances):
    print(f'Feature: {name}, Importance: {importance}')

# Apply weights based on feature importances
X_weighted = X * importances # Each column in X is scaled by its corresponding importance from the random forest model

# Train the linear regression model with the coefficients accounting for the importance
model = LinearRegression()
model.fit(X_weighted, y)


# Simulate park experience over a day (time range: 10am to 7pm)
for hour in range(10, 20):
    print(f"\n--- Time: {hour}:00 ---")
    for node in G.nodes(data=True):
        base_wait = node[1]["base_wait"]
        base_crowd = node[1]["base_crowd"]
        popularity = node[1]["popularity"]
        
        # Update wait time and crowd level dynamically
        wait_time, crowd_level = dynamic_crowd_wait(hour, base_wait, base_crowd)
        
        # Calculate satisfaction score
        satisfaction = calculate_satisfaction(wait_time, crowd_level, popularity)
        
        # Print status
        print(f"{node[0]} - Wait Time: {wait_time:.1f} mins, Crowd Level: {crowd_level:.1f}, Satisfaction: {satisfaction:.1f}")

# Shortest-path optimization based on Dijkstra’s algorithm
# but is shortest-path relevant to wait time and guest satisfaction?
def find_shortest_path(graph, start, end):
    return nx.dijkstra_path(graph, start, end, weight="distance")

# Optimize itinerary
def optimized_itinerary(start, attractions_list, current_hour):
    total_time = 0
    itinerary = [start]
    current_location = start
    
    for attraction in attractions_list:
        # Calculate dynamic wait time and crowd level for attraction
        wait_time, crowd_level = dynamic_crowd_wait(current_hour, 
                                                    G.nodes[attraction]["base_wait"], 
                                                    G.nodes[attraction]["base_crowd"])
        
        # Satisfaction score
        satisfaction = calculate_satisfaction(wait_time, crowd_level, G.nodes[attraction]["popularity"])
        
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

"""
let's import data from the csv file instead of hard coding it
if the Ride doesn't exist in the csv file, keep it but
- current columns in csv file: index, name of attraction, duration, popularity score (Jamie is doing this)
- add columns like usage, crowd level, cleanliness etc
- cleanliness taken from the survey (justify this)
- crowd level from dynamic queue
- menu variety from survey
"""

# create another similar csv file with the same column names and play around with the values

#############
# Hollywood #
#############

G.add_node("Mel's Mixtape", type = "Seasonal", zone = "Hollywood",
           duration = 20, popularity = 80,
           timeslots = {datetime.time(10, 35), datetime.time(13, 0), datetime.time(15, 0), datetime.time(17, 55)},
           crowd_level = 60)

G.add_node("Margo, Edith and Agnes Meet-and-Greet", type = "Seasonal", zone = "Hollywood",
           duration = 5, popularity = 80,
           timeslots = {datetime.time(10, 5), datetime.time(11, 55), datetime.time(13, 55), datetime.time(15, 55), datetime.time(17, 25)},
           crowd_level = 60)

G.add_node("Illuminations Minion Monsters", type = "Seasonal", zone = "Hollywood",
           duration = 5, popularity = 80,
           timeslots = {datetime.time(10, 15), datetime.time(12, 25), datetime.time(14, 5), datetime.time(15, 50)},
           crowd_level = 60)

G.add_node("Restroom 1", type = "Restroom", zone = "Hollywood",
           cleanliness = 90, usage = 30)

G.add_node("Starbucks", type = "Dining Outlet", zone = "Hollywood",
           menu_variety = 10, capacity = 150, 
           crowd_level = 80, actual_wait_time = 100)

G.add_node("Mel's Drive-In", type = "Dining Outlet", zone = "Hollywood",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("KT's Grill", type = "Dining Outlet", zone = "Hollywood",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Star Snacks", type = "Food Cart", zone = "Hollywood",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Pops! Popcorn Delight", type = "Food Cart", zone = "Hollywood",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("That's a Wrap!", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Candylicious", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Universal Studios Store", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 100, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Hello Kitty Studio", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 99, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Minion Mart", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 99, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("UNIVRS", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 99, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

# add edges within Hollywood
G.add_edge("Universal Studios Store", "Hello Kitty", distance = 1)
G.add_edge("Universal Studios Store", "Pops Popcorn", distance = 2)
G.add_edge("Hello Kitty", "Pops Popcorn", distance = 2)
G.add_edge("Minion Mart", "Pops Popcorn", distance = 2)
G.add_edge("Minion Mart", "Starbucks", distance = 1)
G.add_edge("Starbucks", "Pops Popcorn", distance = 1)
G.add_edge("Starbucks", "Candylicious", distance = 1)
G.add_edge("Candylicious", "Pops Popcorn", distance = 2)
G.add_edge("Hello Kitty", "Mel's Drive-In", distance = 5)
G.add_edge("Minion Mart", "Mel's Drive-In", distance = 4)
G.add_edge("Restroom 1", "Mel's Drive-In", distance = 1)
G.add_edge("Restroom 1", "Star Snacks", distance = 2)
G.add_edge("Restroom 1", "KT's Grill", distance = 1)
G.add_edge("KT's Grill", "UNIVRS", distance = 3)
G.add_edge("Star Snacks", "UNIVRS", distance = 3)
G.add_edge("Mel's Drive-In", "UNIVRS", distance = 4)
G.add_edge("Mel's Drive-In", "Mel's Mixtape", distance = 1)
G.add_edge("Mel's Drive-In", "UNIVRS", distance = 3)



############
# New York #
############

G.add_node("Lights Camera Action Hosted by Steven Spielberg", type = "Ride", zone = "New York",
           duration = 10, capacity = 150, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Sesame Street Spaghetti Space Chase", type = "Ride", zone = "New York",
           duration = 10, capacity = 20, popularity = 99, expected_wait_time = 10,
           crowd_level = 80, actual_wait_time = 5)

G.add_node("Rhythm Truck", type = "Seasonal", zone = "New York",
           duration = 20, popularity = 80,
           timeslots = {datetime.time(11, 30), datetime.time(13, 30), datetime.time(15, 30), datetime.time(18, 20)},
           crowd_level = 60)

G.add_node("Restroom 2", type = "Restroom", zone = "New York",
           cleanliness = 90, usage = 30)

G.add_node("Loui's NY Pizza Parlor", type = "Dining Outlet", zone = "New York",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Big Bird's Emporium", type = "Retail", zone = "New York",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

# add edges within New York
G.add_edge("Restroom 2", "Loui's NY Pizza Parlor", distance = 3)



###############
# Sci-Fi City #
###############

G.add_node("Transformers", type = "Ride", zone = "Sci-Fi City",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Accelerator", type = "Ride", zone = "Sci-Fi City",
           duration = 10, capacity = 30, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Battlestar Galactica: Human", type = "Ride", zone = "Sci-Fi City",
           duration = 3, capacity = 20, popularity = 100, expected_wait_time = 15, staff = 8,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Battlestar Galactica: Cylon", type = "Ride", zone = "Sci-Fi City",
           duration = 3, capacity = 20, popularity = 100, expected_wait_time = 15, staff = 8,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Restroom 3", type = "Restroom", zone = "Sci-Fi City",
           cleanliness = 90, usage = 30)

G.add_node("StarBot Cafe", type = "Dining Outlet", zone = "Sci-Fi City",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Galactic Treats", type = "Food Cart", zone = "Sci-Fi City",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Frozen Fuel", type = "Food Cart", zone = "Sci-Fi City",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Planet Yen", type = "Food Cart", zone = "Sci-Fi City",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Transformers Supply Vault", type = "Retail", zone = "Sci-Fi City",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

# add edges within Sci-Fi City
G.add_edge("Restroom 3", "Accelerator", distance = 3)



#################
# Ancient Egypt #
#################

G.add_node("Revenge of the Mummy", type = "Ride", zone = "Ancient Egypt",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Treasure Hunters", type = "Ride", zone = "Ancient Egypt",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Restroom 4", type = "Restroom", zone = "Ancient Egypt",
           cleanliness = 90, usage = 30)

G.add_node("Oasis Spice Cafe", type = "Dining Outlet", zone = "Ancient Egypt",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Cairo Market", type = "Food Cart", zone = "Ancient Egypt",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Pharaoh's Dessert Oasis", type = "Food Cart", zone = "Ancient Egypt",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Carter's Curiosities", type = "Retail", zone = "Ancient Egypt",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

# add edges within Ancient Egypt
G.add_edge("Restroom 4", "Treasure Hunters", distance = 3)



##################
# The Lost World #
##################

G.add_node("Jurassic Park Rapids", type = "Ride", zone = "The Lost World",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Dino-Soarin", type = "Ride", zone = "The Lost World",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Canopy Flyer", type = "Ride", zone = "The Lost World",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("WaterWorld", type = "Seasonal", zone = "The Lost World",
           duration = 20, capacity = 200, popularity = 100,
           timeslots = {datetime.time(12, 45), datetime.time(15, 0), datetime.time(17, 15)},
           crowd_level = 60)

G.add_node("Restroom 5", type = "Restroom", zone = "The Lost World",
           cleanliness = 90, usage = 30)

G.add_node("Restroom 6", type = "Restroom", zone = "The Lost World",
           cleanliness = 90, usage = 30)

G.add_node("Discovery Food Court", type = "Dining Outlet", zone = "The Lost World",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Fossil Fuels", type = "Dining Outlet", zone = "The Lost World",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Mariner's Market", type = "Food Cart", zone = "The Lost World",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("Jungle Bites", type = "Food Cart", zone = "The Lost World",
           menu_variety = 5, popularity = 99, expected_wait_time = 3, staff = 1,
           crowd_level = 80, actual_wait_time = 3)

G.add_node("The Dino-Store", type = "Retail", zone = "Hollywood",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

# add edges within The Lost World
G.add_edge("Restroom 5", "Discovery Food Court", distance = 3)
G.add_edge("Restroom 6", "WaterWorld", distance = 3)


################
# Far Far Away #
################

G.add_node("Puss In Boots", type = "Ride", zone = "Far Far Away",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Magic Potion Spin", type = "Ride", zone = "Far Far Away",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Shrek 4D Adventure", type = "Ride", zone = "Far Far Away",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Enchanted Airways", type = "Ride", zone = "Far Far Away",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Donkey Live", type = "Ride", zone = "Far Far Away",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Fortune Favours The Furry", type = "Ride", zone = "Far Far Away",
           duration = 10, capacity = 40, popularity = 99, expected_wait_time = 15,
           crowd_level = 80, actual_wait_time = 10)

G.add_node("Restroom 7", type = "Restroom", zone = "Far Far Away",
           cleanliness = 90, usage = 30)

G.add_node("Friar's Good Food", type = "Dining Outlet", zone = "Far Far Away",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Goldilocks", type = "Dining Outlet", zone = "Far Far Away",
           menu_variety = 10, capacity = 150, popularity = 99, expected_wait_time = 20, staff = 12,
           crowd_level = 80, actual_wait_time = 30)

G.add_node("Fairy Godmother's Potion Shop", type = "Retail", zone = "Far Far Away",
           capacity = 150, popularity = 90, expected_wait_time = 10, staff = 3,
           crowd_level = 80, actual_wait_time = 10)

# add edges within New York
G.add_edge("Restroom 7", "Goldilocks", distance = 3)



# add edges, each number represents the node (to be updated)
# G.add_edge("Transformers", "Revenge of the Mummy", distance = 150)
# G.add_edge("Revenge of the Mummy", "Jurassic Park Rapids", distance = 200)
# G.add_edge("Jurassic Park Rapids", "Battlestar Galactica", distance = 250)
# G.add_edge("Battlestar Galactica", "Shrek 4D", distance = 100)
# G.add_edge("Shrek 4D", "Puss in Boots’ Giant Journey", distance = 80)
# G.add_edge("Puss in Boots’ Giant Journey", "WaterWorld", distance = 100)
# G.add_edge("WaterWorld", "Far Far Away Castle", distance = 150)
# G.add_edge("Hawker’s Market", "Toilets", distance = 30)
# G.add_edge("Jurassic Park Rapids", "The Lost World", distance = 120)
# G.add_edge("Revenge of the Mummy", "Ancient Egypt Maze", distance = 90)
# G.add_edge("The Lost World", "Ancient Egypt", distance = 170)
# G.add_edge("Battlestar Galactica", "Hawker’s Market", distance = 200)
# assume roads won"t be congested

# walkthrough: change parameters of nodes and edges so that the satisfactory score is maximised and time spent in total is minimised
# constrained range of values
# capacity and staff available at the venue
# peak hours, special events, seasonal variations

# visualisation of graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, font_weight="bold", edge_color="gray")
labels = nx.get_edge_attributes(G, "distance")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Universal Studios Singapore Attractions and Rides with Zones")
plt.Show()

# visualisation with plotly
pos = nx.spring_layout(G)
edge_x = []
edge_y = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
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
node_colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in G.nodes()]

for node, data in G.nodes(data=True):
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    
    # Basic name for node label
    node_text.append(node)
    
    # Detailed hover information
    tooltip_text = f"Name: {node}<br>Type: {data.get('type', 'N/A')}"
    tooltip_text += f"<br>Zone: {data.get('zone', 'N/A')}"
    tooltip_text += f"<br>Crowd Level: {data.get('crowd_level', 'N/A')}"
    tooltip_text += f"<br>Capacity: {data.get('capacity', 'N/A')}"
    tooltip_text += f"<br>Speed: {data.get('speed', 'N/A')}"
    tooltip_text += f"<br>Menu Variety: {data.get('menu_variety', 'N/A')}"
    tooltip_text += f"<br>Popularity: {data.get('popularity', 'N/A')}"
    tooltip_text += f"<br>Cleanliness: {data.get('cleanliness', 'N/A')}"
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
        size=8,               
        color="white"          
    ),
    marker=dict(
        size=90,               
        color=node_colors,
        line=dict(width=2)
    )
)


# Plotting the graph
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Theme Park Attractions",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig = go.Figure(data=[edge_trace, node_trace])  

# Save the figure to an HTML file
html_file = "graph_output.html"
pio.write_html(fig, file=html_file, auto_open=False)

# Open the HTML file in the default web browser
webbrowser.open_new_tab(html_file)
