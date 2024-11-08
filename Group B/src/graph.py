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

########################
## Read in Graph Data ##
########################

nodes = pd.read_csv("../data/theme_park_nodes.csv")
edges = pd.read_csv("../data/theme_park_edges.csv")

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
ASK CHIRAG TMRW IF OUR IDEA ANSWERS THE QUESTION

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
popularity
staff
weather (indirect factor) --> Need to code which node is indoor/outdoor!


Supervised learning requires an output!
Run ML to determine the most important factor
If any variable is particularly important, how to improve the model?
"""

######################################
## Calculating Actual Waiting Times ##
######################################

# we can use the waiting time as a proxy for the crowd level using the csv file
# replace with the csv file data rather than generating it by ourselves
def waiting_time(time_of_day, ride_duration, crowd_level, popularity, staff, weather): # get the expected waiting time from the csv file
    # update this every 5 min
    # loop over the csv file to collect the data we want
    waiting_time = ride_duration + crowd_level + popularity + 0.5 * staff + 0.5 * weather
    return waiting_time

X = None # import csv file
y = None # generate wait times based on X

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


####################################
## Calculating Satisfaction Score ##
####################################

# Arguments are the properties of the node. Can put in the object

# Calculate satisfaction/desirability score based on crowd level, wait time, and popularity
# Suggestion: track how long a guest spends waiting, maximise satisfaction score AND minimise wait time.
def calculate_satisfaction(actual_wait_time, crowd_level, popularity, menu_variety, cleanliness, weather, ride_quality):
    satisfaction_score = (10 - 0.5 * actual_wait_time - 0.3 * crowd_level
                          + 0.2 * popularity + 0.2 * menu_variety
                          + 0.5 * cleanliness - 0.3 * weather
                          + 0.4 * ride_quality) # we input a first guess of the coefficients here first
    return satisfaction_score
# Satisfaction score refers to the score for a single NODE, not the visitors.
# but it should be based on the visitor!

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


# for seasonal variations, try to use things like Halloween
# an example is to have an increased percentage of staff
# (e.g. increase staff by 20%, what's the change in satisfaction? Consider the costs of doing this also)
# look at the DIFFERENCES in outcomes by increasing a variable by a certain percentage
# make a graph of this for business suggestions
# satisfaction score will never be 100 because some people will always have issues
# we can just choose the parameters with higher importance
# --> if we have 3 variables, we need to consider all possible combinations!? 3C1 + 3C2 + 3C3
# tweak some parameters for the dynamic queue --> e.g. staff deployment


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


"""
let's import data from the csv file instead of hard coding it
if the Ride doesn't exist in the csv file, keep it but
- current columns in csv file: index, name of attraction, duration, popularity score (Jamie is doing this)
- add columns like usage, crowd level, cleanliness etc
- cleanliness taken from the survey (justify this)
- crowd level from dynamic queue
- menu variety from survey
"""

###################################
## Creating Graph Data Structure ##
###################################

G = nx.Graph()

for node in nodes:
    G.add_node(node["name"], node["type"], node["zone"])

for edge in edges:
    G.add_edge(edge["source"], edge["target"], edge["distance"])

"""
Assumptions:
Roads are not congested, although the ideal is to reduce congestion.
Everyone walks at the same speed (this means that it takes the same time for anyone to get from point A to point B).
"""


###############################################
## Shortest-Path Optimisation using Dijkstra ##
###############################################
# run through all possible paths (from Jamie) to maximise satisfaction and minimise wait time by changing the values inside the nodes, NOT the path
# let's try to do this by Friday
def find_shortest_path(graph, start, end):
    return nx.dijkstra_path(graph, start, end, weight="distance")

# output in dictionary format, e.g. {an adhoc node : all adjacent nodes}

# try to use OOP to manipulate and decide what changes is best for the visitors
# need to have the parameters that we can change to do the ML iteration
# input: 

##########################
## Optimising Itinerary ##
##########################

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




###############################
## Visualisation with Plotly ##
###############################

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
