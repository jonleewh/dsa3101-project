import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
import os
from PIL import Image

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
    "Entrance": (5, 0)
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
