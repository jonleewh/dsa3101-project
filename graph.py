import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
import random

# ideal: try to make the graph interactive where the person can hover around the circle and view details about each attraction

# issues:
# 1. the name of each attraction should attempt to fit within the confines of the circle
# 2. Showing of key details of each attraction
# 3. Inputting of accurate data for each zone and each attraction
# 4. Enter nodes first followed by edges

# use plotly or bokeh

# Every 30 minutes update expected_duration? 10am, 10:30am, 11am, 11:30am, 12pm, ...., 5pm

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

# add nodes, ensure properties are in the format (key = value)
def dynamic_crowd_wait(time_of_day, base_wait, base_crowd):
    # Simulate peak crowd/wait times with sinusoidal pattern
    crowd_level = base_crowd + 20 * np.sin(time_of_day * np.pi / 12)
    wait_time = base_wait + 10 * np.sin(time_of_day * np.pi / 12)
    return max(0, wait_time), max(0, crowd_level)

# Calculate satisfaction score based on crowd level, wait time, and popularity
def calculate_satisfaction(wait_time, crowd_level, popularity):
    # Satisfaction score example: higher with low crowd/wait and high popularity
    return max(0, popularity - wait_time - crowd_level)

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
        total_time += wait_time + travel_time
        itinerary.append((attraction, wait_time, satisfaction))
        current_location = attraction
        
        # Simulate moving to the next hour
        current_hour = (current_hour + 1) % 24
    
    return itinerary, total_time

# Hollywood
G.add_node("Mel's Mixtape", type = "show", zone = "Hollywood", duration = 30, capacity=100, crowd_level=60)
G.add_node("Restroom1", type = "restroom", zone = "Hollywood", cleanliness = 90, usage = 30)
G.add_node("Restroom2", type = "restroom", zone = "Hollywood", cleanliness = 90, usage = 30)
G.add_node("Starbucks", type = "F&B", zone = "Hollywood", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Mel's Drive-In", type = "F&B", zone = "Hollywood", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("KT's Grill", type = "F&B", zone = "Hollywood", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Star Snacks", type = "F&B", zone = "Hollywood", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Pops! Popcorn Delight", type = "F&B", zone = "Hollywood", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("That's a Wrap!", type = "retail", zone = "Hollywood", crowd_size = 80)
G.add_node("Candylicious", type = "retail", zone = "Hollywood", crowd_size = 80)
G.add_node("Universal Studios Store", type = "retail", zone = "Hollywood", crowd_size = 80)
G.add_node("Hello Kitty Studio", type = "retail", zone = "Hollywood", crowd_size = 80)
G.add_node("Minion Mart", type = "retail", zone = "Hollywood", crowd_size = 80)
G.add_node("UNIVRS", type = "retail", zone = "Hollywood", crowd_size = 80)

# add edges within Hollywood
G.add_edge("Restroom")
G.add_edge()

# New York
G.add_node("Lights Camera Action Hosted by Steven Spielberg", type = "ride", zone = "New York",
           ride_duration = 10, capacity = 150, crowd_size = 80, popularity = 99)
G.add_node("Sesame Street Spaghetti Space Chase", type = "ride", zone = "New York",
           ride_duration = 10, capacity = 150, crowd_size = 80, popularity = 99)
G.add_node("Rhythm Truck", type = "show", zone = "New York", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Restroom", type = "restroom", zone = "New York", cleanliness = 90, usage = 30)
G.add_node("Loui's NY Pizza Parlor", type = "F&B", zone = "New York", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Big Bird's Emporium", type = "retail", zone = "New York", crowd_size = 80)

# Sci-Fi City
G.add_node("Transformers", type = "ride", zone = "Sci-Fi City",
           ride_duration = 5, capacity = 150, crowd_size = 80, popularity = 95)
G.add_node("Accelerator", type = "ride", zone = "Sci-Fi City", popularity = 90)
G.add_node("Battlestar Galactica: Human", type = "ride", zone = "Sci-Fi City", expected_duration = 30,
           crowd_level = 100, actual_duration = 60, popularity = 100) # values will change based on the time, every 30 min?
G.add_node("Battlestar Galactica: Cylon", type = "ride", zone = "Sci-Fi City", expected_duration = 30,
           crowd_level = 100, actual_duration = 60, popularity = 100) # values will change based on the time, every 30 min?
G.add_node("Restroom", type = "restroom", zone = "Sci-Fi City", cleanliness = 90, usage = 30)
G.add_node("StarBot Cafe", type = "F&B", zone = "Sci-Fi City", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Galactic Treats", type = "F&B", zone = "Sci-Fi City", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Frozen Fuel", type = "F&B", zone = "Sci-Fi City", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Planet Yen", type = "F&B", zone = "Sci-Fi City", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Transformers Supply Vault", type = "retail", zone = "Sci-Fi City", crowd_size = 80)

# Ancient Egypt
G.add_node("Revenge of the Mummy", type = "ride", zone = "Ancient Egypt", popularity = 100)
G.add_node("Treasure Hunters", type = "ride", zone = "Ancient Egypt", popularity = 90, crowd_level = 10)
G.add_node("Restroom", type = "restroom", zone = "Ancient Egypt", cleanliness = 90, usage = 30)
G.add_node("Oasis Spice Cafe", type = "F&B", zone = "Ancient Egypt", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Cairo Market", type = "F&B", zone = "Ancient Egypt", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Pharaoh's Dessert Oasis", type = "F&B", zone = "Ancient Egypt", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Carter's Curiosities", type = "retail", zone = "Ancient Egypt", crowd_size = 80)

# The Lost World
G.add_node("Jurassic Park Rapids", type = "ride", zone = "The Lost World", popularity = 85)
G.add_node("Dino-Soarin", type = "ride", zone = "The Lost World", popularity = 85)
G.add_node("Canopy Flyer", type = "ride", zone = "The Lost World", popularity = 85)
G.add_node("WaterWorld", type = "show", zone = "The Lost World", popularity = 80)
G.add_node("Restroom", type = "restroom", zone = "The Lost World", cleanliness = 90, usage = 30)
G.add_node("Restroom", type = "restroom", zone = "The Lost World", cleanliness = 90, usage = 30)
G.add_node("Discovery Food Court", type = "F&B", zone = "The Lost World", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Mariner's Market", type = "F&B", zone = "The Lost World", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Jungle Bites", type = "F&B", zone = "The Lost World", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Drying Pod", type = "drying pod", zone = "The Lost World", capacity = 5)
G.add_node("The Dino-Store", type = "retail", zone = "Hollywood", crowd_size = 80)

# Far Far Away
G.add_node("Puss In Boots", type = "ride", zone = "Far Far Away", popularity = 75)
G.add_node("Magic Potion Spin", type = "ride", zone = "Far Far Away", popularity = 75)
G.add_node("Shrek 4D Adventure", type = "ride", zone = "Far Far Away", popularity = 75)
G.add_node("Enchanted Airways", type = "ride", zone = "Far Far Away", crowd_level = 70)
G.add_node("Donkey Live", type = "ride", zone = "Far Far Away", crowd_level = 70)
G.add_node("Fortune Favours The Furry", type = "ride", zone = "Far Far Away", crowd_level = 70)
G.add_node("Restroom", type = "restroom", zone = "Far Far Away", cleanliness = 90, usage = 30)
G.add_node("Friar's Good Food", type = "F&B", zone = "Far Far Away", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Goldilocks", type = "F&B", zone = "Far Far Away", menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node("Fairy Godmother's Potion Shop", type = "retail", zone = "Far Far Away", crowd_size = 80)

# add edges, each number represents the node
G.add_edge("Transformers", "Revenge of the Mummy", distance = 150)
G.add_edge("Revenge of the Mummy", "Jurassic Park Rapids", distance = 200)
G.add_edge("Jurassic Park Rapids", "Battlestar Galactica", distance = 250)
G.add_edge("Battlestar Galactica", "Shrek 4D", distance = 100)
G.add_edge("Shrek 4D", "Puss in Boots’ Giant Journey", distance = 80)
G.add_edge("Puss in Boots’ Giant Journey", "WaterWorld", distance = 100)
G.add_edge("WaterWorld", "Far Far Away Castle", distance = 150)
G.add_edge("Hawker’s Market", "Toilets", distance = 30)
G.add_edge("Jurassic Park Rapids", "The Lost World", distance = 120)
G.add_edge("Revenge of the Mummy", "Ancient Egypt Maze", distance = 90)
G.add_edge("The Lost World", "Ancient Egypt", distance = 170)
G.add_edge("Battlestar Galactica", "Hawker’s Market", distance = 200)
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
plt.show()

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
    tooltip_text = f"Name: {node}<br>Type: {data.get("type", "N/A")}"
    tooltip_text += f"<br>Zone: {data.get("zone", "N/A")}"
    tooltip_text += f"<br>Crowd Level: {data.get("crowd_level", "N/A")}"
    tooltip_text += f"<br>Capacity: {data.get("capacity", "N/A")}"
    tooltip_text += f"<br>Speed: {data.get("speed", "N/A")}"
    tooltip_text += f"<br>Menu Variety: {data.get("menu_variety", "N/A")}"
    tooltip_text += f"<br>Popularity: {data.get("popularity", "N/A")}"
    tooltip_text += f"<br>Cleanliness: {data.get("cleanliness", "N/A")}"
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
