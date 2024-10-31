import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ideal: try to make the graph interactive where the person can hover around the circle and view details about each attraction

# issues:
# 1. the name of each attraction should attempt to fit within the confines of the circle
# 2. Showing of key details of each attraction
# 3. Inputting of accurate data for each zone and each attraction
# 4. Enter nodes first followed by edges

# use plotly or bokeh

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

# Simulate park experience over a day (time range: 0 to 24 hours)
for hour in range(24):
    print(f"\n--- Time: {hour}:00 ---")
    for node in G.nodes(data=True):
        base_wait = node[1]['base_wait']
        base_crowd = node[1]['base_crowd']
        popularity = node[1]['popularity']
        
        # Update wait time and crowd level dynamically
        wait_time, crowd_level = dynamic_crowd_wait(hour, base_wait, base_crowd)
        
        # Calculate satisfaction score
        satisfaction = calculate_satisfaction(wait_time, crowd_level, popularity)
        
        # Print status
        print(f"{node[0]} - Wait Time: {wait_time:.1f} mins, Crowd Level: {crowd_level:.1f}, Satisfaction: {satisfaction:.1f}")


# Shortest-path optimization based on Dijkstra’s algorithm
def find_shortest_path(graph, start, end):
    return nx.dijkstra_path(graph, start, end, weight='distance')

# Optimize itinerary
def optimized_itinerary(start, attractions_list, current_hour):
    total_time = 0
    itinerary = [start]
    current_location = start
    
    for attraction in attractions_list:
        # Calculate dynamic wait time and crowd level for attraction
        wait_time, crowd_level = dynamic_crowd_wait(current_hour, 
                                                    G.nodes[attraction]['base_wait'], 
                                                    G.nodes[attraction]['base_crowd'])
        
        # Satisfaction score
        satisfaction = calculate_satisfaction(wait_time, crowd_level, G.nodes[attraction]['popularity'])
        
        # Pathfinding to minimize travel time between attractions
        path = find_shortest_path(G, current_location, attraction)
        travel_time = sum(G.edges[path[i], path[i+1]]['distance'] for i in range(len(path)-1))
        
        # Update itinerary and time
        total_time += wait_time + travel_time
        itinerary.append((attraction, wait_time, satisfaction))
        current_location = attraction
        
        # Simulate moving to the next hour
        current_hour = (current_hour + 1) % 24
    
    return itinerary, total_time

# New York
G.add_node('Hawker’s Market', type = 'F&B', zone = 'New York', menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node('Toilet', type = 'toilet', zone = 'New York', cleanliness = 90, usage = 30)

# Hollywood
G.add_node('WaterWorld', type = 'show', zone = 'Hollywood', duration = 30, capacity=100, crowd_level=60)

# Sci-Fi City
G.add_node('Transformers', type = 'ride', zone = 'Sci-Fi City', popularity = 95)
G.add_node('Battlestar Galactica', type = 'ride', zone = 'Sci-Fi City', expected_duration = 30,
           crowd_level = 100, actual_duration = 60, popularity = 88) # values will change based on the time

# Every 30 minutes update expected_duration? 10am, 10:30am, 11am, 11:30am, 12pm, ...., 5pm

# Ancient Egypt
G.add_node('Revenge of the Mummy', type = 'ride', zone = 'Ancient Egypt', popularity = 90)

# The Lost World
G.add_node('Jurassic Park Rapids', type = 'ride', zone = 'The Lost World', popularity = 85)

# Far Far Away
G.add_node('Shrek 4D', type = 'ride', zone = 'Far Far Away', popularity = 75)
G.add_node('Far Far Away Castle', type = 'attraction', zone = 'Far Far Away', crowd_level = 70)

# Hollywood
G.add_node('Toilet', type = 'toilet', zone = 'Hollywood', cleanliness=90, proximity=5, usage=30)

# add edges, each number represents the node
G.add_edge('Transformers', 'Revenge of the Mummy', distance = 150)
G.add_edge('Revenge of the Mummy', 'Jurassic Park Rapids', distance = 200)
G.add_edge('Jurassic Park Rapids', 'Battlestar Galactica', distance = 250)
G.add_edge('Battlestar Galactica', 'Shrek 4D', distance = 100)
G.add_edge('Shrek 4D', 'Puss in Boots’ Giant Journey', distance = 80)
G.add_edge('Puss in Boots’ Giant Journey', 'WaterWorld', distance = 100)
G.add_edge('WaterWorld', 'Far Far Away Castle', distance = 150)
G.add_edge('Hawker’s Market', 'Toilets', distance = 30)
G.add_edge('Jurassic Park Rapids', 'The Lost World', distance = 120)
G.add_edge('Revenge of the Mummy', 'Ancient Egypt Maze', distance = 90)
G.add_edge('The Lost World', 'Ancient Egypt', distance = 170)
G.add_edge('Battlestar Galactica', 'Hawker’s Market', distance = 200)
# assume roads won't be congested

# walkthrough: change parameters of nodes and edges so that the satisfactory score is maximised and time spent in total is minimised
# constrained range of values
# capacity and staff available at the venue
# peak hours, special events, seasonal variations

# visualisation of graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')
labels = nx.get_edge_attributes(G, 'distance')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Universal Studios Singapore Attractions and Rides with Zones")
plt.show()
