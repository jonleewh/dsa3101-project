import networkx as nx
import matplotlib.pyplot as plt

# ideal: try to make the graph interactive where the person can hover around the circle and view details about each attraction

# issues:
# 1. the name of each attraction should attempt to fit within the confines of the circle
# 2. Showing of key details of each attraction
# 3. Inputting of accurate data for each zone and each attraction
# 4. Enter nodes first followed by edges

G = nx.Graph()

class Attraction:
    def __init__(self, name, node_type, zone, crowd_level):
        self.name = name
        self.node_type = node_type
        self.zone = zone
        self.crowd_level = crowd_level
    
    # Method to update crowd level
    def update_crowd_level(self, new_crowd):
        self.crowd_level = new_crowd
        print(f"{self.name} now has a crowd level of {self.crowd_level}")
    
    # Method to get details of the attraction
    def get_info(self):
        print(f"{self.name} is a {self.node_type} in {self.zone} with a current crowd level of {self.crowd_level}")

# add nodes, ensure properties are in the format (key = value)

# New York
G.add_node('Hawker’s Market', type = 'F&B', zone = 'New York', menu_variety = 10, capacity = 150, crowd_size = 80)
G.add_node('Toilet', type = 'toilet', zone = 'New York', cleanliness = 90, usage=30)

# Hollywood
G.add_node('WaterWorld', type = 'show', zone = 'Hollywood', duration = 30, capacity=100, crowd_level=60)

# Sci-Fi City
G.add_node('Transformers', type = 'ride', zone = 'Sci-Fi City', speed = 60, popularity = 95)
G.add_node('Battlestar Galactica', type = 'ride', zone = 'Sci-Fi City', speed = 80, popularity = 88)

# Ancient Egypt
G.add_node('Revenge of the Mummy', type = 'ride', zone = 'Ancient Egypt', speed = 70, popularity = 90)

# The Lost World
G.add_node('Jurassic Park Rapids', type = 'ride', zone = 'The Lost World', speed = 50, popularity = 85)

# Far Far Away
G.add_node('Shrek 4D', type = 'ride', zone = 'Far Far Away', speed = 10, popularity = 75)
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


# visualisation of graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')
labels = nx.get_edge_attributes(G, 'distance')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Universal Studios Singapore Attractions and Rides with Zones")
plt.show()
