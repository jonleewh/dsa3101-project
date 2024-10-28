import networkx as nx
import matplotlib.pyplot as plt

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
G.add_node('Transformers', type = 'ride', zone = 'Sci-Fi City', speed = 60, popularity = 95)
G.add_node('Battlestar Galactica', type = 'ride', zone = 'Sci-Fi City', speed = 80, popularity = 88)
G.add_node('Revenge of the Mummy', type = 'ride', zone = 'Ancient Egypt', speed = 70, popularity = 90)
G.add_node('Jurassic Park Rapids', type = 'ride', zone = 'The Lost World', speed = 50, popularity = 85)
G.add_node('Shrek 4D', type = 'ride', zone = 'Far Far Away', speed = 10, popularity = 75)
G.add_node('Far Far Away Castle', type = 'attraction', zone = 'Far Far Away', crowd_level = 70)
G.add_node('WaterWorld', type = 'show', zone = 'Hollywood', duration = 30, capacity=100, crowd_level=60)
G.add_node('Hawker’s Market', type = 'F&B', zone = 'New York', menu_variety=10, capacity=150, crowd_size=80)
G.add_node('Toilets', type = 'toilet', zone = 'Main Street', cleanliness=90, proximity=5, usage=30)

# add edges, each number represents the node
G.add_edge('Transformers', 'Revenge of the Mummy', distance=150)
G.add_edge('Revenge of the Mummy', 'Jurassic Park Rapids', distance=200)
G.add_edge('Jurassic Park Rapids', 'Battlestar Galactica', distance=250)
G.add_edge('Battlestar Galactica', 'Shrek 4D', distance=100)
G.add_edge('Shrek 4D', 'WaterWorld', distance=200)
G.add_edge('WaterWorld', 'Far Far Away Castle', distance=150)
G.add_edge('Hawker’s Market', 'Toilets', distance=30)

# visualisation of graph
nx.draw(G, with_labels=True)
plt.show()
