import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

class Attraction:
    def init_(self, name, node_type, crowd_level):
        self.name = name
        self.node_type = node_type
        self.crowd_level = crowd_level
    
    # Method to update crowd Level
    def update_crowd_level(self, new_crowd):
        self.crowd_level = new_crowd
        print(f"(self.name) now has a crowd level of (self.crowd_level}")
    
    # Method to get details of the attraction
    def get_info(self):
        return
        # return f"{self.name) is a {self.node_type) with a current crowd level of {self.crowd)

# add nodes, ensure properties are in the format (key = value)
G.add_node('Roller Coaster', type = 'ride', speed = 100, popularity = 90, wait_time = 20)
G.add_node('Ferris Wheel', type = 'ride', speed = 10, popularity = 70, wait_time = 15)
G.add_node("Burger Stand", type = 'F&B', menu_variety = 5, capacity = 50, crowd_size = 30)
G.add_node('ToiletA', type = 'toilet', cleanliness = 80, proximity = 10, usage = 50)

# add edges, each number represents the node
G.add_edge('Roller Coaster', 'Ferris Wheel', distance = 200)
G.add_edge('Roller Coaster', 'Burger Stand', distance = 100)
G.add_edge('Ferris Wheel', 'ToiletA', distance = 50)
G.add_edge('ToiletA', 'Burger Stand', distance = 70)

# visualisation of graph
nx.draw(G, with_labels=True)
plt.show()
