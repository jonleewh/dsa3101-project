import datetime
import pandas as pd
import os

# For all F&B outlets, change menu_variety to affordability

nodes_data = [
    # Entrance
    {"name": "Entrance"},
    
    # Hollywood
    {"name": "Mel's Mixtape", "type": "Seasonal", "zone": "Hollywood",
     "duration": 20, "timeslots": [datetime.time(10, 35), datetime.time(13, 0), datetime.time(15, 0), datetime.time(17, 55)],
     "crowd_level": 60},

    {"name": "Margo, Edith and Agnes Meet-and-Greet", "type": "Seasonal", "zone": "Hollywood",
     "duration": 5, "timeslots": [datetime.time(10, 5), datetime.time(11, 55), datetime.time(13, 55), datetime.time(15, 55), datetime.time(17, 25)],
     "crowd_level": 60},

    {"name": "Illuminations Minion Monsters", "type": "Seasonal", "zone": "Hollywood",
     "duration": 5, "timeslots": [datetime.time(10, 15), datetime.time(12, 25), datetime.time(14, 5), datetime.time(15, 50)],
     "crowd_level": 60},

    {"name": "Restroom 1", "type": "Restroom", "zone": "Hollywood",
     "cleanliness": 84, "usage": 30},

    {"name": "Starbucks", "type": "Dining Outlet", "zone": "Hollywood",
     "affordability": 42.4, "capacity": 150, "crowd_level": 80, "actual_wait_time": 100},

    {"name": "Mel's Drive-In", "type": "Dining Outlet", "zone": "Hollywood",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20, "staff": 12,
     "crowd_level": 80, "actual_wait_time": 30},

    {"name": "KT's Grill", "type": "Dining Outlet", "zone": "Hollywood",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20, "staff": 12,
     "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Star Snacks", "type": "Food Cart", "zone": "Hollywood",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "Pops! Popcorn Delight", "type": "Food Cart", "zone": "Hollywood",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "That's a Wrap!", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Candylicious", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Universal Studios Store", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Hello Kitty Studio", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Minion Mart", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "UNIVRS", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    # New York
    {"name": "Lights Camera Action Hosted by Steven Spielberg", "type": "Ride", "zone": "New York",
     "duration": 10, "capacity": 150, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Sesame Street Spaghetti Space Chase", "type": "Ride", "zone": "New York",
     "duration": 10, "capacity": 20, "expected_wait_time": 10,
     "crowd_level": 80, "actual_wait_time": 5},

    {"name": "Rhythm Truck", "type": "Seasonal", "zone": "New York",
     "duration": 20, 
     "timeslots": [datetime.time(11, 30), datetime.time(13, 30), datetime.time(15, 30), datetime.time(18, 20)],
     "crowd_level": 60},

    {"name": "Restroom 2", "type": "Restroom", "zone": "New York",
     "cleanliness": 84, "usage": 30},

    {"name": "Loui's NY Pizza Parlor", "type": "Dining Outlet", "zone": "New York",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20, "staff": 12,
     "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Big Bird's Emporium", "type": "Retail", "zone": "New York",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    # Sci-Fi City
    {"name": "TRANSFORMERS The Ride: The Ultimate 3D Battle", "type": "Ride", "zone": "Sci-Fi City",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Accelerator", "type": "Ride", "zone": "Sci-Fi City",
     "duration": 10, "capacity": 30, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Battlestar Galactica: HUMAN", "type": "Ride", "zone": "Sci-Fi City",
     "duration": 3, "capacity": 20, "expected_wait_time": 15, "staff": 8,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Battlestar Galactica: CYLON", "type": "Ride", "zone": "Sci-Fi City",
     "duration": 3, "capacity": 20, "expected_wait_time": 15, "staff": 8,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Restroom 3", "type": "Restroom", "zone": "Sci-Fi City",
     "cleanliness": 84, "usage": 30},

    {"name": "StarBot Cafe", "type": "Dining Outlet", "zone": "Sci-Fi City",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20, "staff": 12,
     "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Galactic Treats", "type": "Food Cart", "zone": "Sci-Fi City",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "Frozen Fuel", "type": "Food Cart", "zone": "Sci-Fi City",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "The Sci-Fi Emporium", "type": "Retail", "zone": "Sci-Fi City",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},

    # Ancient Egypt
    {"name": "Revenge of the Mummy", "type": "Ride", "zone": "Ancient Egypt",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Treasure Hunters", "type": "Ride", "zone": "Ancient Egypt",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Restroom 4", "type": "Restroom", "zone": "Ancient Egypt",
     "cleanliness": 84, "usage": 30},

    {"name": "Oasis Spice Cafe", "type": "Dining Outlet", "zone": "Ancient Egypt",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20, "staff": 12,
     "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Cairo Market", "type": "Food Cart", "zone": "Ancient Egypt",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "Pharaoh's Dessert Oasis", "type": "Food Cart", "zone": "Ancient Egypt",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "Carter's Curiosities", "type": "Retail", "zone": "Ancient Egypt",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},
    
    # The Lost World
    {"name": "Jurassic Park Rapids Adventure", "type": "Ride", "zone": "The Lost World",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Dino Soarin'", "type": "Ride", "zone": "The Lost World",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Canopy Flyer", "type": "Ride", "zone": "The Lost World",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "WaterWorld", "type": "Seasonal", "zone": "The Lost World",
     "duration": 20, "capacity": 200, 
     "timeslots": [datetime.time(12, 45), datetime.time(15, 0), datetime.time(17, 15)],
     "crowd_level": 60},

    {"name": "Restroom 5", "type": "Restroom", "zone": "The Lost World",
     "cleanliness": 84, "usage": 30},

    {"name": "Restroom 6", "type": "Restroom", "zone": "The Lost World",
     "cleanliness": 84, "usage": 30},

    {"name": "Discovery Food Court", "type": "Dining Outlet", "zone": "The Lost World",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20,
     "staff": 12, "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Fossil Fuels", "type": "Dining Outlet", "zone": "The Lost World",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20,
     "staff": 12, "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Mariner's Market", "type": "Food Cart", "zone": "The Lost World",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "Jungle Bites", "type": "Food Cart", "zone": "The Lost World",
     "affordability": 5, "expected_wait_time": 3, "staff": 1,
     "crowd_level": 80, "actual_wait_time": 3},

    {"name": "The Dino-Store", "type": "Retail", "zone": "Hollywood",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10},
    
    # Far Far Away
    {"name": "Puss In Boots Giant Journey", "type": "Ride", "zone": "Far Far Away",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Magic Potion Spin", "type": "Ride", "zone": "Far Far Away",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Shrek 4D Adventure", "type": "Ride", "zone": "Far Far Away",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Enchanted Airways", "type": "Ride", "zone": "Far Far Away",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Donkey Live", "type": "Ride", "zone": "Far Far Away",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Fortune Favours The Furry", "type": "Ride", "zone": "Far Far Away",
     "duration": 10, "capacity": 40, "expected_wait_time": 15,
     "crowd_level": 80, "actual_wait_time": 10},

    {"name": "Restroom 7", "type": "Restroom", "zone": "Far Far Away",
     "cleanliness": 84, "usage": 30},

    {"name": "Friar's Good Food", "type": "Dining Outlet", "zone": "Far Far Away",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20,
     "staff": 12, "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Goldilocks", "type": "Dining Outlet", "zone": "Far Far Away",
     "affordability": 42.4, "capacity": 150, "expected_wait_time": 20,
     "staff": 12, "crowd_level": 80, "actual_wait_time": 30},

    {"name": "Fairy Godmother's Potion Shop", "type": "Retail", "zone": "Far Far Away",
     "capacity": 150, "expected_wait_time": 10, "staff": 3,
     "crowd_level": 80, "actual_wait_time": 10}
    
]


edges_data = [
    # Entrance
    {"source": "Universal Studios Store", "target": "Entrance", "distance": 2},
    {"source": "Restroom 1", "target": "Universal Studios Store", "distance": 1},
    
    # Hollywood Zone
    {"source": "Universal Studios Store", "target": "Hello Kitty", "distance": 1},
    {"source": "Universal Studios Store", "target": "Pops Popcorn", "distance": 2},
    {"source": "Hello Kitty", "target": "Pops Popcorn", "distance": 2},
    {"source": "Minion Mart", "target": "Pops Popcorn", "distance": 2},
    {"source": "Minion Mart", "target": "Starbucks", "distance": 1},
    {"source": "Starbucks", "target": "Pops Popcorn", "distance": 1},
    {"source": "Starbucks", "target": "Candylicious", "distance": 1},
    {"source": "Candylicious", "target": "Pops Popcorn", "distance": 2},
    {"source": "Hello Kitty", "target": "Mel's Drive-In", "distance": 5},
    {"source": "Minion Mart", "target": "Mel's Drive-In", "distance": 4},
    {"source": "Restroom 1", "target": "Mel's Drive-In", "distance": 1},
    {"source": "Restroom 1", "target": "Star Snacks", "distance": 2},
    {"source": "Restroom 1", "target": "KT's Grill", "distance": 1},
    {"source": "KT's Grill", "target": "UNIVRS", "distance": 3},
    {"source": "Star Snacks", "target": "UNIVRS", "distance": 3},
    {"source": "Mel's Drive-In", "target": "UNIVRS", "distance": 4},
    {"source": "Mel's Drive-In", "target": "Mel's Mixtape", "distance": 1},
    {"source": "Mel's Drive-In", "target": "UNIVRS", "distance": 3},
    
    # link between Hollywood and New York
    {"source": "Mel's Drive-In", "target": "Lights Camera Action Hosted by Steven Spielberg", "distance": 2},
    
    # New York Zone
    {"source": "Lights Camera Action Hosted by Steven Spielberg", "target": "Sesame Street Spaghetti Space Chase", "distance": 2},
    {"source": "Lights Camera Action Hosted by Steven Spielberg", "target": "Rhythm Truck", "distance": 3},
    {"source": "Lights Camera Action Hosted by Steven Spielberg", "target": "Restroom 2", "distance": 4},
    {"source": "Lights Camera Action Hosted by Steven Spielberg", "target": "Loui's NY Pizza Parlor", "distance": 5},
    {"source": "Sesame Street Spaghetti Space Chase", "target": "Rhythm Truck", "distance": 2},
    {"source": "Sesame Street Spaghetti Space Chase", "target": "Restroom 2", "distance": 3},
    {"source": "Sesame Street Spaghetti Space Chase", "target": "Big Bird's Emporium", "distance": 4},
    {"source": "Rhythm Truck", "target": "Restroom 2", "distance": 2},
    {"source": "Rhythm Truck", "target": "Loui's NY Pizza Parlor", "distance": 3},
    {"source": "Restroom 2", "target": "Loui's NY Pizza Parlor", "distance": 2},
    {"source": "Restroom 2", "target": "Big Bird's Emporium", "distance": 4},
    {"source": "Loui's NY Pizza Parlor", "target": "Big Bird's Emporium", "distance": 3},
    
    # link between New York and Sci-Fi City
    {"source": "Big Bird's Emporium", "target": "TRANSFORMERS The Ride: The Ultimate 3D Battle", "distance": 2},
    
    # Sci-Fi City Zone
    {"source": "TRANSFORMERS The Ride: The Ultimate 3D Battle", "target": "Accelerator", "distance": 3},
    {"source": "TRANSFORMERS The Ride: The Ultimate 3D Battle", "target": "Battlestar Galactica: HUMAN", "distance": 5},
    {"source": "TRANSFORMERS The Ride: The Ultimate 3D Battle", "target": "Restroom 3", "distance": 4},
    {"source": "TRANSFORMERS The Ride: The Ultimate 3D Battle", "target": "StarBot Cafe", "distance": 6},
    {"source": "Accelerator", "target": "Battlestar Galactica: CYLON", "distance": 4},
    {"source": "Accelerator", "target": "Restroom 3", "distance": 2},
    {"source": "Battlestar Galactica: HUMAN", "target": "Battlestar Galactica: CYLON", "distance": 1},
    {"source": "Battlestar Galactica: HUMAN", "target": "Restroom 3", "distance": 5},
    {"source": "Battlestar Galactica: HUMAN", "target": "Frozen Fuel", "distance": 6},
    {"source": "Restroom 3", "target": "StarBot Cafe", "distance": 3},
    {"source": "Restroom 3", "target": "Galactic Treats", "distance": 4},
    {"source": "StarBot Cafe", "target": "Galactic Treats", "distance": 2},
    {"source": "StarBot Cafe", "target": "Frozen Fuel", "distance": 3},
    {"source": "StarBot Cafe", "target": "Planet Yen", "distance": 4},
    {"source": "Galactic Treats", "target": "Frozen Fuel", "distance": 2},
    {"source": "Galactic Treats", "target": "Transformers Supply Vault", "distance": 4},
    {"source": "Frozen Fuel", "target": "Transformers Supply Vault", "distance": 3},
    {"source": "Planet Yen", "target": "Transformers Supply Vault", "distance": 5},
    
    # link between Sci-Fi City and Ancient Egypt
    {"source": "Transformers Supply Vault", "target": "Revenge of the Mummy", "distance": 2},
    
    # Ancient Egypt Zone
    {"source": "Revenge of the Mummy", "target": "Treasure Hunters", "distance": 4},
    {"source": "Revenge of the Mummy", "target": "Restroom 4", "distance": 3},
    {"source": "Revenge of the Mummy", "target": "Oasis Spice Cafe", "distance": 5},
    {"source": "Treasure Hunters", "target": "Restroom 4", "distance": 2},
    {"source": "Treasure Hunters", "target": "Oasis Spice Cafe", "distance": 4},
    {"source": "Treasure Hunters", "target": "Cairo Market", "distance": 6},
    {"source": "Restroom 4", "target": "Oasis Spice Cafe", "distance": 3},
    {"source": "Restroom 4", "target": "Pharaoh's Dessert Oasis", "distance": 5},
    {"source": "Oasis Spice Cafe", "target": "Cairo Market", "distance": 2},
    {"source": "Oasis Spice Cafe", "target": "Pharaoh's Dessert Oasis", "distance": 3},
    {"source": "Cairo Market", "target": "Pharaoh's Dessert Oasis", "distance": 1},
    {"source": "Pharaoh's Dessert Oasis", "target": "Carter's Curiosities", "distance": 4},
    {"source": "Cairo Market", "target": "Carter's Curiosities", "distance": 6},
    
    # link between Ancient Egypt and The Lost World
    {"source": "Cairo Market", "target": "Discovery Food Court", "distance": 3},
    
    # The Lost World Zone
    {"source": "Jurassic Park Rapids Adventure", "target": "Dino Soarin'", "distance": 3},
    {"source": "Jurassic Park Rapids Adventure", "target": "Restroom 5", "distance": 4},
    {"source": "Jurassic Park Rapids Adventure", "target": "Discovery Food Court", "distance": 5},
    {"source": "Dino Soarin'", "target": "Canopy Flyer", "distance": 2},
    {"source": "Dino Soarin'", "target": "Restroom 6", "distance": 3},
    {"source": "Dino Soarin'", "target": "Fossil Fuels", "distance": 5},
    {"source": "Canopy Flyer", "target": "WaterWorld", "distance": 4},
    {"source": "Canopy Flyer", "target": "Restroom 5", "distance": 5},
    {"source": "WaterWorld", "target": "Restroom 5", "distance": 2},
    {"source": "WaterWorld", "target": "Discovery Food Court", "distance": 6},
    {"source": "Restroom 5", "target": "Discovery Food Court", "distance": 3},
    {"source": "Restroom 6", "target": "Fossil Fuels", "distance": 2},
    {"source": "Discovery Food Court", "target": "Mariner's Market", "distance": 2},
    {"source": "Discovery Food Court", "target": "Jungle Bites", "distance": 3},
    {"source": "Mariner's Market", "target": "Jungle Bites", "distance": 1},
    {"source": "Fossil Fuels", "target": "The Dino-Store", "distance": 7},
    {"source": "The Dino-Store", "target": "Restroom 6", "distance": 6},
    {"source": "Restroom 7", "target": "Goldilocks", "distance": 3},
    
    # link between The Lost World and Far Far Away
    {"source": "Goldilocks", "target": "Revenge of the Mummy", "distance": 2},
    
    # Far Far Away Zone
    {"source": "Puss In Boots Giant Journey", "target": "Magic Potion Spin", "distance": 2},
    {"source": "Puss In Boots Giant Journey", "target": "Enchanted Airways", "distance": 3},
    {"source": "Magic Potion Spin", "target": "Shrek 4D Adventure", "distance": 2},
    {"source": "Shrek 4D Adventure", "target": "Enchanted Airways", "distance": 2},
    {"source": "Enchanted Airways", "target": "Donkey Live", "distance": 4},
    {"source": "Donkey Live", "target": "Fortune Favours The Furry", "distance": 3},
    {"source": "Shrek 4D Adventure", "target": "Friar's Good Food", "distance": 5},
    {"source": "Friar's Good Food", "target": "Goldilocks", "distance": 1},
    {"source": "Goldilocks", "target": "Fairy Godmother's Potion Shop", "distance": 2},
    {"source": "Fairy Godmother's Potion Shop", "target": "Restroom 7", "distance": 3},
    
    # link between Far Far Away and Hollywood
    {"source": "Fairy Godmother's Potion Shop", "target": "Mel's Mixtape", "distance": 10},
    
]

# Convert timeslot sets to string for CSV compatibility
for node in nodes_data:
    if "timeslots" in node:
        node["timeslots"] = ", ".join([t.strftime("%H:%M") for t in node["timeslots"]])

# Convert to DataFrames for easier CSV export
nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)

# Save to CSV
os.chdir('Group B/src')
nodes_df.to_csv("../data/theme_park_nodes.csv", index=False)
edges_df.to_csv("../data/theme_park_edges.csv", index=False)
