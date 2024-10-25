from collections import deque
import time

class Customer: # generate unique customers
    def __init__(self, id, arrival_time):
        self.id = id # unique
        self.fast_pass = None # 0 for no fast_pass, 1 for have fast_pass
        self.arrival_time = arrival_time
        self.start_time = None
        self.end_time = None

    def __repr__(self): # representation method 
        return f"Customer {self.id, self.fast_pass}"

class RideQueues:
    def __init__(self, ride_time, ride_capacity):
        self.regular_queue = deque()  
        self.fast_pass_queue = deque()
        self.ride_time = ride_time  # Time of each rides
        self.ride_capacity = ride_capacity # Number of max number of ppl a ride can take
        self.current_time = 0  # Simulation time in seconds
        self.total_served = 0  # Counter for served customers

    def add_customer(self, customer):
        if isinstance(customer, Customer):
            if customer.fast_pass == 1:
                self.fast_pass_queue.append(customer)
                print(f"{customer} added to the fast-pass queue at time {self.current_time}.") # for debugtest
            else:
                self.regular_queue.append(customer)
                print(f"{customer} added to the regular queue at time {self.current_time}.") # for debugtest
        else:
            print("Only Customer objects can be added to the queue.") # for debugtest
        

    def process_queue(self):
        while self.queue: # while queue has customers
            riding_customers = []
            for i in (self.ride_capacity):
                riding_customers.append(self.fast_pass_queue.popleft()) # Get customers in fast_pass_queue
                riding_customers.append(self.regular_queue.popleft()) # Get customers in regular_queue
                self.total_served += 1
            customer.start_time = self.current_time  # When customer starts the ride
            customer.end_time = self.current_time + self.ride_time
            self.current_time += self.ride_time  # Simulate time passing

            print(f"{customer} started at {customer.start_time} and finished at {customer.end_time}.") # for debugtest
            self.calculate_waiting_time(customer)

    def calculate_waiting_time(self, customer):
        """Calculate the waiting time for each customer"""
        waiting_time = customer.start_time - customer.arrival_time
        print(f"{customer} waited for {waiting_time} seconds.")


# Function to serve customers, giving priority to the fast-pass queue
def serve_next():
    if fast_pass_queue:
        # Serve from the fast-pass queue if it's not empty
        person = fast_pass_queue.popleft()
        print(f"{person} from Fast Pass Queue is being served.")
    elif regular_queue:
        # Serve from the regular queue if fast-pass queue is empty
        person = regular_queue.popleft()
        print(f"{person} from Regular Queue is being served.")
    else:
        print("No one is waiting in the queues.")

# Create two separate queues: one for regular customers and one for fast-pass customers
regular_queue = deque()
fast_pass_queue = deque()

# Adding people to the queues
regular_queue.append("Person 1 (Regular)")
regular_queue.append("Person 2 (Regular)")
fast_pass_queue.append("Person 3 (Fast Pass)")
fast_pass_queue.append("Person 4 (Fast Pass)")


# Usage example
ride_queue = RideQueue(ride_time=5)  # Each ride takes 5 seconds

# Simulate customers arriving at different times
customers = [Customer(id=i, arrival_time=i*2) for i in range(5)]  # Customers arriving every 2 seconds

# Add customers to the queue
for customer in customers:
    ride_queue.add_customer(customer)

# Process the queue and serve the customers
ride_queue.process_queue()








# Simulate serving people
serve_next()  # Person 3 (Fast Pass)
serve_next()  # Person 4 (Fast Pass)
serve_next()  # Person 1 (Regular)
serve_next()  # Person 2 (Regular)
serve_next()  # No one is waiting in the queues
