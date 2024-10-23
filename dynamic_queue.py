from collections import deque
import time

class Customer:
    def __init__(self, id, arrival_time):
        self.id = id
        self.arrival_time = arrival_time
        self.start_time = None
        self.end_time = None

    def __repr__(self):
        return f"Customer {self.id}"

class RideQueue:
    def __init__(self, service_time_per_customer):
        self.queue = deque()  # Queue to hold customers
        self.service_time_per_customer = service_time_per_customer  # Time to process each customer
        self.current_time = 0  # Simulation time in seconds
        self.total_served = 0  # Counter for served customers

    def add_customer(self, customer):
        """Add customer to the queue"""
        print(f"{customer} added to the queue at time {self.current_time}.")
        self.queue.append(customer)

    def process_queue(self):
        """Serve customers in the queue based on the service time"""
        while self.queue:
            customer = self.queue.popleft()  # Get the first customer in the queue
            customer.start_time = self.current_time  # When customer starts the ride
            customer.end_time = self.current_time + self.service_time_per_customer
            self.current_time += self.service_time_per_customer  # Simulate time passing
            self.total_served += 1

            print(f"{customer} started at {customer.start_time} and finished at {customer.end_time}.")
            self.calculate_waiting_time(customer)

    def calculate_waiting_time(self, customer):
        """Calculate the waiting time for each customer"""
        waiting_time = customer.start_time - customer.arrival_time
        print(f"{customer} waited for {waiting_time} seconds.")

# Usage example
ride_queue = RideQueue(service_time_per_customer=5)  # Each ride takes 5 seconds

# Simulate customers arriving at different times
customers = [Customer(id=i, arrival_time=i*2) for i in range(5)]  # Customers arriving every 2 seconds

# Add customers to the queue
for customer in customers:
    ride_queue.add_customer(customer)

# Process the queue and serve the customers
ride_queue.process_queue()
