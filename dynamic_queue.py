from collections import deque
import time

class Customer: # generate unique customers
    def __init__(self, id, arrival_time):
        self.id = id # unique
        self.fast_pass = None # 0 for no fast_pass, 1 for have fast_pass
        self.arrival_time = arrival_time # when they enter the queue
        self.start_time = None # when they start the ride
        self.end_time = None # when they finish the ride
        self.waiting_time = None # waiting time before they get served

    def __repr__(self): # representation method 
        return f"Customer {self.id, self.fast_pass}"

class RideQueues:
    def __init__(self, ride_time, ride_capacity):
        self.regular_queue = deque()  
        self.fast_pass_queue = deque()
        self.ride_time = ride_time  # Time of each ride
        self.ride_capacity = ride_capacity # Max number of ppl a ride can take
        self.current_time = 0  # Simulation time in seconds
        self.total_served = 0  # Counter for served customers WITHIN WHICH TIME FRAME? ONE RIDE?

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
        # assumption: ride_capacity is EVEN number; fast-pass vs regular queue each take up 1/2 of ride_capacity
        # operation: take min(1/2*ride_capacity customeres out of fast_pass_queue, len(fast_pass_queue)), the rest from regular_queue
        
        riding_customers = []
        
        # take min(1/2*ride_capacity customeres out of fast_pass_queue, len(fast_pass_queue))
        if self.fast_pass_queue: # while fast_pass_queue has customers
            for i in range(1, min(self.ride_capacity//2, len(fast_pass_queue))):
                person = self.fast_pass_queue.popleft()
                riding_customers.append(person) # Get customers in fast_pass_queue
                print(f"{person} from Fast Pass Queue is being served.") # for debugging
                self.total_served += 1
                print("total serving so far:", self.total_served) # for debugging
        
        fast_pass_filled_ride = len(riding_customers) # number of fast_pass customers going to the current ride
        
        # the rest from regular_queue
        if self.regular_queue:
            for i in range(1, self.ride_capacity - fast_pass_filled_ride):
                person = self.regular_queue.popleft()
                riding_customers.append(person) # Get customers in regular_queue
                print(f"{person} from Regular Queue is being served.") # for debugging
                self.total_served += 1
                print("total serving so far:", self.total_served) # for debugging
        
        else:
            print("No one is waiting in the queues.")
        
        # now we update the time
        for customer in riding_customers: 
            customer.start_time = self.current_time  # When customer starts the ride
            customer.end_time = self.current_time + self.ride_time
            self.current_time += self.ride_time  # Simulate time passing
            print(f"{customer} started at {customer.start_time} and finished at {customer.end_time}.") # for debugtest
            self.calculate_waiting_time(customer)
            

    def calculate_waiting_time(self, customer):
        customer.waiting_time = customer.start_time - customer.arrival_time
        print(f"{customer} waited for {waiting_time} seconds.")


