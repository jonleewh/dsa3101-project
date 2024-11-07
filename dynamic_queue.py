from collections import deque
import time

class Customer: # generate unique customers
    def __init__(self, id, arrival_time, fast_pass):
        self.id = id # unique
        self.fast_pass = fast_pass # 0 for no fast_pass, 1 for have fast_pass
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

    def advance_time(self):
        # Update the current time based on the ride time
        self.current_time += self.ride_time
        print(f"Time advanced to: {self.current_time} seconds.")

    def add_customer(self, customer):
        if isinstance(customer, Customer):
            if customer.arrival_time <= self.current_time:
                if customer.fast_pass == 1:
                    self.fast_pass_queue.append(customer)
                    print(f"{customer} added to the fast-pass queue at time {self.current_time}.") # for debugtest
                else:
                    self.regular_queue.append(customer)
                    print(f"{customer} added to the regular queue at time {self.current_time}.") # for debugtest
            else:
                print(f"{customer} has not yet arrived at time {self.current_time}.")
        else:
            print("Only Customer objects can be added to the queue.") # for debugtest
        

    def process_queue(self):
        # assumption: ride_capacity is EVEN number; fast-pass vs regular queue each take up 1/2 of ride_capacity
        # operation: take min(1/2*ride_capacity customeres out of fast_pass_queue, len(fast_pass_queue)), the rest from regular_queue
        
        riding_customers = []
        
        # take min(1/2*ride_capacity customeres out of fast_pass_queue, len(fast_pass_queue))
        if self.fast_pass_queue: # while fast_pass_queue has customers
            for i in range(min(self.ride_capacity//2, len(self.fast_pass_queue))):
                person = self.fast_pass_queue.popleft()
                riding_customers.append(person) # Get customers in fast_pass_queue
                print(f"{person} from Fast Pass Queue is being served.") # for debugging
                self.total_served += 1
                print("total serving so far:", self.total_served) # for debugging
        
        fast_pass_filled_ride = len(riding_customers) # number of fast_pass customers going to the current ride
        
        # the rest from regular_queue
        if self.regular_queue:
            for i in range(min(self.ride_capacity - fast_pass_filled_ride, len(self.regular_queue))):
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
            print(f"{customer} started at {customer.start_time} and finished at {customer.end_time}.") # for debugtest
            self.calculate_waiting_time(customer)
        
        # and lastly, we update the current time
        self.current_time += self.ride_time
        print(f"The time now is {self.current_time}.")

        return riding_customers
            

    def calculate_waiting_time(self, customer):
        customer.waiting_time = customer.start_time - customer.arrival_time
        print(f"{customer} waited for {customer.waiting_time} seconds.")





# TESTING HERE

def test_ride_queues():
    # Create a RideQueues instance
    ride_time = 10  # Each ride takes 10 seconds
    ride_capacity = 6  # Max capacity of the ride is 6 people
    ride_queues = RideQueues(ride_time, ride_capacity)

    # Create some customers
    customers = [
        Customer(id=1, arrival_time=0, fast_pass=0),  # Regular customer
        Customer(id=2, arrival_time=1, fast_pass=1),  # Fast-pass customer
        Customer(id=3, arrival_time=2, fast_pass=0),  # Regular customer
        Customer(id=4, arrival_time=3, fast_pass=1),  # Fast-pass customer
        Customer(id=5, arrival_time=4, fast_pass=0),  # Regular customer
        Customer(id=6, arrival_time=5, fast_pass=1),  # Fast-pass customer
        Customer(id=7, arrival_time=6, fast_pass=0),  # Regular customer
        Customer(id=8, arrival_time=7, fast_pass=1),  # Fast-pass customer
    ]

    # Add customers to the queues
    for customer in customers:
        ride_queues.add_customer(customer)

    # Process the queue for the first ride
    curr_list_of_cust = ride_queues.process_queue()
    print("\n--- End of Ride 1 ---\n")

    # update list of remaining customers:
    if curr_list_of_cust:
        customers = [customer for customer in customers if customer not in curr_list_of_cust]


    # Add customers to the queues
    for customer in customers:
        ride_queues.add_customer(customer)

    # Process the queue for the second ride
    curr_list_of_cust = ride_queues.process_queue()
    print("\n--- End of Ride 2 ---\n")

    # update list of remaining customers:
    if curr_list_of_cust:
        customers = [customer for customer in customers if customer not in curr_list_of_cust]


    # Add customers to the queues
    for customer in customers:
        ride_queues.add_customer(customer)

    # Process the queue for the third ride
    curr_list_of_cust = ride_queues.process_queue()
    print("\n--- End of Ride 3 ---\n")

    # update list of remaining customers:
    if curr_list_of_cust:
        customers = [customer for customer in customers if customer not in curr_list_of_cust]


    # Add customers to the queues
    for customer in customers:
        ride_queues.add_customer(customer)

    # Try processing when there are no customers left
    ride_queues.process_queue()
    print("\n--- End of Ride 4 ---\n")


# Run the test cases
test_ride_queues()