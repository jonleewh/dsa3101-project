#### Import relevant packages/libraries/data

from collections import deque
import math
from datetime import datetime, timedelta
import pandas as pd
import copy
import random


########################################################################################################################

class Visitor:
    def __init__(self, id, itinerary, fast_pass: int, arrival_time, attraction_generator_df):
        """
        :param id: a unique ID
        :param itinerary: a list of itinerary (where they will go, in sequence)
        :param fast_pass: 0 for regular ticket, 1 for fast-pass ticket
        """

        self.id = id
        self.fast_pass = fast_pass 
        self.itinerary = copy.deepcopy(itinerary) 
        self.attraction_generator_df = attraction_generator_df 



        self.current_time = arrival_time # First set to when they are 'spawned'
        self.current_location = "Entrance"
        self.next_location = "Entrance"

        self.status = "none" # "moving", "queuing", "being served", "none", "done", "completed"
        self.count_down = 0 # Count down to next location update, use when travelling
        self.service_count_down = 0
        self.queuing = 0 # time (in minutes) spent queueing at one single attraction

    def __repr__(self):
        return f"Visitor {self.id, self.fast_pass}"

    def advance_time(self):
        self.current_time += timedelta(minutes=1)

    def find_next_location(self):
        if not self.itinerary: # finish everything
            self.status = "completed"
            self.next_location = None # flag for find-count_down()
            print(f"{self} has completed their itinerary.")
            return  # Exit early
        
        self.status = "moving"
        print(f"The current itinerary for {self} is {self.itinerary}.")
        
        choice_index = self.itinerary.pop(0)
        print(f"The current choice index for {self} is {choice_index}.")

        choices = self.attraction_generator_df[self.attraction_generator_df['index'] == choice_index]['name']
        if not choices.empty:
            options = random.choices(choices.tolist(),k=1) # random.choices() return a list
            if options:
                self.next_location = options[0]
                print(f"The next location for {self} is {self.next_location}.")
            else:
                self.status = "completed"
                self.next_location = None # flag for find-count_down()
        else:
            self.status = "completed"
            self.next_location = None

    def find_count_down(self):
        # Check if there is a next location before proceeding
        if self.next_location is None:
            print(f"{self} has no next location to move towards.")
            return  # Exit early if no next location

        matching_rows = paths_df[(paths_df['source'] == self.current_location) & (paths_df['target'] == self.next_location)]
        if not matching_rows.empty:
            self.count_down = matching_rows['distance'].iloc[0]
            self.status = "moving"
            print(f"The current moving count down time is {self.count_down}.")
            
########################################################################################################################

class Entrance:
    def __init__(self):
        self.name = "Entrance"
        self.serving_visitors = []

    def add_visitor(self, visitor):
        self.serving_visitors.append(visitor)
        visitor.status = "being served"
        visitor.service_count_down = 0
        print(f"{visitor} is at the Entrance at time {visitor.current_time}.")

    def process(self):
        if self.serving_visitors:
            for visitor in self.serving_visitors:
                visitor.status = "done" 
                print(f"{visitor} is prepared to leave the Entrance at time {visitor.current_time}.")
        self.serving_visitors = [] # reset

########################################################################################################################

class Attraction: # Including shows, rides
    def __init__(self, name: str, ride_duration: int, ride_capacity: int):
        """
        Initialize an attraction with normal and express queues.
        :param name: Name of the attraction.
        :param ride_duration: Duration of the ride
        :param ride_capacity: Number of visitors that can be processed at once.
        """
        self.name = name

        self.regular_queue = deque()  
        self.fast_pass_queue = deque()

        self.ride_duration = int(ride_duration)  
        self.ride_capacity = ride_capacity 

        self.current_time = None  

        self.riding_visitors = [] # list of visitors being served in the current ride

        self.total_served = 0  
        self.rides_served = 0 
    
        self.in_service = False # True if ride is happening
        self.last_ride_time = datetime.strptime("10:00", "%H:%M")


    def __repr__(self): # representation method 
        return f"{self.name}"
 
    def advance_time(self):
        self.current_time += timedelta(minutes=1)

    def add_visitor(self, visitor: Visitor):
        visitor.status = "queuing"
        if visitor.fast_pass == 1:
            self.fast_pass_queue.append(visitor)
            print(f"{visitor} added to the fast-pass queue for {self} at time {self.current_time}.") # for debugtest
        else:
            self.regular_queue.append(visitor)
            print(f"{visitor} added to the regular queue for {self} at time {self.current_time}.") # for debugtest

    def process_queue(self):
        # take min(1/2*ride_capacity visitors out of fast_pass_queue, len(fast_pass_queue))
        if self.fast_pass_queue: # while fast_pass_queue has visitors
            self.fast_pass_queue = deque(visitor for visitor in self.fast_pass_queue if visitor.status != 'done')
            for _ in range(int(min(self.ride_capacity//2, len(self.fast_pass_queue)))):
                person = self.fast_pass_queue.popleft()
                if person not in self.riding_visitors:
                    self.riding_visitors.append(person) # Get visitors in fast_pass_queue
                    print(f"{person} from Fast Pass Queue for {self} is being served at {self.current_time}.") # for debugging
                    self.total_served += 1
                    print(f"{self} 's total serving so far: {self.total_served}")  # for debugging
        
        fast_pass_filled_ride = len(self.riding_visitors) # number of fast_pass visitors going to the current ride
        
        # the rest from regular_queue
        if self.regular_queue:
            self.regular_queue = deque(visitor for visitor in self.regular_queue if visitor.status != 'done')
            for _ in range(int(min(self.ride_capacity - fast_pass_filled_ride, len(self.regular_queue)))):
                person = self.regular_queue.popleft()
                if person not in self.riding_visitors:
                    self.riding_visitors.append(person) # Get visitors in regular_queue
                    print(f"{person} from Regular Queue for {self} is being served at {self.current_time}.") # for debugging
                    self.total_served += 1
                    print(f"{self} 's total serving so far: {self.total_served}")  # for debugging
        
        if not self.fast_pass_queue and not self.regular_queue:
                print(f"No one is waiting in the queues for {self} at time", self.current_time)


    def process_ride(self): 
        if self.current_time == self.last_ride_time + timedelta(minutes=self.ride_duration): # time to start a ride
            self.rides_served += 1
            self.last_ride_time = self.current_time 
            if self.riding_visitors:
                for visitor in self.riding_visitors:
                    visitor.status = "done" # finish the ride
                    print(f"{visitor} is prepared to leave {self} at time {visitor.current_time}.")
                self.riding_visitors = [] # reset
            self.process_queue()


    def get_data(self): 
        if math.isnan(self.ride_duration):
            self.ride_duration = 0
        crowd_level = {'fast_pass_queue':len(self.fast_pass_queue), 
                       'regular_queue':len(self.regular_queue)}
        curr_wait_time = {'fast_pass_queue':math.floor(crowd_level['fast_pass_queue']/self.ride_capacity)*self.ride_duration, 
                          'regular_queue':math.floor(crowd_level['regular_queue']/self.ride_capacity)*self.ride_duration}
        return self.current_time, crowd_level, curr_wait_time

########################################################################################################################

class Seasonal: # Including shows, rides, has fixed timing
    def __init__(self, name: str, ride_duration: int, ride_capacity: int, timeslot):
        """
        Initialize an attraction with normal and express queues.
        :param name: Name of the attraction.
        :param ride_duration: Duration of the ride
        :param ride_capacity: Number of visitors that can be processed at once.
        :param timeslot: schedule of the seasonal attraction, list
        """
        self.name = name
        self.regular_queue = deque()  
        self.fast_pass_queue = deque()
        self.ride_duration = int(ride_duration)  
        self.ride_capacity = ride_capacity 
        self.riding_visitors = []
        self.timeslot = timeslot
        self.endings = [item+timedelta(minutes=self.ride_duration) for item in timeslot]
        self.current_time = None
        self.total_served = 0  
        self.rides_served = 0 
    
    def __repr__(self): 
        return f"{self.name}"
    
    def advance_time(self):
        self.current_time += timedelta(minutes=1)


    def add_visitor(self, visitor: Visitor):
        if visitor.current_time > self.timeslot[-1]: # if visitor arrives after the last timeslot:
            visitor.status = "done"
            print(f"{visitor} 's arrival is after {self} 's last show, they need to proceed wth the next item on their itinerary.")
        else:
            visitor.status = "queuing"
            if visitor.fast_pass == 1:
                self.fast_pass_queue.append(visitor)
                print(f"{visitor} added to the fast-pass queue for {self} at time {self.current_time}.") # for debugtest
            else:
                self.regular_queue.append(visitor)
                print(f"{visitor} added to the regular queue for {self} at time {self.current_time}.") # for debugtest

    def process_queue(self):
        # take min(1/2*ride_capacity visitors out of fast_pass_queue, len(fast_pass_queue))
        if self.fast_pass_queue: # while fast_pass_queue has visitors
            self.fast_pass_queue = deque(visitor for visitor in self.fast_pass_queue if visitor.status != 'done')
            for _ in range(int(min(self.ride_capacity//2, len(self.fast_pass_queue)))):
                person = self.fast_pass_queue.popleft()
                if person not in self.riding_visitors:
                    self.riding_visitors.append(person) # Get visitors in fast_pass_queue
                    person.status = "being served"
                    print(f"{person} from Fast Pass Queue for {self} is being served at {self.current_time}.") # for debugging
                    self.total_served += 1
                    print(f"{self} 's total serving so far: {self.total_served}")  # for debugging
        
        fast_pass_filled_ride = len(self.riding_visitors) # number of fast_pass visitors going to the current ride
        
        # the rest from regular_queue
        if self.regular_queue:
            self.regular_queue = deque(visitor for visitor in self.regular_queue if visitor.status != 'done')
            for _ in range(int(min(self.ride_capacity - fast_pass_filled_ride, len(self.regular_queue)))):
                person = self.regular_queue.popleft()
                if person not in self.riding_visitors:
                    self.riding_visitors.append(person) # Get visitors in regular_queue
                    person.status = "being served"
                    print(f"{person} from Regular Queue for {self} is being served at {self.current_time}.") # for debugging
                    self.total_served += 1
                    print(f"{self} 's total serving so far: {self.total_served}")  # for debugging
        
        if not self.fast_pass_queue and not self.regular_queue:
            print(f"No one is waiting in the queues for {self} at time", self.current_time)


    def process_ride(self): 
        if self.current_time in self.timeslot: # time to start a ride
            print(f"{self} is happening at {self.current_time}, serving {len(self.riding_visitors)} visitors.")
            self.rides_served += 1
            self.process_queue()

        if self.current_time == self.timeslot[-1]: # ie last show liao
            print(f"{self} 's LAST TIME SLOT is happening at {self.current_time}, serving {len(self.riding_visitors)} visitors.")
        

    def release_visitor(self):
        if self.current_time in self.endings: # time to end the ride, release visitors
            if self.riding_visitors:    
                for visitor in self.riding_visitors:
                    visitor.status = "done" # finish the ride
                    print(f"{visitor} is prepared to leave {self} at time {self.current_time}.")
                    print(f"The stats for {visitor} is as listed: {visitor.current_location}, {visitor.next_location}, {visitor.itinerary}")
            self.riding_visitors = [] # reset

    
    def get_data(self): # Note: should only be called AFTER calling process_queue()
        if math.isnan(self.ride_duration):
            self.ride_duration = 0
        crowd_level = {'fast_pass_queue':len(self.fast_pass_queue), 
                       'regular_queue':len(self.regular_queue)}
        curr_wait_time = {'fast_pass_queue':math.floor(crowd_level['fast_pass_queue']/self.ride_capacity)*self.ride_duration, 
                          'regular_queue':math.floor(crowd_level['regular_queue']/self.ride_capacity)*self.ride_duration}
        return self.current_time, crowd_level, curr_wait_time

########################################################################################################################

class Utility: # Including toilets, dining outlets, souvenir shops
    def __init__(self, name: str, service_duration: int, util_capacity: int):
        """
        Initialize a utility in the part.
        :param name: Name of the utility.
        :param service_time: Duration spent inside the Utility
        :param util_capacity: Number of visitors that can be processed at once.
        """
        self.name = name
        self.service_duration = int(service_duration)
        self.util_capacity = util_capacity 
        self.queue = deque()
        self.serving_visitors = [] # To track visitors being served at the current moment
        self.current_time = None
        self.total_served = 0  # cumulative number of served visitors 
        self.most_current_visitor_start_time = None # The moment the most recent visitor start using the utility

    def __repr__(self): # representation method 
        return f"{self.name,self.util_capacity}"
    
    def add_visitor(self, visitor):
        if len(self.serving_visitors) < self.util_capacity: # have vacancy
            self.serving_visitors.append(visitor)
            visitor.service_count_down = self.service_duration
            self.total_served += 1
            visitor.status = "being served"
            print(f"{visitor} directly being served because there is vacancy at {self} at {self.current_time}.")
        else:
            self.queue.append(visitor)
            visitor.status = "queuing"
            print(f"There is no vacancy at {self} thus {visitor} joins the queue at {self.current_time}.")

    def process_queue(self):
        while len(self.serving_visitors) < self.util_capacity and self.queue: # have vacancy
            visitor = self.queue.popleft()
            self.serving_visitors.append(visitor)
            visitor.status = "being served"
            print(f"After queueing, {visitor} is served at {self} at {self.current_time}")

    def process(self):
        if self.serving_visitors:
            for visitor in self.serving_visitors[:]:
                visitor.service_count_down -= 1
                if visitor.service_count_down <= 0 :
                    self.serving_visitors.remove(visitor)
                    visitor.status = "done" # redundance?
                    print(f"{visitor} is prepared to leave {self} at time {visitor.current_time}.")
        self.process_queue()
      

    def advance_time(self):
        self.current_time += timedelta(minutes=1)
        print(f"The time now at {self.name} is {self.current_time}.")

    def get_data(self):
        if math.isnan(self.service_duration):
            self.service_duration = 0
        crowd_level = len(self.queue)
        print(f"At {self.current_time}, the crowd level in {self.name} is {crowd_level}")
        curr_wait_time = math.ceil(crowd_level / self.util_capacity) * self.service_duration
        print(f"At {self.current_time}, the estimated wait time in {self.name} is {curr_wait_time} minutes")
        return self.current_time, crowd_level, curr_wait_time
    
    def get_visitors_served(self):
        return self.total_served

########################################################################################################################

class ThemePark:
    def __init__(self, spawning_dict, all_possible_itineraries):
        """
        :param spawning_dict: a dictionary containing number of new visitors for each time slot
        """
        self.spawning_dict = spawning_dict
        self.all_possible_itineraries = all_possible_itineraries
        self.current_time = None  # Overall time counter for the simulation, take from the time_slot object
        self.entrance = []
        self.attractions = []
        self.utilities = []
        self.seasonals = []
        self.existing_visitors = []
        self.visitors_count = 0 # to track and create new unique visitors

    def add_entrance(self, entrance: Entrance):
        self.entrance.append(entrance)

    def add_attraction(self, attraction: Attraction):
        self.attractions.append(attraction)

    def add_utility(self, utility: Utility):
        self.utilities.append(utility)

    def add_seasonal(self, seasonal: Seasonal):
        self.seasonals.append(seasonal)

    def spawn_visitor(self, time, attraction_generator_df):
        visitors = [] # List to store Visitor objects
        
        for _ in range(int(self.spawning_dict[time.strftime('%H:%M')])):
            new_visitor = Visitor(id= self.visitors_count, 
                                    itinerary=random.choice(self.all_possible_itineraries), 
                                    fast_pass=random.choices([0,1], weights=[0.8, 0.2],k=1)[0], # 20% of purchasing express pass
                                    arrival_time=time,
                                    attraction_generator_df = attraction_generator_df)
            self.visitors_count += 1
            visitors.append(new_visitor)
        return visitors
    
    def open_park(self, time): 
        for item in self.attractions+self.utilities+self.seasonals:
            item.current_time = time
 
    def advance_time(self):
        self.current_time += timedelta(minutes=1)

########################################################################################################################
