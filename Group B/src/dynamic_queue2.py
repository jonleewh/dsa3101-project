import random
from collections import deque
import numpy as np

class Attraction:
    def __init__(self, name: str, avg_wait_time: int, capacity: int):
        """
        Initialize an attraction with normal and express queues.
        :param name: Name of the attraction.
        :param avg_wait_time: Average wait time for the attraction.
        :param capacity: Number of visitors that can be processed at once.
        """
        self.name = name
        self.avg_wait_time = avg_wait_time
        self.capacity = capacity
        self.normal_queue = deque()  # Queue for regular visitors
        self.express_queue = deque()  # Queue for express pass visitors
        self.in_progress = deque()  # List of visitors currently in the attraction (visitor ID, start time)
        self.time = 0  # Local time counter for the attraction
        self.crowd_levels = []  # List to store crowd levels over time

    def add_visitor_to_queue(self, visitor_id: int, express_pass: bool):
        """
        Add a visitor to the appropriate queue based on pass type.
        :param visitor_id: Unique ID for the visitor.
        :param express_pass: True if visitor has express pass, else False.
        """
        if express_pass:
            self.express_queue.append(visitor_id)
        else:
            self.normal_queue.append(visitor_id)

    def process_queue(self):
        """
        Process visitors from the express and normal queues based on priority and capacity.
        """
        # Prioritize express queue
        while len(self.in_progress) < self.capacity and self.express_queue:
            visitor_id = self.express_queue.popleft()
            self.in_progress.append((visitor_id, self.time))

        # Fill remaining capacity with normal queue visitors
        while len(self.in_progress) < self.capacity and self.normal_queue:
            visitor_id = self.normal_queue.popleft()
            self.in_progress.append((visitor_id, self.time))

    def update_attraction(self, visitor_wait_times: np.ndarray):
        """
        Remove visitors whose wait time has completed and accumulate their waiting time.
        :param visitor_wait_times: Array to track the total waiting time per visitor.
        """
        while self.in_progress and (self.time - self.in_progress[0][1]) >= self.avg_wait_time:
            visitor_id, start_time = self.in_progress.popleft()
            wait_time = self.time - start_time  # Time spent in queue
            visitor_wait_times[visitor_id] += wait_time  # Accumulate waiting time for this visitor

    def increment_time(self, visitor_wait_times: np.ndarray):
        """
        Advance time, update attraction status, process queues, and record crowd level.
        :param visitor_wait_times: Array to track waiting time per visitor.
        """
        self.update_attraction(visitor_wait_times)
        self.process_queue()
        # Record the current crowd level (total people in queues and in progress)
        current_crowd_level = {
            "normal_queue": len(self.normal_queue),
            "express_queue": len(self.express_queue),
            "in_progress": len(self.in_progress)
        }
        self.crowd_levels.append(current_crowd_level)
        self.time += 1  # Increment local time

    def get_crowd_level_summary(self):
        """
        Get average crowd levels over the simulation.
        :return: Dictionary of average crowd levels.
        """
        avg_normal_queue = np.mean([level["normal_queue"] for level in self.crowd_levels])
        avg_express_queue = np.mean([level["express_queue"] for level in self.crowd_levels])
        avg_in_progress = np.mean([level["in_progress"] for level in self.crowd_levels])
        return {
            "avg_normal_queue": avg_normal_queue,
            "avg_express_queue": avg_express_queue,
            "avg_in_progress": avg_in_progress
        }

    def __str__(self):
        return f"{self.name} - Avg Wait Time: {self.avg_wait_time} minutes"


class ThemePark:
    def __init__(self, num_visitors: int):
        """
        Initialize the theme park and visitor tracking.
        :param num_visitors: Total number of visitors in the simulation.
        """
        self.visitor_count = num_visitors
        self.global_time = 0  # Overall time counter for the simulation
        self.attractions = []
        self.visitor_wait_times = np.zeros(num_visitors)  # Store accumulated wait times for each visitor

    def add_attraction(self, attraction: Attraction):
        """
        Add an attraction to the park.
        :param attraction: Attraction object.
        """
        self.attractions.append(attraction)

    def assign_visitors(self, express_pass_ratio=0.2):
        """
        Randomly assign visitors to attractions, with a percentage getting express passes.
        :param express_pass_ratio: Fraction of visitors with express passes.
        """
        for visitor_id in range(self.visitor_count):
            attraction = random.choice(self.attractions)
            express_pass = random.random() < express_pass_ratio  # Assign express pass based on ratio
            attraction.add_visitor_to_queue(visitor_id, express_pass)

    def simulate(self, steps: int):
        """
        Run the simulation for a fixed number of time steps.
        :param steps: Number of simulation steps (time units).
        """
        for _ in range(steps):
            for attraction in self.attractions:
                attraction.increment_time(self.visitor_wait_times)
            self.global_time += 1

    def output_waiting_times(self):
        """
        Output the total accumulated waiting times for each visitor.
        """
        for visitor_id, wait_time in enumerate(self.visitor_wait_times):
            print(f"Visitor {visitor_id} - Total Wait Time: {wait_time} minutes")

    def get_crowd_levels(self):
        """
        Output the average crowd levels for each attraction.
        """
        for attraction in self.attractions:
            crowd_level_summary = attraction.get_crowd_level_summary()
            print(f"{attraction.name} Crowd Levels: {crowd_level_summary}")

    def get_waiting_times_summary(self):
        """
        Get the average waiting time and total time spent for analysis.
        :return: Dictionary summarizing average waiting time and total waiting time.
        """
        return {
            "Average Wait Time": np.mean(self.visitor_wait_times),
            "Total Wait Time": np.sum(self.visitor_wait_times)
        }


# Example Usage
park = ThemePark(num_visitors=50)

# Create attractions with average wait times and capacity
roller_coaster = Attraction("Roller Coaster", avg_wait_time=45, capacity=20)
ferris_wheel = Attraction("Ferris Wheel", avg_wait_time=30, capacity=15)
haunted_house = Attraction("Haunted House", avg_wait_time=25, capacity=10)

# Add attractions to the park
park.add_attraction(roller_coaster)
park.add_attraction(ferris_wheel)
park.add_attraction(haunted_house)

# Assign visitors to queues (20% with express passes)
park.assign_visitors(express_pass_ratio=0.2)

# Run the simulation for a specified number of time steps
park.simulate(steps=60)

# Output total waiting times for each visitor
park.output_waiting_times()

# Output average crowd levels for each attraction
park.get_crowd_levels()

# Summary of average waiting times for analysis
waiting_times_summary = park.get_waiting_times_summary()
print("\nWaiting Times Summary:", waiting_times_summary)