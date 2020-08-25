# Import routines

import numpy as np
import math
import random

# Import Itertools
from itertools import permutations  #It gives various functions that work with iterators to produce complex iterators and help us to solve problems easily and efficiently in terms of time as well as memory.

# Defining hyperparameters
m = 5  # number of cities, ranges from 0 ..... m-1
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        self.action_space = [(0, 0)] + list(permutations([i for i in range(m)], 2))
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        state_encod = [0 for _ in range(m+t+d)]  ## initialize vector state
        state_encod[self.state_get_loc(state)] = 1  ## set the location value into vector - City to 1
        state_encod[m+self.state_get_time(state)] = 1  ## set the location value into vector - time to 1
        state_encod[m+t+self.state_get_day(state)] = 1  ## set the location value into vector - day to 1

        return state_encod

    # Use this function if you are using architecture-2

    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0 for _ in range(m+t+d+m+m)]    ## initialize vector state + action space
        state_encod[self.state_get_loc(state)] = 1     ## set the location value into vector
        state_encod[m+self.state_get_time(state)] = 1  ## set time value into vector
        state_encod[m+t+self.state_get_day(state)] = 1 ## set day value into vector
        if (action[0] != 0):
            state_encod[m+t+d+self.action_get_pickup(action)] = 1  ## Setting up pikup location
        if (action[1] != 0):
            state_encod[m+t+d+m+self.action_get_drop(action)] = 1  ## Setting up drop location

        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:                    ## At Location A, we get TWO (λ) Requests on an Average             
            requests = np.random.poisson(2)  ## sample the number of requests from a Poisson distribution using the mean λ
        if location == 1:                    ## At Location B, we get TWELVE (λ) Requests on an Average
            requests = np.random.poisson(12) ## Poisson Distribution
        if location == 2:                    ## At Location C, we get FOUR (λ) Requests on an Average
            requests = np.random.poisson(4)  ## Poisson Distribution
        if location == 3:                    ## At Location D, we get SEVEN (λ) Requests on an Average
            requests = np.random.poisson(7)  ## Poisson Distribution
        if location == 4:                    ## At Location E, we get EIGHT (λ) Requests on an Average
            requests = np.random.poisson(8)  ## Poisson Distribution

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0]  # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions

    def Get_DateTime_Updated(self, time, day, RideTime):
        RideTime = int(RideTime)
        if (time + RideTime) < 24: #Take Current Time
            time = time + RideTime
        else:
            time = (time + RideTime) % 24 
            NumberOfDays = (time + RideTime) // 24 # Getting number of days
            day = (day + NumberOfDays) % 7

        return time, day
    
    def reward_func(self, WaitTime, TransitTime, RideTime):
        """Takes in state, action and Time-matrix and returns the reward"""
        RevenueTime = RideTime
        IdleTime   = WaitTime + TransitTime        
        reward = (R * RevenueTime) - (C * (RevenueTime + IdleTime))

        return reward    
    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        
        ## Initialize Time - Rewards happens depends on time
        total_time   = 0
        TransitTime = 0    # Current Location to pickup location
        WaitTime    = 0    # Refuse all requests
        RideTime    = 0    # Pickup location to drop location
        
        # Derive the current location, time, day and request locations
        CurrentLocation = self.state_get_loc(state)
        PickupLocation = self.action_get_pickup(action)
        DropLocation = self.action_get_drop(action)
        CurrentTime = self.state_get_time(state)
        CurrentDay = self.state_get_day(state)
   
        ## Next State depends on the action taken by the driver
        if ((PickupLocation== 0) and (DropLocation == 0)):  ## Cab driver rejects  the ride - Action (0,0)
            # Refuse all requests, so wait time is 1 unit, next location is current location
            WaitTime = 1
            NextLocation = CurrentLocation
        elif (CurrentLocation == PickupLocation): ## Cab driver pick up ath the same location
            RideTime = Time_matrix[CurrentLocation][DropLocation][CurrentTime][CurrentDay]
            NextLocation = DropLocation  ## Drop location
        else:
            TransitTime      = Time_matrix[CurrentLocation][PickupLocation][CurrentTime][CurrentDay] # Time taken to reach pickup point
            new_time, new_day = self.Get_DateTime_Updated(CurrentTime, CurrentDay, TransitTime)
          
            RideTime = Time_matrix[PickupLocation][DropLocation][new_time][new_day] # Time taken to drop passengers
            NextLocation  = DropLocation

        # Calculate total time as sum of all durations
        total_time = (WaitTime + TransitTime + RideTime)
        next_time, next_day = self.Get_DateTime_Updated(CurrentTime, CurrentDay, total_time)
        
        # Construct next_state using the NextLocation and the new time states.
        next_state = [NextLocation, next_time, next_day]
        
        return next_state, WaitTime, TransitTime, RideTime

    def step(self, state, action, Time_matrix):
        # Get the next state and the various time durations
        next_state, WaitTime, TransitTime, RideTime = self.next_state_func(
            state, action, Time_matrix)
        # Calculating reward based on different time
        rewards = self.reward_func(WaitTime, TransitTime, RideTime)
        total_time = WaitTime + TransitTime + RideTime
        
        return rewards, next_state, total_time

    def state_get_loc(self, state):
        return state[0]

    def state_get_time(self, state):
        return state[1]

    def state_get_day(self, state):
        return state[2]

    def action_get_pickup(self, action):
        return action[0]

    def action_get_drop(self, action):
        return action[1]

    def state_set_loc(self, state, loc):
        state[0] = loc

    def state_set_time(self, state, time):
        state[1] = time

    def state_set_day(self, state, day):
        state[2] = day

    def action_set_pickup(self, action, pickup):
        action[0] = pickup

    def action_set_drop(self, action, drop):
        action[1] = drop
        
    def reset(self):
        return self.action_space, self.state_space, self.state_init