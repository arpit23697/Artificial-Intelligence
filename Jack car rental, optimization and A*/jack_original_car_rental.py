#import all the libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt

#initialising all the global varaibles

TOTAL_LOCATIONS = 3

MAX_CAR_LOC1 = 6
MAX_CAR_LOC2 = 6
MAX_CAR_LOC3 = 6

MAX_CAR = [6 , 6 , 6]

LAMBDA_REQ_LOC1 = 3
LAMBDA_REQ_LOC2 = 2
LAMBDA_REQ_LOC3 = 2

LAMBDA_RETURN_LOC1 = 3
LAMBDA_RETURN_LOC2 = 1
LAMBDA_RETURN_LOC3 = 1

MAX_CAR_SHIFT = 3

GAMMA = 0.9

CAR_REQUEST_REWARD = 10
CAR_MOVE_COST_LOC1_LOC2 = -2
CAR_MOVE_COST_LOC2_LOC3 = 0
CAR_MOVE_COST_LOC1_LOC3 = -2

THRESHOLD_CONVERGENCE = 0.1

P_BACKUP = {}

#this is for the state space
state_space = []

for x in range(MAX_CAR_LOC1 + 1):
    for y in range(MAX_CAR_LOC2 + 1):
        for z in range(MAX_CAR_LOC3 + 1):
            state_space.append((x , y , z))
            

#for the state values
state_values = np.zeros((MAX_CAR_LOC1+1 , MAX_CAR_LOC2+1 , MAX_CAR_LOC3+1))
policy = np.zeros((MAX_CAR_LOC1+1 , MAX_CAR_LOC2+1 , MAX_CAR_LOC3+1))
#action space

#action is also a four tuple:
    #first element : Gives the first location
    #second element : Gives the second location
    #third element : Number of cars to be moved from first location to third location
    #fourth element : Number of cars to be moved from second location to third location
    
action_space = []

for x in range(TOTAL_LOCATIONS):
    for y in range(x+1 , TOTAL_LOCATIONS):
        for a in range(MAX_CAR_SHIFT + 1):
            for b in range(5 - a + 1):
                action_space.append( (x , y , a , b) )
                action_space.append( (y , x , a , b) )

action_space = sorted(action_space)

#given the value of the x and the lambda it will give the probability of the distribution
def poisson (x , lam):
    global P_BACKUP
    key = (x , lam)
    if key not in P_BACKUP.keys():
        P_BACKUP[key] = np.exp(-lam) * pow(lam , x) / math.factorial(x)
    
    return P_BACKUP[key]

#given an action it will check if the given action is valid or not in that particular state


#action is also a four tuple:
    #first element : Gives the first location
    #second element : Gives the second location
    #third element : Number of cars to be moved from first location to third location
    #fourth element : Number of cars to be moved from second location to third location

def action_valid (action , state):
    
    #extract the values
    car1 , car2 , car3 = state
    loc1 , loc2 , move_13 , move_23 = action
    
    #the action will be invalid if the cars at particular location is less than it want to sent to 
    #other location
    
    if (state[loc1] < move_13):
        return False
    
    if (state[loc2] < move_23):
        return False
    
    return True
    
#Given the state and the action it will return the new state
#Note this function do not check if the action is valid or not in that particular state
#this will also return the reward

def other_location (loc1 , loc2):
    return (3 - (loc1 + loc2))
    



def next_state (action , state):
    loc1 , loc2 , move_13 , move_23 = action
    
    #this is the third location
    loc3 = other_location(loc1 , loc2)
    
    #now move the cars from loc1 to loc3 and loc2 to loc3
    new_state = [-1 , -1 , -1]
    
    new_state[loc1] = state[loc1] - move_13
    new_state[loc2] = state[loc2] - move_23
    
    new_state[loc3] = state[loc3] + move_13 + move_23
    new_state[loc3] = min(new_state[loc3] , MAX_CAR[loc3] )

    reward = 0
    if ( (loc1 == 0 and loc3 == 1) or (loc1 == 1 and loc3 == 0)):
        reward += move_13 * CAR_MOVE_COST_LOC1_LOC2
    elif ((loc1 == 0 and loc3 == 2) or (loc1 == 2 and loc3 == 0)):
        reward += move_13 * CAR_MOVE_COST_LOC1_LOC3
    elif((loc1 == 1 and loc3 == 2) or (loc1 == 2 and loc3 == 1)):
        reward += move_13 * CAR_MOVE_COST_LOC2_LOC3
        
    if ( (loc2 == 0 and loc3 == 1) or (loc2 == 1 and loc3 == 0)):
        reward += move_23 * CAR_MOVE_COST_LOC1_LOC2
    elif ((loc2 == 0 and loc3 == 2) or (loc2 == 2 and loc3 == 0)):
        reward += move_23 * CAR_MOVE_COST_LOC1_LOC3
    elif((loc2 == 1 and loc3 == 2) or (loc2 == 2 and loc3 == 1)):
        reward += move_23 * CAR_MOVE_COST_LOC2_LOC3
    
    
    
    return (new_state[0] , new_state[1] , new_state[2] , reward)

#given the state and the action this will give the new value of taking that action in that state
#note this function is not going to check the validity of the action
#it will just return the new value of the state given the state and the action


def new_state_value (action , state , state_values):
    
    ret = -1
    #after moving the cars what is the cost of moving and the next state n_state
    cars_loc1 , cars_loc2 , cars_loc3, ret = next_state(action , state)
    n_state = (cars_loc1 , cars_loc2, cars_loc3)
    
    #from n_state go to all the states with poisson distribution and accumulate the reward
    
    #rentals mean they will give this many car to the values
    #if request is greater than cars available then request changes to cars available
    for rentals_loc1 in range(0 , MAX_CAR_LOC1):
        for rentals_loc2 in range(0 , MAX_CAR_LOC2):
            for rentals_loc3 in range(0 , MAX_CAR_LOC3):
                rental_prob = poisson(rentals_loc1 , LAMBDA_REQ_LOC1) * poisson(rentals_loc2 , LAMBDA_REQ_LOC2) * poisson(rentals_loc3 , LAMBDA_REQ_LOC3) 
                
                total_rental_loc1 = min(cars_loc1 , rentals_loc1)
                total_rental_loc2 = min(cars_loc2 , rentals_loc2)
                total_rental_loc3 = min(cars_loc3 , rentals_loc3)
                
                rewards = (total_rental_loc1 + total_rental_loc2 + total_rental_loc3) * CAR_REQUEST_REWARD
                
                for returns_loc1 in range(0 , MAX_CAR_LOC1):
                    for returns_loc2 in range(0 , MAX_CAR_LOC2):
                        for returns_loc3 in range(0 , MAX_CAR_LOC3):
                            
                            return_prob = poisson(returns_loc1 , LAMBDA_RETURN_LOC1) * poisson(returns_loc2 , LAMBDA_RETURN_LOC2) * poisson(returns_loc3 , LAMBDA_RETURN_LOC3)
                            prob = return_prob * rental_prob
                            
                            next_state1 = min (MAX_CAR_LOC1 , cars_loc1 - total_rental_loc1 + returns_loc1)
                            next_state2 = min(MAX_CAR_LOC2 , cars_loc2 - total_rental_loc2 + returns_loc2)
                            next_state3 = min(MAX_CAR_LOC3 , cars_loc3 - total_rental_loc3 + returns_loc3)
                            
                            next_st = (next_state1 , next_state2 , next_state3)
                            ret += prob * (rewards + GAMMA * state_values[next_st])
                            
                            
    
    
    return ret

#this is synchronous update
#this will return the optimal value function

#given the old state values it will give the new state values after one complete iteration

def value_iteration(state_values , policy):
    
    #for all the states take all the actions
    threshold = -math.inf
    
    for s , state in zip(range(len(state_space)) , state_space): 
        new_value = []
        if (s % 20 == 0):
            print(s)
        for action in action_space:
            
            temp = -1
            if(action_valid(action , state)):
                temp = new_state_value(action , state , state_values)
                new_value.append(temp)
                
            else:
                new_value.append(-math.inf)
                
        optimal_value = max(new_value)
        optimal_action = np.argmax(new_value)
        
        threshold = max(threshold , abs(state_values[state] - optimal_value) )
        
        #this is updating as soon as you get answer
        state_values[state] = optimal_value
        policy[state] = optimal_action
        
    return (threshold , state_values , policy)
            
threshold = math.inf

num_of_iterations = 0
while (threshold > THRESHOLD_CONVERGENCE):
    num_of_iterations += 1
    print("Number of iterations : " , num_of_iterations , threshold)
    threshold , state_values , policy = value_iteration(state_values , policy)
    print("Number of iterations : " , num_of_iterations , threshold)

print(state_values[: , : , 0])