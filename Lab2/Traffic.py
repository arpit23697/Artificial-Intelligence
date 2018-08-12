#importing all the libraries
import numpy as np
import pickle
import math
import pandas as pd

#this is to read the file using np.load and then converting it into numpy array
road_network = np.array(np.load("roads/road"))

#reading the path of each vehicle
vehicle = np.array(np.load("roads/vehicle"))

#this is to read the time at which they are leaving
time = np.array(np.load("roads/time", encoding='bytes'))

#creating the environment 
#road contains all the network
#status of road stores the numbers of vehciles on the road currently
class environment :
    def __init__ (self , road_network):
        self.road = road_network
        self.status_of_road = np.zeros(road_network.shape)
        
    def determine_speed(self , x):
        return math.exp(0.5 * x) / (1 + math.exp(0.5*x)) + 15 / (1 + math.exp(0.5 * x))
    
#detailed status stores when they cross each junction
#path stores the path each vehcile is going to follow
#time status stores the time by which each vehcile had travelled
#done stores how much each vehicle has travelled
#which vehicle returns the vehicle to be moved
class agents :
    
    #these are the variables
    detailed_status = np.array( (0 , 0) )         #this is to store the timing of each of the vehicle when it crossed the given node
    path = np.array((0 , 0))
    time_status = np.array((0 , 0))
    done = np.array((0 , 0))
    
    def __init__ (self, time , vehicle):
        self.path = vehicle                        #this is to store the path that the vehicles will follow
        self.time_status = time                    #this is to keep track of which vehicle have completed how much time
        
        #we divide by 60 so as to keep the time in hours
        self.detailed_status = np.hstack ([time/60 , math.inf * np.ones((vehicle.shape[0] , vehicle.shape[1] - 1)) ]) #
        
        #To keep track of which point the vehicle is in the path
        self.done =  np.hstack ([np.ones(time.shape , dtype=  bool) , np.zeros((vehicle.shape[0] , vehicle.shape[1] - 1), dtype= bool) ])
        
    def which_vehicle(self):
        return self.time_status.argmin()                 #this returns the index of the minimum element
        
#initialising the agent and the environment
agent = agents(time , vehicle)
env = environment (road_network)

#solving the problem
while(not np.all(agent.done)):
    
    current = agent.which_vehicle()
    index = -1
    #print("Current vehicle : " , current )
    
    for temp in range(agent.path.shape[1]):
        if (agent.done[current, temp] == False):
            index= temp;
            break

    if (index == -1):
        env.status_of_road[agent.path[current ,3] , agent.path[current , 4]] -=1
        agent.time_status[current , 0] = math.inf 
        continue
    
    agent.done[current, index] = True
    #print(agent.done)
   
    x = agent.path[current , index-1]
    y = agent.path[current , index]
    
    number_of_vehicle = env.status_of_road[x , y]
    env.status_of_road[x , y] += 1
    #print(x , y ,index , number_of_vehicle)
    #this means that the road is free
    if index-2 >= 0:
        env.status_of_road[agent.path[current , index-2] ,x] -=1
    #print(number_of_vehicle)
    speed = env.determine_speed(number_of_vehicle)
    time_required = env.road[x , y] / speed
    time_required = time_required * 60
    
    #print(agent.time_status[current , 0] , time_required)
    agent.time_status[current , 0] += time_required
    
    #this is for the status
    agent.detailed_status[current,  index] = agent.time_status[current , 0]/60
    
#this is to print the csv file using the pandas
df = pd.DataFrame (agent.detailed_status)
df.columns = ['site1' , 'site2' , 'site3' , 'site4' , 'site5']
df.to_csv('ans.csv')