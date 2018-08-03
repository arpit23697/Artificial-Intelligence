#!/usr/bin/python3

class environment:
	location = -1
	def __init__(self , x):
		self.location = x
	def percept (self , x):
		if (x == self.location):
			return True
		else :
			return False

class agent:
	currentLocation = -1
	def __init__ (self, x):
		self.currentLocation = 0		
	def moveRight (self , x):
		self.currentLocation += x
	def moveLeft (self , x):
		self.currentLocation -= x

increment = 1
left = True
me = agent(0)
env = environment(10)
steps = 0
print("Location of the agent : " , me.currentLocation)
print("location of the land : " , env.location)
#print(env.percept(me.currentLocation))
while (not env.percept(me.currentLocation)):
	steps+=increment
	print("Location : " , me.currentLocation)	
	if (left):
		left = False
		me.moveLeft(increment)
	else:
		left = True
		me.moveRight(increment)
	increment+=1	

print("Number of steps it takes : " , steps) 
 
