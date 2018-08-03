#!/usr/bin/env

class environment:
	dirt_x = -1
	dirt_y = -1
	def __init__ (self, x , y):
		self.dirt_x = x
		self.dirt_y = y
	
	def precept (self , x , y):
		if (self.dirt_x == x and self.dirt_y == y):
			return True
		else:
			return False

class agent:
	current_x = -1
	current_y = -1
	def __init__ (self , x , y):
		self.current_x = x
		self.current_y = y

	def moveLeft(self , x):
		self.current_x -= x

	def moveRight(self , x):
		self.current_x += x

	def moveUp (self , x):
		self.current_y += x

	def moveDown (self , x):
		self.current_y -= x

me = agent(10 , -5)
env = environment(5 , 6)
print("Location of dirt : " , env.dirt_x , env.dirt_y) 
print("Location of vaccum : " , me.current_x , me.current_y)

dir1 = 0
increment = 0
step = 0
while (not env.precept(me.current_x , me.current_y)):
	step += 1
		
	
	print("step : " , step)
	print("Location of vaccum after step : " , me.current_x , me.current_y)	
	if (dir1 == 0):
		increment+=1		
		dir1 = 1
		for _ in range(increment):
			me.moveRight(1)
			print("Location of vaccum : " , me.current_x , me.current_y)
			if (env.precept(me.current_x , me.current_y)):
				break
			
	elif (dir1== 1):
		dir1 = 2
		for _ in range(increment):
			me.moveUp(1)
			print("Location of vaccum : " , me.current_x , me.current_y)
			if (env.precept(me.current_x , me.current_y)):
				break

	elif (dir1 == 2):
		increment+=1
		dir1 = 3
		for _ in range(increment):
			me.moveLeft(1)
			print("Location of vaccum : " , me.current_x , me.current_y)
			if (env.precept(me.current_x , me.current_y)):
				break
	elif (dir1 == 3):
		dir1 = 0
		temp = increment
		for _ in range(increment):
			me.moveDown(1)
			print("Location of vaccum : " , me.current_x , me.current_y)			
			if (env.precept(me.current_x , me.current_y)):
				break
			
