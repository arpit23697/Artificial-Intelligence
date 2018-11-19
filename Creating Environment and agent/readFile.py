#!/usr/bin/python3
file = open("in.txt" , "r")
char = ""
ans = 0
lastSpace = True
while(1):
	char = file.read(1)	
	if not char:
		break	
	if(ord(char) == 10 or ord(char) == 32):
		if (lastSpace):
			lastSpace = True
		else:
			ans+=1
			lastSpace = True
	else:
		lastSpace = False
print("number of words :  ",ans)

