#FOR LOOPS ----------------

colors =["purple", "teal", "magenta", True, 8.9]

for color in colors:
	print(color) #prints all values in list "colors"

#------------------------------------

numbers = [4,6,2,9,7,10]
	
for num in numbers
	print(num * num) #squares list

# WHILE LOOPS----------------------------

colors =["purple", "teal", "magenta", "red", "green"]

index = 0
while index < len(colors): #len of colors counts to the end of the list (5 items in this case)
	print(colors[index]) #square brackets references starting point defined in index =0
	index +=1 #adds each time for the len(colors) or lenght of list 

#---------------------------------------------

colors =["purple", "teal", "magenta", "red", "green"]

index = 0
while index < len(colors):
	print(f"{index} {colors[index]}")
	index +=1 

#PRACTICE TEST--------------------------------------------------

sounds = ["super", "cali", "fragil", "istic", "expi", "ali", "docious"]
# Define your code below:
result = ''
for s in sounds:
    result += s.upper()

#can also do upper seperately result = result.upper() # See leson 103 for other solution
