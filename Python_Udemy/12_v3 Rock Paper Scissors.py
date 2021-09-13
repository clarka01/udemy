#-------------------------------------------must use this prior to using random ints
from random import randint
#opening text for game
print("rock...")
print("paper...")
print("scissors...")
print("man vs machine")

#-----------------------------------------player input command
player=input ('p1 make move').lower()

#----------------------------------------computer input/Random number generation:
rand_num = randint(0,2)
if rand_num == 0:
	computer = "rock"
elif rand_num == 1:
	computer = "paper"
else:
	computer="scissors"
#----------------------------------------function of computer choice (above) based on randint:
print(f"computer plays {computer}" )

#------------------------------------conditional variables to determine winner:
if player==computer:
	print('tie')
elif player== "rock":
	if computer=="scissors":
		print('man wins')
	else:
		print('comp wins')
elif player== "paper":
	if computer=="rock":
		print('man wins')
	else:
		print('comp wins')
elif player=="scissors":
	if computer=="paper":
		print('man wins')
	else:
		print('comp wins')
else:
	print("something's wrong")