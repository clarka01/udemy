from random import randint
player_wins = 0
computer_wins = 0
winning_score = 3 # just have to change this to determine best of winner

while player_wins < winning_score and computer_wins < winning_score: # this will add up the scores and compare to winning_score
	print(f'Player Score: {player_wins} Computer Score: {computer_wins}')

	#-------------------------------------------must use this prior to using random ints
	#opening text for game
	print("rock...")
	print("paper...")
	print("scissors...")
	print("man vs machine")

	#-----------------------------------------player input command
	player=input ('p1 make move').lower()
		if player =="quit" or player =="q":
			break	
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
			player_wins +=1
		else:
			print('comp wins')
			computer_wins +=1
	elif player== "paper":
		if computer=="rock":
			print('man wins')
			player_wins +=1
		else:
			print('comp wins')
			computer_wins +=1
	elif player=="scissors":
		if computer=="paper":
			print('man wins')
			player_wins +=1
		else:
			print('comp wins')
			computer_wins +=1
	else:
		print("something's wrong")

if player_wins > computer_wins: 
	print("congrats!")		
elif player_wins == computer_wins
	print("its a tie")
else:
	print("oh no")
print(f'FINAL SCORES: Player Score: {player_wins} Computer Score: {computer_wins}')
