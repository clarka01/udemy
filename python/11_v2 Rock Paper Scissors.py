import random
print("rock...")
print("paper...")
print("scissors...")


player1=input ('p1 make move')
print('***NO CHEATING!\n\n' * 20)
player2=input ('p2 make move')

#Random numbers:
rand_num = random.randint(0,2)
if rand_num == 0:
	computer = "rock"
elif rand_num == 1:
	computer = "paper"
else:
	computer="scissors"
print(computer)

if player1==player2:
	print('tie')

elif player1== "rock":
	if player2=="scissors":
		print('p1 wins')
	elif player2=="paper":
		print('p2 wins')

elif player1== "paper":
	if player2=="rock":
		print('p1 wins')
	if player2=="scissors":
		print('p2 wins')

elif player1=="scissors":
	if player2=="paper":
		print('p1 wins')
	if player2=="rock":
		print('p2 wins')

else:
	print("something's wrong")