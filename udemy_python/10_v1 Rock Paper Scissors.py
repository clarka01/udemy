print("rock...")
print("paper...")
print("scissors...")

player1=input ('p1 make move')
player2=input ('p2 make move')

if player1== "rock" and player2=="scissors":
	print('p1 wins')
elif player1== "rock" and player2=="paper":
	print('p2 wins')
elif player1=="paper" and player2=="rock":
	print('p1 wins')
elif player1=="paper" and player2=="scissors":
	print('p2 wins')
elif player1=="scissors" and player2=="rock":
	print('p2 wins')
elif player1=="scissors" and player2=="paper":
	print('p1 wins')	
elif player1==player2:
	print('tie')
else:
	print("something's wrong")
