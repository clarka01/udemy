import random

random_number = random.randint(1,10)  # numbers 1 - 10
# can remove this because of true statement 'guess = None' but would add because code expects to return number (see line )

while True: #this will loop forever as long as player wants to play again
	guess = input("Pick a number from 1 to 10: ")
	guess = int(guess)
	if guess < random_number:
		print("TOO LOW!")
	elif guess > random_number:
		print("TOO HIGH!")
	else:
		print("YOU WON!!!!")
		play_again = input("Do you want to play again? (y/n) ")
		if play_again == "y":
			random_number = random.randint(1,10)  # numbers 1 - 10
			guess = None
		else:
			print("Thank you for playing!")
			break # this version loops forever

