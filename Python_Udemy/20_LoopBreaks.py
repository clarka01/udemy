# while True:
#     command = input("Type 'exit' to exit: ")
#     if (command == "exit"):
#         break

# for x in range(1, 101):
#     print(x)
#     if x == 3:
#         break

times = int(input("How many times do I have to tell you? "))#integer added to input verbiage
for time in range(times): # of times defines the range (2 = 3, 9 = 10, etc. becuase it starts at 0, not 1)
	print("CLEAN UP YOUR ROOM!") #prints the amount of times in the range until loops amount of times as number below
	if time >= 2: #really >=3 times
		print("do you even listen anymore?") #if 3 or more times, prints do you even listen anymore
		break

# from random import randint
# number = 0
# i=0
# while number != 5:
# 	i += 1
# 	number =randint(1,10)