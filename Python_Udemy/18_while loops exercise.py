#1st way to do this:

# for a in range(2): #number of times variable "a" repeats based on 2nd/3rd line
# 	for num in range(1,11): #prints *,**,***, etc. to 10
# 		print("*" * num) #prints num * 1-10

#2nd way to do this:

# "for a in range (2):" <--can add this as a string multiplier
times = 1
while times <11:
	print("*" * times)
	times +=1

#without string multiplication - ugly solution

# for num in range (1,11):
# 	count=1
# 	smileys =""
# 	while count < num:
# 		smileys +="*"
# 		count +=1
# 	print(smileys)

	