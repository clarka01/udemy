age=input("how old?")

if age:
	age=int(age)
	if age >=18 and age <21:
		print ("need wristband")
	elif age >=21:
		print("in and drink")
	elif age <18:
		print("no entry")
	else: 
		print("enter age")
#no entry under 18, wristband 18-20, 21 entry and drink