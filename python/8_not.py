age=21
# 2-8 $2 ticket
#<65 $5 ticket
#$10 for everyone else
if not((age>=2 and age <=8)) or age >=65:
	print("you pay $10")
else:
	print("you are a child or senior")