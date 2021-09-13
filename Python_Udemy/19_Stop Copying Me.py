# msg = input("say something: ")

while msg != "stop copying me":
	print(msg)
	msg = input()
print("ok fine")

msg = input("say something: ")

while msg != "stop copying me":
	msg = input(f"{msg}\n")
print("ok fine")