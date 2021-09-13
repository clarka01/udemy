
#Basic Return
def speak_pig():
    return 'oink'


#Generate Even #s between 1-49
    def generate_evens():
    return [x for x in range(1,50) if x%2 == 0]


#using a loop

def generate_evens():
    result = []
    for x in range(1,50):
        if x % 2 == 0:
            result.append(x)
    return result


#Parameter (num) and Argument (4,8)

def square(num):
	return num * num

print(square(4))
print(square(8))

#concatenation method

def yell(word):
    return word.upper() + "!"

#string format()) method:
def yell(word):
    return "{}!".format(word.upper())
#f-string

def yell(word):
    return f"{word.upper()}!"


#SUM OF ODD NUMBERS: 

    #OLD-VERSION----OLD-VERSION----OLD-VERSION-----
# def sum_odd_numbers(numbers): 
#     total = 0
#     for num in numbers:
#         if num % 2 != 0:
#             total += num
#         return total #Returning too early :(
#OLD-VERSION----OLD-VERSION----OLD-VERSION-----


# NEW AND IMPROVED VERSION :)
def sum_odd_numbers(numbers):
    total = 0
    for num in numbers:
        if num % 2 != 0:
            total += num
    return total

print(sum_odd_numbers([1,2,3,4,5,6,7]))

#Default Functions:

def add(a,b):
    return a+b

def math(a,b,fn=add): #fn=add can be fn=subtract, or any other function
    return fn(a,b)

def subtract(a,b):
    return a-b

#parameters are listed in order
print(math(4,5)) #results in add(4,5) which is 9

print(math(4,5,subtract)) #results in subtract(4,5) which is -1

#print animal noises exercise:

def speak(animal="dog"):
    if animal == "pig":
        return "oink"
    elif animal == "duck":
         return "quack"
    elif animal == "cat":
        return "meow"
    elif animal == "dog": #default value
        return "woof"
    else:
        return "?"

# more compact version

noises = { #dictionary version
    "dog": "woof", 
    "pig": "oink", 
    "duck": "quack", 
    "cat": "meow"
}

def speak(animal="dog"):
    noises = {"dog": "woof", "pig": "oink", "duck": "quack", "cat": "meow"} #
    noise = noises.get(animal)
    if noise:
        return noise
    return "?"

#even more compact version

def speak(animal='dog'):
    noises = {'pig':'oink', 'duck':'quack', 'cat':'meow', 'dog':'woof'}
    return noises.get(animal, '?')
'''
number_compare(1,1) # "Numbers are equal"
number_compare(1,0) # "First is greater"
number_compare(2,4) # "Second is greater"
'''

def number_compare(a,b):
    if a > b:
            return "First is greater"
    elif a < b:
            return "Second is greater"
    return "Numbers are equal"

def speak(animal='dog'):
    noises = {'pig':'oink', 'duck':'quack', 'cat':'meow', 'dog':'woof'}
    return noises.get(animal, '?')

 #days of the week exercise:

 def return_day(num):
    days = ["Sunday","Monday", "Tuesday","Wednesday","Thursday","Friday","Saturday"]
    # Check to see if num valid
    if num > 0 and num <= len(days):
        # use num - 1 because lists start at 0 
        return days[num-1]
    return None

    '''
number_compare(1,1) # "Numbers are equal"
number_compare(1,0) # "First is greater"
number_compare(2,4) # "Second is greater"
'''

def number_compare(a,b):
    if a > b:
            return "First is greater"
    elif a < b:
            return "Second is greater"
    return "Numbers are equal"

print(number_compare(1,2))

"""In my solution, I use the built-in count()  
to count the number of times letter  appears in string .  
I also downcase both string  and letter  to make it case-insensitive 
(you could also upcase both instead)

single_letter_count("Hello World", "h") # 1
single_letter_count("Hello World", "z") # 0
single_letter_count("HelLo World", "l") # 3
"""

def single_letter_count(string,letter):
   	return string.lower().count(letter.lower()) #by making return string.lower, ensure's count can count all letters regardless of case

print(single_letter_count("booooooooo", "o"))


'''
Exercise 167: 
list_manipulation([1,2,3], "remove", "end") # 3
list_manipulation([1,2,3], "remove", "beginning") #  1
list_manipulation([1,2,3], "add", "beginning", 20) #  [20,1,2,3]
list_manipulation([1,2,3], "add", "end", 30) #  [1,2,3,30]
'''

def list_manipulation(collection, command, location, value=None):
    if(command == "remove" and location == "end"):
        return collection.pop()
    elif(command == "remove" and location == "beginning"):
        return collection.pop(0)
    elif(command == "add" and location == "beginning"):
        collection.insert(0,value)
        return collection
    elif(command == "add" and location == "end"):
        collection.append(value)
        return collection
    
list_manipulation([1,2,3], "add", "beginning", 20)

'''
is_palindrome('testing') # False
is_palindrome('tacocat') # True
is_palindrome('hannah') # True
is_palindrome('robert') # False
is_palindrome('amanaplanacanalpanama') # True
'''

def is_palindrome(string):
    stripped = string.replace(" ", "")
    return stripped == stripped[::-1]

is_palindrome("123 456 789 8 765 432 1")

'''
exercise 170
frequency([1,2,3,4,4,4], 4) # 3
frequency([True, False, True, True], False) # 1
'''

def frequency(list1, search_term):
    return list1.count(search_term)

 '''
 multiply only even numbers
 exercise 171
 '''

def multiply_even_numbers(lst):
    total = 1
    for val in lst:
        if val % 2 == 0:
            total = total * val
    return total

'''
capitalize("tim") # "Tim"
capitalize("matt") # "Matt"
'''

def capitalize(word):
    return word[0:1:].upper() + word[1::]