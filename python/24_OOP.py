#%%

# CLASSES & OBJECT ORIENTED PROGRAMMING

'''   Define OOP
    # understand encapsulation and abstraction
    # create classes and instances and attach methods and properties to each
    # Create classes and instances and attach
        # methods and properties to each

    # Class: blueprint for creating new objects
    # Object: instance of a class

    # encapsulation: bundling of data and methods that act on that data
    # abstraction: hiding of information and giving access to only what is necessary
'''



class User:
    def __init__(self, first, last, age):
        self.first = first
        self.last = last
        self.age = age

user1 = User('Joe', 'Smith', 68)
user2 = User('Blanca', 'Lopez', 41)

print(user1.first, user1.last)


#%%

class Comment:
    def __init__(self, username, text, likes=0):
        self.username = username
        self.text = text
        self.likes = likes
        
c = Comment('davey123', "lol you're so silly", 3)

# instantiate instance of this class:
print(c.username)
print(c.text)
print(c.likes)

#%%

# 242. UNDERSCORE METHODS

# _name #secret variable
# __name # mangles name, puts class name first
# __name__ # used for python specific methods; shouldn't use this convention


'''
There are no secrets in python, but this is code convention.
This variable would be intended for internal use

'''

class Person:
    def __init__(self):
        self.name = "Tony"
        self._secret = "hi!"
        self.__msg = "I like turtles!"

p = Person()

print(p.name)
print(p._secret)
# print(dir(p))
print(p.__msg) # this does exist; python "mangles" the name


#%%

# 243. Adding Instance Methods

class User:
    def __init__(self, first, last, age):
        self.first = first
        self.last = last
        self.age = age
    
    def full_name(self):
        return f'{self.first} {self.last}'
    
    def initials(self):
        return f'{self.first[0]}.{self.last[0]}.'
    
    def likes(self, thing):
        return f'{self.first} likes {thing}'
    
    def is_senior(self):
        return self.age >= 65
    
    def birthday(self):
        self.age += 1
        return f"Happy {self.age}th, {self.first}"
    
    def say_hi(self):
        print('Hello!')


user1 = User('Joe', 'Smith', 68)
user2 = User('Blanca', 'Lopez', 41)

print(user2.full_name())
print(user2.initials())

print(user1.likes('Ice Cream'))
print(user2.likes('Chips'))

print(user2.is_senior()) #true false
print(user2.birthday()) #adds 1 year to age
print(user2.age) #prints age
print(user1.age)
print(user1.say_hi())


#%%

# exercise 

# Define Bank Account Below:
class BankAccount:
 
    def __init__(self, name):
        self.owner = name
        self.balance = 0.0
 
    def getBalance(self):
        return self.balance
 
    def withdraw(self, amount):
        self.balance -= amount
        return self.balance
 
    def deposit(self, amount):
        self.balance += amount
        return self.balance
    

acct = BankAccount('Darcy')
print(acct.getBalance())
print(acct.owner) # no parenthesis around attributes
print(acct.deposit(10))
print(acct.withdraw(3))
print(acct.getBalance())


#%%

# Class Attributes:

# adding instance attributes (some pets allowed)

class Pet:
    allowed = ['cat', 'dog', 'fish', 'rat']
    def __init__(self, name, species):
        if species not in Pet.allowed:
            raise ValueError(f"You can't have a {species} pet")
        self.name = name
        self.species = species
    def set_species(self, species):
        if species not in Pet.allowed:
            print(f"You can't have a {species} pet")        
        self.species = species

cat= Pet('Blue', 'cat')

dog = Pet('Sirius','dog')
# pig = Pet('oink', 'pig') #ValueError

#override
cat.species = 'pig'

Pet.allowed.append('tiger')
tiger = Pet('sheerkhan', 'tiger')

#%%

# adding attributes with instance methods

class Chicken:
    total_eggs = 0
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.eggs = 0
    
    def lay_egg(self, n):
        self.eggs += n
        Chicken.total_eggs += n
        return self.eggs


chicken1 = Chicken('sp1', 'chicky')
chicken1.lay_egg(1)

chicken2 = Chicken('sp2', 'chunky')
chicken1.lay_egg(400)



#%%

# Class Methods
