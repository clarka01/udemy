# Udmey Colt Steele OOP Pt 2
'''
- Inheritance
- Multiple interhetance, 
- Method resolution order
- Polymorphism
- Add Special methods to classes

'''

#%% 244 INHERITANCE: 
#               allows new objects to take on the properties of existing objects
#              (a "base" or "parent" class)

class Animal:
    cool = True

    def make_sound(self, sound):
        print(f"This animal says {sound}")


class Cat(Animal):
    pass

blue = Cat()
blue.make_sound("Meow")
print(blue.cool)
print(Cat.cool)
print(Animal.cool)

print(isinstance(blue, Cat))
print(isinstance(blue, Animal))
print(isinstance(blue, object))


#%% 245 INHERITANCE:
    # assign age (problematic)
    # can assign property
class Human:
    def __init__(self, first, last, age):
        self.first = first
        self.last = last
        if age >= 0:
            self._age = age
        else:
            self._age = 0

    # def get_age(self):
    #    return self._age   
    # def set_age(self, new_age):
    #    if new_age >= 0:
           
    @property
    def age(self):
        return self._age # see self."_age" above
    
    @age.setter # accesses _age
    def age(self, value):
        if value >= 0:
            self._age = value
        else:
            raise ValueError("age can't be negative")
        
    @property # property interacting with this, not an assignment
    def full_name(self):
        return f"{self.first} {self.last}"
    
    # @full_name.setter
    # def full_name(self, name):
    #     self.first, self.last = name.split(' ')


jane = Human('Jane', 'Goodall', -10)
# print(jane.age)
# jane.set_age(45)
# print(jane.age)
print(jane.age)
print(jane.full_name)
print(jane.__dict__)


#%% 257 -- super()

class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species     
    
    def __repr__(self):
        return f"{self.name} is a {self.species}"
    
    def make_sound(self, sound):
        print(f"This animal says {sound}")

class Cat(Animal):
    def __init__(self, name,species, breed, toy):
        super().__init__(name, species = 'Cat') # super points to the parent class
        # Animal.__init__(self, name, species) # this method is the long way
        self.breed = breed
        self.toy = toy

    def play(self):
        print(f"{self.name} plays with {self.toy}")

blue = Cat("Blue", "Cat","Scottish Fold", "String")
print(blue)
print(blue.species)
print(blue.breed)
print(blue.toy)
#%%




class User:
    active_users = 0
    @classmethod
    def display_active_users(cls):
        return f"There are currently {cls.active_users} active users"
    
    @classmethod # helps data that comes in as comma separated string:
    def from_string(cls, data_str):
        first,last,age = data_str.split(',')
        return cls(first, last, int(age))
    
    def __init__(self, first, last, age):
        self.first = first #this is the above parameter (could both be name, but first works better)
        self.last = last
        self.age = age
        User.active_users += 1
    
    def __repr__(self):
        return f"{self.first} is {self.age}, and initials are {self.initials()}"

    def logout(self):
        User.active_users -= 1
        return f"{self.first} has logged out"
    
    # methods below...
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
    

class Moderator(User):
    total_mods = 0 # keeps count of moderators
    def __init__(self, first, last, age, community):
        super().__init__(first, last, age)
        self.community = community
        Moderator.total_mods += 1

    @classmethod
    def display_active_mods(cls):
        return f"There are currently {cls.total_mods} active mods"
    
    def remove_post(self):
        return f"{self.full_name()} removed a post from the {self.community} community"
    
print(User.display_active_users())
jasmine = Moderator('Jasmine','O\'Conner', 61, 'Piano')
jasmine2 = Moderator('Jasmine','O\'Conner', 61, 'Piano')

print(jasmine.full_name())
print(jasmine.community)

print(User.display_active_users())
u1 = User('Tom','Garcia', 35)
u2 = User('Dave','Garcia', 33)
u2 = User('Felipe','Garcia', 43)

print(User.display_active_users())
print(Moderator.display_active_mods())


#%% EXAMPLE: 

class Character():
    def __init__(self, name, hp, level):
        self.name = name
        self.hp = hp
        self.level = level

class NPC(Character):
    def __init__(self, name, hp, level):
        super().__init__(name, hp, level)
 
    def speak(self):
        return f"{self.name} says: 'I heard monsters running around last night!'"        
    
villager = NPC("Bob", 100, 12)
print(villager.name)
print(villager.hp)  
print(villager.level)
print(villager.speak())


#%% 258 -- multiple inheritance

class aquatic:
    def __init__(self, name):
        print("AQUATIC INIT!")
        self.name = name
    
    def swim(self):
        return f"{self.name} is swimming"
    
    def greet(self):
        return f"I am {self.name} of the sea!"
    
class ambulatory:
    def __init__(self, name):
        print("AMBULATORY INIT!")
        self.name = name
    
    def walk(self):
        return f"{self.name} is walking"
    
    def greet(self):
        return f"I am {self.name} of the land!"
    

class Penguin(ambulatory, aquatic): # order matters, only calling 2.
    def __init__(self, name):
        print("PENGUIN INIT!")
        super().__init__(name = name)
        # ambulatory.__init__(self, name = name)
        # aquatic.__init__(self, name = name)

# jaws = aquatic("Jaws")
# lassie = ambulatory("Lassie")
captain_cook = Penguin("Captain Cook")

# print(captain_cook.swim())
# print(captain_cook.walk())
# print(captain_cook.greet())
# print(captain_cook.name)

# print(f'Is Captain Cook an instance of Penguin? {isinstance(captain_cook, Penguin)}')
# print(f'Is Captain Cook an instance of aquatic? {isinstance(captain_cook, aquatic)}')
# print(f'Is Captain Cook an instance of ambulatory? {isinstance(captain_cook, ambulatory)}')

# method resolution order
print(Penguin.__mro__) 
Penguin.mro()
print(help(Penguin)) 

#%%

# LOGICAL EXAMPLE

class A:
    def do_something(self):
        print("Method Defined In: A")

class B(A):
    def do_something(self):
        print("Method Defined In: B")
        super().do_something() # calls A

class C(A):
    def do_something(self):
        print("Method Defined In: C")

class D(B,C):
    def do_something(self):
        print("Method Defined In: D")
        super().do_something() # calls B

thing = D()
thing.do_something()
# print(D.mro())
# print(help(D))
 
    #    A
    #   / \
    #  B   C
    #   \ /
    #    D
# D.mro() == [D, B, C, A, object]


#EXAMPLE 2:

class Mother:
    def __init__(self):
        self.eye_color = "brown"
        self.hair_color = "dark brown"
        self.hair_type = "curly"
 
 
class Father:
    def __init__(self):
        self.eye_color = "blue"
        self.hair_color = "blond"
        self.hair_type = "straight"
 
 
class Child(Mother, Father):
    pass





#%% 259 -- Polymorphism

# A key principle in OOP is the idea of polymorphism 
#   - an object can take on many (poly) forms (morph).
# Polymorphism means that the same class method works 
#   in a similar way for different classes.
# We can have multiple classes inherit from the same superclass, 
#   and they can have their own
#   implementations of the same method (i.e. method with the same name). 
#   When invoked, each subclass's
    # method will be called instead of the superclass's method.

# aninal example (not impleneted error)
class Animal:
    def speak(self):
        raise NotImplementedError("Subclass needs to implement this method")
    
class Dog(Animal):
    def speak(self):
        return "woof"
    
class Cat(Animal):
    def speak(self):
        return "meow"
    
class Fish(Animal):
    pass

fido = Dog()
print(fido.speak())
blue = Cat()
print(blue.speak())
f = Fish()
f.speak()

#%% 260 -- Special Methods

# Special methods allow us to use Python specific functions 
#   on objects created through our class.
# These methods are always surrounded by double underscores 
#   (also called "dunder" for double underscore)
# The __init__ method is one of these special methods.
# We can also define our own special methods.
# For example, if we want to be able to compare two instances of a class, we can define a __gt__ method
#   (greater than) or a __lt__ method (less than).

class Human:
    def __init__(self, height):
        self.height = height

    def __len__(self):
        return self.height
    
Colt = Human(60)
len(Colt)

#%%

## add method:
from copy import copy
from random import choice

class Human:
    def __init__(self, first, last, age, gender):
        self.first = first #this is the above parameter (could both be name, but first works better)
        self.last = last
        self.age = age
        self.gender = gender


    def __repr__(self):
        return f'Human named {self.first} {self.last}, a {self.gender}, and is {self.age} years old'
    
    def __len__(self):
        return self.age
    
    def __add__(self, other):
        if isinstance(other, Human): # check if other is in fact  Human instance
            return Human(first = 'Newborn', 
                         last = self.last,
                         age = 0,
                         gender = choice(['male', 'female'])
                ) # passes 
        return "You can't add that!"
    
    #takes instance of Human and a number (other)
    def __mul__(self, other):
        if isinstance(other, int):
            return [copy(self) for i in range(other)] # other is number
        return "You can't multiply by that value"

m = Human('Jenny', 'Larsen', 47, 'male')
f = Human('Dave', 'Davidson', 49, 'female')

print(m)
print(len(f))

k = (m + f) # takes father's last name becuase father is listen last
k*3


# print(j * 'face') # doesn't work
# triplets = j * 3
# triplets[1].first = 'Jessica' # all are the same, so this changes all
# print(triplets)

#kevin & jessica have triplets
# triplets = (k+j) * 3
# print(triplets)
# [print(t) for t in triplets]

#%%

    # def get_age(self):
    #    return self._age   
    # def set_age(self, new_age):
    #    if new_age >= 0:
           
    @property
    def age(self):
        return self._age # see self."_age" above
    
    @age.setter # accesses _age
    def age(self, value):
        if value >= 0:
            self._age = value
        else:
            raise ValueError("age can't be negative")
        
    @property # property interacting with this, not an assignment
    def full_name(self):
        return f"{self.first} {self.last}"
    
    # @full_name.setter
    # def full_name(self, name):
    #     self.first, self.last = name.split(' ')










