# SECTION 19 FROM COLT STEEL PYTHON 3 COURSE





#%% 176: STAR ARGS

def sum_all_nums(*args):
    total = 0
    for num in args:
        total += num
    return total

print(sum_all_nums(4,6,9,4,10)) # 33

#%%
def ensure_correct_info(*args):
	if "Colt" in args and "Steele" in args:
		return "Welcome back Colt!"
	return "Note sure who you are"

print(ensure_correct_info("hello", False, 78)) # Not sure who you are...
print(ensure_correct_info(1, True, "Steele", "Colt"))


# EXAMPLE:
def contains_purple(*args):
    if "purple" in args: return True
    return False

print(contains_purple(25, "purple")) # True









#%% 178: KWARGS

def fav_colors(**kwargs):
    for person, color in kwargs.items():
        print(f"{person}'s favorite color is {color}")

fav_colors(colt="purple", ruby="red", ethel="teal")

#%%
def special_greeting(**kwargs):
    if "David" in kwargs and kwargs["David"] == "special":
        return "You get a special greeting David!"
    elif "David" in kwargs:
        return f"{kwargs['David']} David!"
    return "Not sure who this is..."

print(special_greeting(David='Hello')) # Hello David!
print(special_greeting(Bob='hello')) # Not sure who this is...
print(special_greeting(David='special')) # You get a special greeting David!


# EXAMPLE
def combine_words(word,**kwargs):
    if 'prefix' in kwargs:
        return kwargs['prefix'] + word
    elif 'suffix' in kwargs:
        return word + kwargs['suffix']
    return word

combine_words('child') # 'child'
combine_words('child', prefix = 'man')
combine_words('child', suffix = 'ish')







#%% 180: Ordering Parameters
    # 1. parameters
    # 2. *args
    # 3. default parameters
    # 4. **kwargs

def display_info(a, b, *args, instructor="Colt", **kwargs):
  # return [a, b, args, instructor, kwargs]
  print(type(args))

print(display_info(1, 2, 3, last_name="Steele", job="Instructor"))

# a - 1
# b - 2
# args (3)
# instructor - "Colt"
# kwargs - {'last_name': "Steele", "job": "Instructor"}

[1, 2, (3,), 'Colt', {'last_name': 'Steele', 'job': 'Instructor'}]









#%% 181: Tuple Unpacking

def sum_all_values(*args):
    print(args)
    total = 0
    for num in args:
        total += num
    print(total)

sum_all_values(1,30,2,5,6) # (1, 30, 2, 5, 6) 44

# must pass * in with the argument to unpack the tuple
# tuple unpacking
nums = [1,2,3,4,5,6]
sum_all_values(*nums)










#%%  182: Dictionary Unpacking

def display_names(first, second):
    print(f"{first} says hello to {second}")

names = {"first": "Colt", "second": "Rusty"}

# display_names(names) # nope..
display_names(**names)  # yup!


def add_and_multiply_numbers(a,b,c,**kwargs):
    print(a + b * c)
    print("OTHER STUFF....")
    print(kwargs)

data = dict(a=1,b=2,c=3,d=55,name="Tony")

# can add more key value pairs to the dictionary:
add_and_multiply_numbers(**data, cat='blue') # 7
