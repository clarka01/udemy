# donations = dict(sam=25.0, lena=88.99, chuck=13.0, linus=99.5, stan=150.0, lisa=50.25, harrison=10.0)
 
# total_donations = 0
 
# for donations in donations.values():
#  total_donations += donations

# def number_compare(a,b):
#     if a > b:
#             return "First is greater"
#     elif a < b:
#             return "Second is greater"
#     return "Numbers are equal"

# print(number_compare(1,2))

# def single_letter_count(string,letter):
#    	return string.lower().count(letter)

# print(single_letter_count("bOO", "o"))

def combine_words(word,**kwargs):
    if 'prefix' in kwargs:
        return kwargs['prefix'] + word
    elif 'suffix' in kwargs:
        return word + kwargs['suffix']
    return word

combine_words("child", prefix="man")