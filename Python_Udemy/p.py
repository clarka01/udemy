k = 0

def encrypt_message(in_message):

    # initialize the output (encrypted) message
    out_message = ''

    for in_char in in_message:
        
        if in_char.isalpha():
            
            # if letter, encrypt it
            out_message += chr(ord(in_char) + k)
        
        else:
            
            # otherwise, keep it as is
            out_message += in_char

    return out_message

print(encrypt_message(f'Semester two is going fast. My k value is {k}.'))