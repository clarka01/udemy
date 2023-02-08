{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from random import randint  # use randint(a, b) to generate a random number between a and b\n",
    " \n",
    "number = 0 #store random number in here, each time through\n",
    "i = 0  # i should be incremented by one each iteration\n",
    " \n",
    "while number != 5: #keep looping while number is not 5\n",
    "    i += 1\n",
    "    number = randint(1, 10) #update number to be a new random int from 1-10"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
