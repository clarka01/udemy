{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "times tell you? 4\n",
      "time 1: clean up your room\n",
      "time 2: clean up your room\n",
      "time 3: clean up your room\n",
      "time 4: clean up your room\n"
     ]
    }
   ],
   "source": [
    "### \n",
    "times = input(\"times tell you? \")\n",
    "times = int(times)\n",
    "\n",
    "for repeats in range(times):\n",
    "\tprint(f\"time {repeats+1}: clean up your room\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-8-648b2ca79b1c>, line 3)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-8-648b2ca79b1c>\"\u001b[1;36m, line \u001b[1;32m3\u001b[0m\n\u001b[1;33m    if n % 2 != 0\u001b[0m\n\u001b[1;37m                 ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "x=0\n",
    "for n in range(10, 21): #remember range is exclusive, so we have to go up to 21\n",
    "    if n % 2 != 0\n",
    "\n",
    "x += n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
