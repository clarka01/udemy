#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/1-intro-to-linear-algebra.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Intro to Linear Algebra

# This topic, *Intro to Linear Algebra*, is the first in the *Machine Learning Foundations* series. 
# 
# It is essential because linear algebra lies at the heart of most machine learning approaches and is especially predominant in deep learning, the branch of ML at the forefront of today’s artificial intelligence advances. Through the measured exposition of theory paired with interactive examples, you’ll develop an understanding of how linear algebra is used to solve for unknown values in high-dimensional spaces, thereby enabling machines to recognize patterns and make predictions. 
# 
# The content covered in *Intro to Linear Algebra* is itself foundational for all the other topics in the Machine Learning Foundations series and it is especially relevant to *Linear Algebra II*.

# Over the course of studying this topic, you'll: 
# 
# * Understand the fundamentals of linear algebra, a ubiquitous approach for solving for unknowns within high-dimensional spaces. 
# 
# * Develop a geometric intuition of what’s going on beneath the hood of machine learning algorithms, including those used for deep learning. 
# * Be able to more intimately grasp the details of machine learning papers as well as all of the other subjects that underlie ML, including calculus, statistics, and optimization algorithms. 

# **Note that this Jupyter notebook is not intended to stand alone. It is the companion code to a lecture or to videos from Jon Krohn's [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations) series, which offer detail on the following:**
# 
# *Segment 1: Data Structures for Algebra*
# 
# * What Linear Algebra Is  
# * A Brief History of Algebra 
# * Tensors 
# * Scalars 
# * Vectors and Vector Transposition
# * Norms and Unit Vectors
# * Basis, Orthogonal, and Orthonormal Vectors
# * Arrays in NumPy  
# * Matrices 
# * Tensors in TensorFlow and PyTorch
# 
# *Segment 2: Common Tensor Operations* 
# 
# * Tensor Transposition
# * Basic Tensor Arithmetic
# * Reduction
# * The Dot Product
# * Solving Linear Systems
# 
# *Segment 3: Matrix Properties*
# 
# * The Frobenius Norm
# * Matrix Multiplication
# * Symmetric and Identity Matrices
# * Matrix Inversion
# * Diagonal Matrices
# * Orthogonal Matrices
# 

# ## Segment 1: Data Structures for Algebra
# 
# **Slides used to begin segment, with focus on introducing what linear algebra is, including hands-on paper and pencil exercises.**

# ### What Linear Algebra Is

# In[1]:
import sys


import numpy as np
import matplotlib.pyplot as plt

# start:
t = np.linspace(0, 40, 1000) # start, finish, n points

# Distance travelled by robber: $d = 2.5t

d_r = 2.5 * t 

# Distance travelled by sheriff: $d = 3(t-5)
d_s = 3 * (t-5)

# Plotting distance travelled over time:
fig, ax = plt.subplots()
plt.title('A Bank Robber Caught')
plt.xlabel('time (in minutes)')
plt.ylabel('distance (in km)')
ax.set_xlim([0, 40])
ax.set_ylim([0, 100])
ax.plot(t, d_r, c='green')
ax.plot(t, d_s, c='brown')
plt.axvline(x=30, color='purple', linestyle='--')
_ = plt.axhline(y=75, color='purple', linestyle='--')



#%% solar panel example:
t = np.linspace(0, 50, 1000) # start, finish, n points

m1 = 1*t
m2 = 4 * (t-30)

fig, ax = plt.subplots()
plt.title('A Bank Robber Caught')
plt.xlabel('time (in minutes)')
plt.ylabel('distance (in km)')
ax.set_xlim([0, 50])
ax.set_ylim([0, 100])
ax.plot(t, m1, c='green')
ax.plot(t, m2, c='brown')
plt.axvline(x=40, color='purple', linestyle='--')
_ = plt.axhline(y=40, color='purple', linestyle='--')




#%% ### Scalars (Rank 0 Tensors) in Base Python


x = 25
x
type(x) # if we'd like more specificity (e.g., int16, uint8), we need NumPy or another numeric library

y = 3
py_sum = x + y
py_sum
type(py_sum)

x_float = 25.0
float_sum = x_float + y
float_sum
type(float_sum)



#%% ### Scalars in PyTorch
# 
# * PyTorch and TensorFlow are the two most popular *automatic differentiation* 
#           libraries (a focus of the [*Calculus I*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/3-calculus-i.ipynb) 
#           and [*Calculus II*](https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/4-calculus-ii.ipynb) subjects in the *ML Foundations* series) in Python, itself the most popular programming language in ML
# * PyTorch tensors are designed to be pythonic, i.e., to feel and behave like NumPy arrays
# * The advantage of PyTorch tensors relative to NumPy arrays is that they easily be used for operations on GPU (see [here](https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_tensor.html) for example) 
# * Documentation on PyTorch tensors, including available data types, is [here](https://pytorch.org/docs/stable/tensors.html)


import torch

x_pt = torch.tensor(25) # type specification optional, e.g.: dtype=torch.float16
x_pt
x_pt.shape

#%% ### Scalars in TensorFlow (version 2.0 or later)
# 
# Tensors created with a wrapper, all of which [you can read about here](https://www.tensorflow.org/guide/tensor):  
# 
# * `tf.Variable`
# * `tf.constant`
# * `tf.placeholder`
# * `tf.SparseTensor`
# 
# Most widely-used is `tf.Variable`, which we'll use here. 
# 
# As with TF tensors, in PyTorch we can similarly perform operations, and we can easily convert to and from NumPy arrays
# 
# Also, a full list of tensor data types is available [here](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType).

import tensorflow as tf

x_tf = tf.Variable(25, dtype=tf.int16) # dtype is optional
x_tf
x_tf.shape
y_tf = tf.Variable(3, dtype=tf.int16)
x_tf + y_tf

# same as above (explicit)
tf_sum = tf.add(x_tf, y_tf)
tf_sum

tf_sum.numpy()  # note that NumPy operations automatically convert 
                # tensors to NumPy arrays, and vice versa

type(tf_sum.numpy())

tf_float = tf.Variable(25., dtype=tf.float16)
tf_float


#%% ### Vectors (Rank 1 Tensors) in NumPy

x = np.array([25, 2, 5]) # type argument is optional, e.g.: dtype=np.float16
x

len(x) # returns length of vector
x.shape # tuple of dimensions
type(x) # specifies type
x[0] # zero-indexed
type(x[0]) #type of element (int n this case)

# ### Vector Transposition
# Transposing a regular 1-D array has no effect...
x_t = x.T
x_t

x_t.shape

# ...but it does we use nested "matrix-style" brackets: 
y = np.array([[25, 2, 5]])
y
y.shape

# ...but can transpose a matrix with a dimension of length 1, 
#    which is mathematically equivalent: 
y_t = y.T
y_t
y_t.shape # this is a column vector as it has 3 rows and 1 column

# Column vector can be transposed back to original row vector: 
y_t.T 
y_t.T.shape 


#%% ### Zero Vectors
# Have no effect if added to another vector

z = np.zeros(3) 
z


#%% ### Vectors in PyTorch and TensorFlow

#pytorch
x_pt = torch.tensor([25, 2, 5])
x_pt

#tensorflow
x_tf = tf.Variable([25, 2, 5])
x_tf




#%% ### $L^2$ Norm

x
(25**2 + 2**2 + 5**2)**(1/2)
np.linalg.norm(x)
# So, if units in this 3-dimensional vector space are meters, 
# then the vector $x$ has a length of 25.6m



#%% ### $L^1$ Norm

x
np.abs(25) + np.abs(2) + np.abs(5)


#%% ### Squared $L^2$ Norm

x
(25**2 + 2**2 + 5**2)
# we'll cover tensor multiplication more soon but to prove point quickly: 
np.dot(x, x)
assert np.dot(x,x) == (25**2 + 2**2 + 5**2)



#%% ### Max Norm

x
np.max([np.abs(25), np.abs(2), np.abs(5)])



#%% ### Orthogonal Vectors

i = np.array([1, 0])
i
j = np.array([0, 1])
j
np.dot(i, 2j) # detail on the dot operation coming up...




#%% ### Matrices (Rank 2 Tensors) in NumPy

# Use array() with nested brackets: 
X = np.array([[25, 2], [5, 26], [3, 7]])
X
X.shape
X.size

# Select left column of matrix X (zero-indexed)
X[:,0]

# Select middle row of matrix X: 
X[1,:]

# Another slicing-by-index example: 
X[0:2, 0:2]


#%% ### Matrices in PyTorch

X_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
X_pt

X_pt.shape # more pythonic
X_pt[1,:]


#%% ### Matrices in TensorFlow

X_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])
X_tf

tf.rank(X_tf)
tf.shape(X_tf)

X_tf[1,:]




#%% ### Higher-Rank Tensors

# As an example, rank 4 tensors are common for images, where each dimension corresponds to: 

# 1. Number of images in training batch, e.g., 32
# 2. Image height in pixels, e.g., 28 for [MNIST digits](http://yann.lecun.com/exdb/mnist/)
# 3. Image width in pixels, e.g., 28
# 4. Number of color channels, e.g., 3 for full-color images (RGB)

images_pt = torch.zeros([32, 28, 28, 3])


# images_pt
images_tf = tf.zeros([32, 28, 28, 3])

#%%










#====================================PART 2===============================




#%% ## Segment 2: Common Tensor Operations

# ### Tensor Transposition

X
X.T 
X_pt.T # pytorch
tf.transpose(X_tf) # less Pythonic


#%% ### Basic Arithmetical Properties

# Adding or multiplying with scalar applies operation to all elements 
# and tensor shape is retained: 

X*2
X+2
X*2+2

X_pt*2+2 # Python operators are overloaded; 
         # could alternatively use torch.mul() or torch.add()

# alternative (explicit, but not needed):
torch.add(torch.mul(X_pt, 2), 2)


X_tf*2+2 # Operators likewise overloaded; could equally use tf.multiply() tf.add()
tf.add(tf.multiply(X_tf, 2), 2)


# If two tensors have the same size, operations are often 
# by default applied element-wise. 
# This is **not matrix multiplication**, 
# which we'll cover later, but is rather 
# called the **Hadamard product** or 
# simply the **element-wise product**. 

# The mathematical notation is $A \odot X$

X
A = X+2
A
A + X
A * X #Hadamard product / element-wise product
A_pt = X_pt + 2

A_pt + X_pt
A_pt * X_pt
A_tf = X_tf + 2
A_tf + X_tf
A_tf * X_tf


#%% ### Reduction

# Calculating the sum across all elements of a tensor is a common operation. 
# For example: 
    # * For vector ***x*** of length *n*, we calculate $\sum_{i=1}^{n} x_i$
    # * For matrix ***X*** with *m* by *n* dimensions, we calculate $\sum_{i=1}^{m} \sum_{j=1}^{n} X_{i,j}$

X
X.sum()

torch.sum(X_pt)
tf.reduce_sum(X_tf)

# Can also be done along one specific axis alone, e.g.:
X.sum(axis=0) # summing over all rows
X.sum(axis=1) # summing over all columns
torch.sum(X_pt, 0)
tf.reduce_sum(X_tf, 1)


# Many other operations can be applied with reduction along all or a selection of axes, e.g.:
# 
# * maximum
# * minimum
# * mean
# * product
# 
# They're fairly straightforward and used less often than summation, so you're welcome to look them up in library docs if you ever need them.

#%% ### The Dot Product

# If we have two vectors (say, ***x*** and ***y***) with the same length *n*, we can calculate the dot product between them. This is annotated several different ways, including the following: 
# 
# * $x \cdot y$
# * $x^Ty$
# * $\langle x,y \rangle$
# Regardless which notation you use (I prefer the first), the calculation is the same; we calculate products in an element-wise fashion and then sum reductively across the products to a scalar value. That is, $x \cdot y = \sum_{i=1}^{n} x_i y_i$
# The dot product is ubiquitous in deep learning: It is performed at every artificial neuron in a deep neural network, which may be made up of millions (or orders of magnitude more) of these neurons.
x

y = np.array([0, 1, 2])
y
25*0 + 2*1 + 5*2
np.dot(x, y)
x_pt
y_pt = torch.tensor([0, 1, 2])
y_pt

# numpy
np.dot(x_pt, y_pt)


# must be floats:
torch.dot(torch.tensor([25, 2, 5.]), torch.tensor([0, 1, 2.]))

#tensorflow:
x_tf
y_tf = tf.Variable([0, 1, 2])
y_tf
tf.reduce_sum(tf.multiply(x_tf, y_tf))






#%% ### Solving Linear Systems

# In the **Substitution** example, the two equations in the system are: 
# $$ y = 3x $$
# $$ -5x + 2y = 2 $$
# 
# The second equation can be rearranged to isolate $y$: 
# $$ 2y = 2 + 5x $$
# $$ y = \frac{2 + 5x}{2} = 1 + \frac{5x}{2} $$

x = np.linspace(-10, 10, 1000) # start, finish, n points

y1 = 3 * x
y2 = 1 + (5*x)/2

fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')
ax.set_xlim([0, 3])
ax.set_ylim([0, 8])
ax.plot(x, y1, c='green')
ax.plot(x, y2, c='brown')
plt.axvline(x=2, color='purple', linestyle='--')
_ = plt.axhline(y=6, color='purple', linestyle='--')


# In the **Elimination** example, the two equations in the system are:
# $$ 2x - 3y = 15 $$
# $$ 4x + 10y = 14 $$
# 
# Both equations can be rearranged to isolate $y$. Starting with the first equation: 
# $$ -3y = 15 - 2x $$
# $$ y = \frac{15 - 2x}{-3} = -5 + \frac{2x}{3} $$
# 
# Then for the second equation: 
# $$ 4x + 10y = 14 $$
# $$ 2x + 5y = 7 $$
# $$ 5y = 7 - 2x $$
# $$ y = \frac{7 - 2x}{5} $

y1 = -5 + (2*x)/3

y2 = (7-2*x)/5


fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')

# Add x and y axes: 
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')

ax.set_xlim([-2, 10])
ax.set_ylim([-6, 4])
ax.plot(x, y1, c='green')
ax.plot(x, y2, c='brown')
plt.axvline(x=6, color='purple', linestyle='--')
_ = plt.axhline(y=-1, color='purple', linestyle='--')


#%% ## Segment 3: Matrix Properties

# ### Frobenius Norm

X = np.array([[1, 2], [3, 4]])
X
(1**2 + 2**2 + 3**2 + 4**2)**(1/2)
np.linalg.norm(X) # same function as for vector L2 norm
X_pt = torch.tensor([[1, 2], [3, 4.]]) # torch.norm() supports floats only
torch.norm(X_pt)
X_tf = tf.Variable([[1, 2], [3, 4.]]) # tf.norm() also supports floats only
tf.norm(X_tf)


#%% ### Matrix Multiplication (with a Vector)
A = np.array([[3, 4], [5, 6], [7, 8]])
A
b = np.array([1, 2])
b
np.dot(A, b) # even though technically dot products are between vectors only
A_pt = torch.tensor([[3, 4], [5, 6], [7, 8]])
A_pt
b_pt = torch.tensor([1, 2])
b_pt
torch.matmul(A_pt, b_pt) # like np.dot(), automatically infers dims in order to perform dot product, matvec, or matrix multiplication
A_tf = tf.Variable([[3, 4], [5, 6], [7, 8]])
A_tf
b_tf = tf.Variable([1, 2])
b_tf
tf.linalg.matvec(A_tf, b_tf)


#%% ### Matrix Multiplication (with Two Matrices)

A
B = np.array([[1, 9], [2, 0]])
B
np.dot(A, B)

# Note that matrix multiplication is not "commutative" (i.e., $AB \neq BA$) so uncommenting the following line will throw a size mismatch error:

# np.dot(B, A)

B_pt = torch.from_numpy(B) # much cleaner than TF conversion
B_pt

# another neat way to create the same tensor with transposition: 
B_pt = torch.tensor([[1, 2], [9, 0]]).T
B_pt
torch.matmul(A_pt, B_pt) # no need to change functions, unlike in TF
B_tf = tf.convert_to_tensor(B, dtype=tf.int32)
B_tf
tf.matmul(A_tf, B_tf)


#%% ### Symmetric Matrices

X_sym = np.array([[0, 1, 2], [1, 7, 8], [2, 8, 9]])
X_sym
X_sym.T
X_sym.T == X_sym


#%% ### Identity Matrices

I = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
I

x_pt = torch.tensor([25, 2, 5])
x_pt

torch.matmul(I, x_pt)


#%% ### Answers to Matrix Multiplication Qs

M_q = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
M_q

V_q = torch.tensor([[-1, 1, -2], [0, 1, 2]]).T
V_q

torch.matmul(M_q, V_q)


#%% ### Matrix Inversion

X = np.array([[4, 2], [-5, -3]])
X

Xinv = np.linalg.inv(X)
Xinv

# As a quick aside, let's prove that $X^{-1}X = I_n$ as per the slides: 
np.dot(Xinv, X)

# ...and now back to solving for the unknowns in $w$: 
y = np.array([4, -7])
y

w = np.dot(Xinv, y)
w


# Show that $y = Xw$: 
np.dot(X, w)


# **Geometric Visualization**
# 
# Recalling from the slides that the two equations in the system are:
# $$ 4b + 2c = 4 $$
# $$ -5b - 3c = -7 $$
# 
# Both equations can be rearranged to isolate a variable, say $c$. Starting with the first equation: 
# $$ 4b + 2c = 4 $$
# $$ 2b + c = 2 $$
# $$ c = 2 - 2b $$
# 
# Then for the second equation: 
# $$ -5b - 3c = -7 $$
# $$ -3c = -7 + 5b $$
# $$ c = \frac{-7 + 5b}{-3} = \frac{7 - 5b}{3} $$


b = np.linspace(-10, 10, 1000) # start, finish, n points
c1 = 2 - 2*b
c2 = (7-5*b)/3

fig, ax = plt.subplots()
plt.xlabel('b', c='darkorange')
plt.ylabel('c', c='brown')

plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')

ax.set_xlim([-2, 3])
ax.set_ylim([-1, 5])
ax.plot(b, c1, c='purple')
ax.plot(b, c2, c='purple')
plt.axvline(x=-1, color='green', linestyle='--')
_ = plt.axhline(y=4, color='green', linestyle='--')


#%% In PyTorch and TensorFlow:

torch.inverse(torch.tensor([[4, 2], [-5, -3.]])) # float type

tf.linalg.inv(tf.Variable([[4, 2], [-5, -3.]])) # also float


# **Exercises**:
# 
# 1. As done with NumPy above, use PyTorch to calculate $w$ from $X$ and $y$. Subsequently, confirm that $y = Xw$.
# 2. Repeat again, now using TensorFlow.

# **Return to slides here.**

#%% ### Matrix Inversion Where No Solution

X = np.array([[-4, 1], [-8, 2]])
X
# Uncommenting the following line results in a "singular matrix" error
# Xinv = np.linalg.inv(X)


# Feel free to try inverting a non-square matrix; this will throw an error too.
# 
# **Return to slides here.**

# ### Orthogonal Matrices
# 
# These are the solutions to Exercises 3 and 4 on **orthogonal matrices** from the slides.
# 
# For Exercise 3, to demonstrate the matrix $I_3$ has mutually orthogonal columns, we show that the dot product of any pair of columns is zero: 

I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
I

column_1 = I[:,0]
column_1

column_2 = I[:,1]
column_2

column_3 = I[:,2]
column_3

np.dot(column_1, column_2)
np.dot(column_1, column_3)
np.dot(column_2, column_3)


# We can use the `np.linalg.norm()` method from earlier in the notebook to demonstrate that each column of $I_3$ has unit norm: 
np.linalg.norm(column_1)
np.linalg.norm(column_2)
np.linalg.norm(column_3)


# Since the matrix $I_3$ has mutually orthogonal columns and each column has unit norm, the column vectors of $I_3$ are *orthonormal*. Since $I_3^T = I_3$, this means that the *rows* of $I_3$ must also be orthonormal. 
# 
# Since the columns and rows of $I_3$ are orthonormal, $I_3$ is an *orthogonal matrix*.

# For Exercise 4, let's repeat the steps of Exercise 3 with matrix *K* instead of $I_3$. We could use NumPy again, but for fun I'll use PyTorch instead. (You're welcome to try it with TensorFlow if you feel so inclined.)

K = torch.tensor([[2/3, 1/3, 2/3], [-2/3, 2/3, 1/3], [1/3, 2/3, -2/3]])
K
Kcol_1 = K[:,0]
Kcol_1
Kcol_2 = K[:,1]
Kcol_2
Kcol_3 = K[:,2]
Kcol_3
torch.dot(Kcol_1, Kcol_2)
torch.dot(Kcol_1, Kcol_3)
torch.dot(Kcol_2, Kcol_3)

# We've now determined that the columns of $K$ are orthogonal.
torch.norm(Kcol_1)
torch.norm(Kcol_2)
torch.norm(Kcol_3)


# We've now determined that, in addition to being orthogonal, the columns of $K$ have unit norm, therefore they are orthonormal. 
# 
# To ensure that $K$ is an orthogonal matrix, we would need to show that not only does it have orthonormal columns but it has orthonormal rows are as well. Since $K^T \neq K$, we can't prove this quite as straightforwardly as we did with $I_3$. 
# 
# One approach would be to repeat the steps we used to determine that $K$ has orthogonal columns with all of the matrix's rows (please feel free to do so). Alternatively, we can use an orthogonal matrix-specific equation from the slides, $A^TA = I$, to demonstrate that $K$ is orthogonal in a single line of code: 

torch.matmul(K.T, K)


# Notwithstanding rounding errors that we can safely ignore, this confirms that $K^TK = I$ and therefore $K$ is an orthogonal matrix. 
