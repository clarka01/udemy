#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jonkrohn/ML-foundations/blob/master/notebooks/2-linear-algebra-ii.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Linear Algebra II: Matrix Operations

# This topic, *Linear Algebra II: Matrix Operations*, builds on the basics of linear algebra. It is essential because these intermediate-level manipulations of tensors lie at the heart of most machine learning approaches and are especially predominant in deep learning. 
# 
# Through the measured exposition of theory paired with interactive examples, you’ll develop an understanding of how linear algebra is used to solve for unknown values in high-dimensional spaces as well as to reduce the dimensionality of complex spaces. The content covered in this topic is itself foundational for several other topics in the *Machine Learning Foundations* series, especially *Probability & Information Theory* and *Optimization*. 

# Over the course of studying this topic, you'll: 
# 
# * Develop a geometric intuition of what’s going on beneath the hood of machine learning algorithms, including those used for deep learning. 
# * Be able to more intimately grasp the details of machine learning papers as well as all of the other subjects that underlie ML, including calculus, statistics, and optimization algorithms. 
# * Reduce the dimensionalty of complex spaces down to their most informative elements with techniques such as eigendecomposition, singular value decomposition, and principal component analysis.

# **Note that this Jupyter notebook is not intended to stand alone. It is the companion code to a lecture or to videos from Jon Krohn's [Machine Learning Foundations](https://github.com/jonkrohn/ML-foundations) series, which offer detail on the following:**
# 
# *Review of Introductory Linear Algebra*
# 
# * Modern Linear Algebra Applications
# * Tensors, Vectors, and Norms
# * Matrix Multiplication
# * Matrix Inversion
# * Identity, Diagonal and Orthogonal Matrices
# 
# *Segment 2: Eigendecomposition*
# 
# * Affine Transformation via Matrix Application
# * Eigenvectors and Eigenvalues
# * Matrix Determinants
# * Matrix Decomposition 
# * Applications of Eigendecomposition
# 
# *Segment 3: Matrix Operations for Machine Learning*
# 
# * Singular Value Decomposition (SVD)
# * The Moore-Penrose Pseudoinverse
# * The Trace Operator
# * Principal Component Analysis (PCA): A Simple Machine Learning Algorithm
# * Resources for Further Study of Linear Algebra

#%% ## Segment 1: Review of Introductory Linear Algebra




import numpy as np
import torch


#%% ### Vector Transposition




x = np.array([25, 2, 5])
x

x.shape

x = np.array([[25, 2, 5]])
x


x.shape


x.T


x.T.shape

x_p = torch.tensor([25, 2, 5])
x_p

x_p.T


x_p.view(3, 1) # "view" because we're changing output but not the way x is stored in memory


#%% ## $L^2$ Norm

x

(25**2 + 2**2 + 5**2)**(1/2)

np.linalg.norm(x)


# So, if units in this 3-dimensional vector space are meters, then the vector $x$ has a length of 25.6m

# the following line of code will fail because torch.norm() requires input to be float not integer
# torch.norm(p)

torch.norm(torch.tensor([25, 2, 5.]))

#%% ### Matrices
X = np.array([[25, 2], [5, 26], [3, 7]])
X

X.shape

X_p = torch.tensor([[25, 2], [5, 26], [3, 7]])
X_p
X_p.shape

#%% ### Matrix Transposition

X
X.T
X_p.T

# Scalars are applied to each element of matrix:
X*3

X*3+3


X_p*3


X_p*3+3


# Using the multiplication operator on two tensors of the same size in PyTorch (or Numpy or TensorFlow) applies element-wise operations. This is the **Hadamard product** (denoted by the $\odot$ operator, e.g., $A \odot B$) *not* **matrix multiplication**: 

A = np.array([[3, 4], [5, 6], [7, 8]])
A

X

X * A

A_p = torch.tensor([[3, 4], [5, 6], [7, 8]])
A_p


X_p * A_p

# Matrix multiplication with a vector: 

b = np.array([1, 2])
b
np.dot(A, b) # even though technically dot products is between 2 vectors

b_p = torch.tensor([1, 2])
b_p


torch.matmul(A_p, b_p)


# Matrix multiplication with two matrices:

B = np.array([[1, 9], [2, 0]])
B



np.dot(A, B) # note first column is same as Xb


B_p = torch.tensor([[1, 9], [2, 0]])
B_p


torch.matmul(A_p, B_p) 


#%% ### Matrix Inversion

X = np.array([[4, 2], [-5, -3]])
X

Xinv = np.linalg.inv(X)
Xinv

y = np.array([4, -7])
y

w = np.dot(Xinv, y)
w


# Show that $y = Xw$: 

np.dot(X, w)

X_p = torch.tensor([[4, 2], [-5, -3.]]) # note that torch.inverse() requires floats
X_p

Xinv_p = torch.inverse(X_p)
Xinv_p

y_p = torch.tensor([4, -7.])
y_p


w_p = torch.matmul(Xinv_p, y_p)
w_p

torch.matmul(X_p, w_p)


#%% ## Segment 2: Eigendecomposition

### Affine Transformation via Matrix Application

# Let's say we have a vector $v$:

v = np.array([3, 1])
v

#%%
# Let's plot $v$ using my `plot_vectors()` function 
# (which is based on Hadrien Jean's `plotVectors()` function 
# from [this notebook](https://github.com/hadrienj/deepLearningBook-Notes/blob/master/2.7%20Eigendecomposition/2.7%20Eigendecomposition.ipynb), 
# under [MIT license](https://github.com/hadrienj/deepLearningBook-Notes/blob/master/LICENSE)).

import matplotlib.pyplot as plt

def plot_vectors(vectors, colors):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each. 

    Arguments
    ---------
    vectors: list of lists or of arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]] 
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
        
    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    """
    plt.figure()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i in range(len(vectors)):
        x = np.concatenate([[0,0],vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=colors[i],)

plot_vectors([v], ['lightblue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)


# "Applying" a matrix to a vector (i.e., performing matrix-vector multiplication) can linearly transform the vector, e.g, rotate it or rescale it.

# The identity matrix, introduced earlier, is the exception that proves the rule: Applying an identity matrix does not transform the vector: 


I = np.array([[1, 0], [0, 1]])
I

Iv = np.dot(I, v)
Iv

v == Iv

plot_vectors([Iv], ['blue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)

# In contrast, consider this matrix (let's call it $E$) that flips vectors over the $x$-axis: 

E = np.array([[1, 0], [0, -1]])
E

Ev = np.dot(E, v)
Ev

plot_vectors([v, Ev], ['lightblue', 'blue'])
plt.xlim(-1, 5)
_ = plt.ylim(-3, 3)


# Or, this matrix, $F$, which flips vectors over the $y$-axis: 
F = np.array([[-1, 0], [0, 1]])
F 

Fv = np.dot(F, v)
Fv

plot_vectors([v, Fv], ['lightblue', 'blue'])
plt.xlim(-4, 4)
_ = plt.ylim(-1, 5)


# Applying a flipping matrix is an example of an **affine transformation**: 
# a change in geometry that may adjust distances or angles between vectors, 
# but preserves parallelism between them.
# 
# In addition to flipping a matrix over an axis (a.k.a., *reflection*), 
# other common affine transformations include:
# * *Scaling* (changing the length of vectors)
# * *Shearing* (example of this on the Mona Lisa coming up shortly)
# * *Rotation* 
# 
# (See [here](https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/) for an outstanding blog post on affine transformations in Python, including how to apply them to images as well as vectors.)

# A single matrix can apply multiple affine transforms simultaneously (e.g., flip over an axis and rotate 45 degrees). As an example, let's see what happens when we apply this matrix $A$ to the vector $v$: 

A = np.array([[-1, 4], [2, -2]])
A

Av = np.dot(A, v)
Av

plot_vectors([v, Av], ['lightblue', 'blue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)

# Another example of applying A:
v2 = np.array([2, 1])
plot_vectors([v2, np.dot(A, v2)], ['lightgreen', 'green'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)

# We can concatenate several vectors together into a matrix (say, $V$), 
# where each column is a separate vector. 
# Then, whatever linear transformations we apply to $V$ 
# will be independently applied to each column (vector): 

v

# recall that we need to convert array to 2D to transpose into column, e.g.:
np.matrix(v).T 

v3 = np.array([-3, -1]) # mirror image of v over both axes
v4 = np.array([-1, 1])

V = np.concatenate((np.matrix(v).T, 
                    np.matrix(v2).T,
                    np.matrix(v3).T,
                    np.matrix(v4).T), 
                   axis=1)
V

IV = np.dot(I, V)
IV

AV = np.dot(A, V)
AV

# function to convert column of matrix to 1D vector: 
def vectorfy(mtrx, clmn):
    return np.array(mtrx[:,clmn]).reshape(-1)

vectorfy(V, 0)

vectorfy(V, 0) == v

plot_vectors([vectorfy(V, 0), vectorfy(V, 1), vectorfy(V, 2), vectorfy(V, 3),
             vectorfy(AV, 0), vectorfy(AV, 1), vectorfy(AV, 2), vectorfy(AV, 3)], 
            ['lightblue', 'lightgreen', 'lightgray', 'orange',
             'blue', 'green', 'gray', 'red'])
plt.xlim(-4, 6)
_ = plt.ylim(-5, 5)


# Now that we can appreciate the linear transformation of vectors by matrices, let's move on to working with eigenvectors and eigenvalues...
# 
# **Return to slides here.**

#%% ### Eigenvectors and Eigenvalues

# An **eigenvector** (*eigen* is German for "typical"; we could translate *eigenvector* to "characteristic vector") is a special vector $v$ such that when it is transformed by some matrix (let's say $A$), the product $Av$ has the exact same direction as $v$.
# 
# An **eigenvalue** is a scalar (traditionally represented as $\lambda$) that simply scales the eigenvector $v$ such that the following equation is satisfied: 
# 
# $Av = \lambda v$

# Easiest way to understand this is to work through an example: 

A


# Eigenvectors and eigenvalues can be derived algebraically (e.g., with the [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm), which was independently developed in the 1950s by both [Vera Kublanovskaya](https://en.wikipedia.org/wiki/Vera_Kublanovskaya) and John Francis), however this is outside scope of the *ML Foundations* series. We'll cheat with NumPy `eig()` method, which returns a tuple of: 
# 
# * a vector of eigenvalues
# * a matrix of eigenvectors

lambdas, V = np.linalg.eig(A) 


# The matrix contains as many eigenvectors as there are columns of A: 

V # each column is a separate eigenvector v

# With a corresponding eigenvalue for each eigenvector:

lambdas


# Let's confirm that $Av = \lambda v$ for the first eigenvector: 

v = V[:,0] 
v

lambduh = lambdas[0] # note that "lambda" is reserved term in Python
lambduh

Av = np.dot(A, v)
Av

lambduh * v

plot_vectors([Av, v], ['blue', 'lightblue'])
plt.xlim(-1, 2)
_ = plt.ylim(-1, 2)

# And again for the second eigenvector of A: 

v2 = V[:,1]
v2

lambda2 = lambdas[1]
lambda2

Av2 = np.dot(A, v2)
Av2

lambda2 * v2


plot_vectors([Av, v, Av2, v2], 
            ['blue', 'lightblue', 'green', 'lightgreen'])
plt.xlim(-1, 4)
_ = plt.ylim(-3, 2)


# Using the PyTorch `eig()` method, we can do exactly the same: 
A

A_p = torch.tensor([[-1, 4], [2, -2.]]) # must be float for PyTorch eig()
A_p

lambdas_cplx, V_cplx = torch.linalg.eig(A_p) # outputs complex numbers because real matrices can have complex eigenvectors


V_cplx # complex-typed values with "0.j" imaginary part are in fact real numbers

V_p = V_cplx.float()
V_p

v_p = V_p[:,0]
v_p

lambdas_cplx

lambdas_p = lambdas_cplx.float()
lambdas_p

lambda_p = lambdas_p[0]
lambda_p

Av_p = torch.matmul(A_p, v_p) # matmul() expects float-typed tensors
Av_p

lambda_p * v_p

v2_p = V_p[:,1]
v2_p

lambda2_p = lambdas_p[1]
lambda2_p

Av2_p = torch.matmul(A_p.float(), v2_p.float())
Av2_p

lambda2_p.float() * v2_p.float()

plot_vectors([Av_p.numpy(), v_p.numpy(), Av2_p.numpy(), v2_p.numpy()], 
            ['blue', 'lightblue', 'green', 'lightgreen'])
plt.xlim(-1, 4)
_ = plt.ylim(-3, 2)


#%% ### Eigenvectors in >2 Dimensions

# While plotting gets trickier in higher-dimensional spaces, we can nevertheless find and use eigenvectors with more than two dimensions. Here's a 3D example (there are three dimensions handled over three rows): 

X = np.array([[25, 2, 9], [5, 26, -5], [3, 7, -1]])
X

lambdas_X, V_X = np.linalg.eig(X) 

V_X # one eigenvector per column of X

lambdas_X # a corresponding eigenvalue for each eigenvector

# Confirm $Xv = \lambda v$ for an example eigenvector: 
v_X = V_X[:,0] 
v_X

lambda_X = lambdas_X[0] 
lambda_X

np.dot(X, v_X) # matrix multiplication

lambda_X * v_X


# **Exercises**:
# 
# 1. Use PyTorch to confirm $Xv = \lambda v$ for the first eigenvector of $X$.
# 2. Confirm $Xv = \lambda v$ for the remaining eigenvectors of $X$ (you can use NumPy or PyTorch, whichever you prefer).

#%% ### 2x2 Matrix Determinants

X = np.array([[4, 2], [-5, -3]])
X

np.linalg.det(X)

N = np.array([[-4, 1], [-8, 2]])
N

np.linalg.det(N)

# Uncommenting the following line results in a "singular matrix" error
# Ninv = np.linalg.inv(N)
N = torch.tensor([[-4, 1], [-8, 2.]]) # must use float not int

torch.det(N) 


#%% ### Generalizing Determinants

X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
X

np.linalg.det(X)


#%% ### Determinants & Eigenvalues
lambdas, V = np.linalg.eig(X)
lambdas

np.product(lambdas)

# Here's $|\text{det}(X)|$ in NumPy: 
np.abs(np.linalg.det(X))

# Let's use a matrix $B$, which is composed of basis vectors, to explore the impact of applying matrices with varying $|\text{det}(X)|$ values: 
B = np.array([[1, 0], [0, 1]])
B

plot_vectors([vectorfy(B, 0), vectorfy(B, 1)],
            ['lightblue', 'lightgreen'])
plt.xlim(-1, 3)
_ = plt.ylim(-1, 3)


# Let's start by applying the matrix $N$ to $B$, recalling from earlier that $N$ is singular: 
N
np.linalg.det(N)
NB = np.dot(N, B)
NB

plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(NB, 0), vectorfy(NB, 1)],
            ['lightblue', 'lightgreen', 'blue', 'green'])
plt.xlim(-6, 6)
_ = plt.ylim(-9, 3)

lambdas, V = np.linalg.eig(N)
lambdas
# Aha! If any one of a matrix's eigenvalues is zero, then the product of the eigenvalues must be zero and the determinant must also be zero. 

# Now let's try applying $I_2$ to $B$: 
I

np.linalg.det(I)

IB = np.dot(I, B)
IB

plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(IB, 0), vectorfy(IB, 1)],
            ['lightblue', 'lightgreen', 'blue', 'green'])
plt.xlim(-1, 3)
_ = plt.ylim(-1, 3)

lambdas, V = np.linalg.eig(I)
lambdas

# All right, so applying an identity matrix isn't the most exciting operation in the world. Let's now apply this matrix $J$ which is more interesting: 
J = np.array([[-0.5, 0], [0, 2]])
J

np.linalg.det(J)

np.abs(np.linalg.det(J))

JB = np.dot(J, B)
JB

plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(JB, 0), vectorfy(JB, 1)],
            ['lightblue', 'lightgreen', 'blue', 'green'])
plt.xlim(-1, 3)
_ = plt.ylim(-1, 3)

lambdas, V = np.linalg.eig(J)
lambdas

# Finally, let's apply the matrix $D$, which scales vectors by doubling along both the $x$ and $y$ axes: 
D = I*2
D

np.linalg.det(D)

DB = np.dot(D, B)
DB

plot_vectors([vectorfy(B, 0), vectorfy(B, 1), vectorfy(DB, 0), vectorfy(DB, 1)],
            ['lightblue', 'lightgreen', 'blue', 'green'])
plt.xlim(-1, 3)
_ = plt.ylim(-1, 3)

lambdas, V = np.linalg.eig(D)
lambdas



#%% ### Eigendecomposition

# The **eigendecomposition** of some matrix $A$ is 
# 
# $A = V \Lambda V^{-1}$
# 
# Where: 
# 
# * As in examples above, $V$ is the concatenation of all the eigenvectors of $A$
# * $\Lambda$ (upper-case $\lambda$) is the diagonal matrix diag($\lambda$). Note that the convention is to arrange the lambda values in descending order; as a result, the first eigenvalue (and its associated eigenvector) may be a primary characteristic of the matrix $A$.

# This was used earlier as a matrix X; it has nice clean integer eigenvalues...
A = np.array([[4, 2], [-5, -3]]) 
A

lambdas, V = np.linalg.eig(A)
V

Vinv = np.linalg.inv(V)
Vinv

Lambda = np.diag(lambdas)
Lambda

# Confirm that $A = V \Lambda V^{-1}$: 
np.dot(V, np.dot(Lambda, Vinv))


# Eigendecomposition is not possible with all matrices. And in some cases where it is possible, the eigendecomposition involves complex numbers instead of straightforward real numbers. 
# 
# In machine learning, however, we are typically working with real symmetric matrices, which can be conveniently and efficiently decomposed into real-only eigenvectors and real-only eigenvalues. If $A$ is a real symmetric matrix then...
# 
# $A = Q \Lambda Q^T$
# 
# ...where $Q$ is analogous to $V$ from the previous equation except that it's special because it's an orthogonal matrix. 

A = np.array([[2, 1], [1, 2]])
A

lambdas, Q = np.linalg.eig(A)

lambdas

Lambda = np.diag(lambdas)
Lambda

Q

# Let's confirm $A = Q \Lambda Q^T$: 
np.dot(Q, np.dot(Lambda, Q.T))

# (As a quick aside, we can demostrate that $Q$ is an orthogonal matrix because $Q^TQ = QQ^T = I$.)

np.dot(Q.T, Q)
np.dot(Q, Q.T)


# **Exercises**:
# 
# 1. Use PyTorch to decompose the matrix $P$ (below) into its components $V$, $\Lambda$, and $V^{-1}$. Confirm that $P = V \Lambda V^{-1}$.
# 2. Use PyTorch to decompose the symmetric matrix $S$ (below) into its components $Q$, $\Lambda$, and $Q^T$. Confirm that $S = Q \Lambda Q^T$.
P = torch.tensor([[25, 2, -5], [3, -2, 1], [5, 7, 4.]])
P

S = torch.tensor([[25, 2, -5], [2, -2, 1], [-5, 1, 4.]])
S

#%% ## Segment 3: Matrix Operations for ML

### Singular Value Decomposition (SVD)

# As on slides, SVD of matrix $A$ is: 
# 
# $A = UDV^T$
# 
# Where: 
# 
# * $U$ is an orthogonal $m \times m$ matrix; its columns are the **left-singular vectors** of $A$.
# * $V$ is an orthogonal $n \times n$ matrix; its columns are the **right-singular vectors** of $A$.
# * $D$ is a diagonal $m \times n$ matrix; elements along its diagonal are the **singular values** of $A$.
A = np.array([[-1, 2], [3, -2], [5, 7]])
A

U, d, VT = np.linalg.svd(A) # V is already transposed
U
VT
d
np.diag(d)
# $D$ must have the same dimensions as $A$ for $UDV^T$ matrix multiplication to be possible: 
D = np.concatenate((np.diag(d), [[0, 0]]), axis=0)
D

np.dot(U, np.dot(D, VT))


# SVD and eigendecomposition are closely related to each other: 
# 
# * Left-singular vectors of $A$ = eigenvectors of $AA^T$.
# * Right-singular vectors of $A$ = eigenvectors of $A^TA$.
# * Non-zero singular values of $A$ = square roots of eigenvalues of $AA^T$ = square roots of eigenvalues of $A^TA$
# 
# **Exercise**: Using the matrix `P` from the preceding PyTorch exercises, demonstrate that these three SVD-eigendecomposition equations are true. 

#%% ### Image Compression via SVD

# The section features code adapted from [Frank Cleary's](https://gist.github.com/frankcleary/4d2bd178708503b556b0).
from PIL import Image
# Fetch photo of Oboe, a terrier, with the book *Deep Learning Illustrated*: 
get_ipython().system(' wget https://raw.githubusercontent.com/jonkrohn/DLTFpT/master/notebooks/oboe-with-book.jpg')

img = Image.open('oboe-with-book.jpg')
_ = plt.imshow(img)

# Convert image to grayscale so that we don't have to deal with the complexity of multiple color channels: 

imggray = img.convert('LA')
_ = plt.imshow(imggray)

# Convert data into numpy matrix, which doesn't impact image data: 
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
_ = plt.imshow(imgmat, cmap='gray')

# Calculate SVD of the image: 
U, sigma, V = np.linalg.svd(imgmat)

# As eigenvalues are arranged in descending order in diag($\lambda$) so too are singular values, by convention, arranged in descending order in $D$ (or, in this code, diag($\sigma$)). Thus, the first left-singular vector of $U$ and first right-singular vector of $V$ may represent the most prominent feature of the image: 

reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
_ = plt.imshow(reconstimg, cmap='gray')

# Additional singular vectors improve the image quality: 
for i in [2, 4, 8, 16, 32, 64]:
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()

# With 64 singular vectors, the image is reconstructed quite well, however the data footprint is much smaller than the original image:
imgmat.shape

full_representation = 4032*3024
full_representation

svd64_rep = 64*4032 + 64 + 64*3024
svd64_rep

svd64_rep/full_representation


# Specifically, the image represented as 64 singular vectors is 3.7% of the size of the original! 
# 
# Alongside images, we can use singular vectors for dramatic, lossy compression of other types of media files.

# **Return to slides here.**

#%% ### The Moore-Penrose Pseudoinverse

# Let's calculate the pseudoinverse $A^+$ of some matrix $A$ using the formula from the slides: 
# 
# $A^+ = VD^+U^T$
A

# As shown earlier, the NumPy SVD method returns $U$, $d$, and $V^T$:
U, d, VT = np.linalg.svd(A)
U
VT
d

# To create $D^+$, we first invert the non-zero values of $d$: 

D = np.diag(d)
D
1/8.669
1/4.104

# ...and then we would take the tranpose of the resulting matrix.
# 
# Because $D$ is a diagonal matrix, this can, however, be done in a single step by inverting $D$: 
Dinv = np.linalg.inv(D)
Dinv

# $D^+$ must have the same dimensions as $A^T$ in order for $VD^+U^T$ matrix multiplication to be possible: 
Dplus = np.concatenate((Dinv, np.array([[0, 0]]).T), axis=1)
Dplus

# (Recall $D$ must have the same dimensions as $A$ for SVD's $UDV^T$, but for MPP $U$ and $V$ have swapped sides around the diagonal matrix.)
# Now we have everything we need to calculate $A^+$ with $VD^+U^T$: 
np.dot(VT.T, np.dot(Dplus, U.T))


# Working out this derivation is helpful for understanding how Moore-Penrose pseudoinverses work, but unsurprisingly NumPy is loaded with an existing method `pinv()`: 
np.linalg.pinv(A)

# **Exercise** 
# Use the `torch.svd()` method to calculate the pseudoinverse of `A_p`, confirming that your result matches the output of `torch.pinverse(A_p)`: 
A_p = torch.tensor([[-1, 2], [3, -2], [5, 7.]])
A_p

torch.pinverse(A_p)

# For regression problems, we typically have many more cases ($n$, or rows of $X$) than features to predict (columns of $X$). Let's solve a miniature example of such an overdetermined situation. 
# 
# We have eight data points ($n$ = 8): 

x1 = [0, 1, 2, 3, 4, 5, 6, 7.] # E.g.: Dosage of drug for treating Alzheimer's disease
y = [1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37] # E.g.: Patient's "forgetfulness score"

title = 'Clinical Trial'
xlabel = 'Drug dosage (mL)'
ylabel = 'Forgetfulness'

fig, ax = plt.subplots()
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
_ = ax.scatter(x1, y)

# Although it appears there is only one predictor ($x_1$), our model requires a second one (let's call it $x_0$) in order to allow for a $y$-intercept. Without this second variable, the line we fit to the plot would need to pass through the origin (0, 0). The $y$-intercept is constant across all the points so we can set it equal to `1` across the board:
x0 = np.ones(8)
x0

# Concatenate $x_0$ and $x_1$ into a matrix $X$: 
X = np.concatenate((np.matrix(x0).T, np.matrix(x1).T), axis=1)
X

# From the slides, we know that we can calculate the weights $w$ using the equation $w = X^+y$: 
w = np.dot(np.linalg.pinv(X), y)
w

# The first weight corresponds to the $y$-intercept of the line, which is typically denoted as $b$: 
b = np.asarray(w).reshape(-1)[0]
b

# While the second weight corresponds to the slope of the line, which is typically denoted as $m$: 
m = np.asarray(w).reshape(-1)[1]
m

# With the weights we can plot the line to confirm it fits the points: 
fig, ax = plt.subplots()

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

ax.scatter(x1, y)

x_min, x_max = ax.get_xlim()
y_at_xmin = m*x_min + b
y_at_xmax = m*x_max + b

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_at_xmin, y_at_xmax], c='C01')


# **DO NOT return to slides here. Onward!**

#%% ### The Trace Operator

# Denoted as Tr($A$). Simply the sum of the diagonal elements of a matrix: $$\sum_i A_{i,i}$$
A = np.array([[25, 2], [5, 4]])
A
25 + 4
np.trace(A)


# The trace operator has a number of useful properties that come in handy while rearranging linear algebra equations, e.g.:
# 
# * Tr($A$) = Tr($A^T$)
# * Assuming the matrix shapes line up: Tr($ABC$) = Tr($CAB$) = Tr($BCA$)

# In particular, the trace operator can provide a convenient way to calculate a matrix's Frobenius norm: $$||A||_F = \sqrt{\mathrm{Tr}(AA^\mathrm{T})}$$

# **Exercises**
# 
# With the matrix `A_p` provided below: 
# 
# 1. Use the PyTorch trace method to calculate the trace of `A_p`.
# 2. Use the PyTorch Frobenius norm method and the trace method to demonstrate that $||A||_F = \sqrt{\mathrm{Tr}(AA^\mathrm{T})}$
A_p

#%% ### Principal Component Analysis

# This PCA example code is adapted from [here](https://jupyter.brynmawr.edu/services/public/dblank/CS371%20Cognitive%20Science/2016-Fall/PCA.ipynb).
from sklearn import datasets
iris = datasets.load_iris()

iris.data.shape
iris.get("feature_names")
iris.data[0:6,:]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(iris.data)
X.shape

X[0:6,:]
_ = plt.scatter(X[:, 0], X[:, 1])

iris.target.shape
iris.target[0:6]

unique_elements, counts_elements = np.unique(iris.target, return_counts=True)
np.asarray((unique_elements, counts_elements))

list(iris.target_names)
_ = plt.scatter(X[:, 0], X[:, 1], c=iris.target)


