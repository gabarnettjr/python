import numpy as np                          #import the numpy library
import matplotlib.pyplot as plt          #import the plotting library

#####################################################################

# The two input vectors, x and y.  These define the known data that
# must be matched by the quadratic approximation:

x = np.array([-1., 0., 1.])                            #x-coordinates
y = np.array([1., 3., 2.])                             #y-coordinates

#####################################################################

# Method 1 (build the quadratic function directly):

def quadratic(X):
    return (X-x[1]) * (X-x[2]) / (x[0]-x[1]) / (x[0]-x[2]) * y[0] + \
           (X-x[0]) * (X-x[2]) / (x[1]-x[0]) / (x[1]-x[2]) * y[1] + \
           (X-x[0]) * (X-x[1]) / (x[2]-x[0]) / (x[2]-x[1]) * y[2]

#####################################################################

# A dense set of x-coordinates where we  will plot both of the
# approximations to see what they look like:

X = np.linspace(x[0], x[-1], 200)

#####################################################################

# Method 2 (indirect but more flexible and generalizable):

e = np.ones(np.shape(x));      #short row-vector containing only ones
A = np.vstack((e, x, x**2)).T                     #Vandermonde matrix

# The column vector "lam" tells you how much of each basis function
# that you need in order to match the known data.  This line solves
# the linear system A*lam=y for lam:
lam = np.linalg.solve(A, y)

E = np.ones(np.shape(X))        #long row-vector containing only ones
B = np.vstack((E, X, X**2)).T
Q = B.dot(lam)

#####################################################################

# Plot the two quadratic functions at all of the x-coordinates
# in the vector X, and make sure that the two methods give the same
# result.  Also, plot the original data on the same graph:

plt.figure()
plt.plot(X, quadratic(X), '-')
plt.plot(X, Q, '--')
plt.plot(x, y, '*')
plt.show()