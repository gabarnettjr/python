import numpy as np                    #import the numpy library as np
import matplotlib.pyplot as plt          #import the plotting library

#####################################################################

# PRELIMINARY INFORMATION

# How to use this python script: In the next code-block, enter the
# desired x and y coordinates of three known data points, as arrays.

# Result: The script will come up with a quadratic function that
# passes through each of the three data points, and two different
# methods will be used to come up with this approximation.

# The first method constructs the quadratic function directly by
# using the Lagrange interpolating polynomial.

# The second method assumes that the interpolant is of the form
# lam[0] + (lam[1])x + (lam[2])x**2, and solves for the three unknown
# quantities lam[0], lam[1], and lam[2].

# After the quadratic interpolant has been calculated using the two
# methods, the results are plotted for comparison in a single viewing
# window.

#####################################################################

# USER INPUT

# The two input vectors, x and y.  These define the known data that
# must be matched by the quadratic approximation.
# Hint: x = np.array([])
#       y = np.array([])

x = np.array([-1., 0., 1.])                            #x-coordinates
y = np.array([1., 3., 0.])                             #y-coordinates

#####################################################################

# Method 1 (build the quadratic function directly)

def quadratic(X):
    return (X-x[1]) * (X-x[2]) / (x[0]-x[1]) / (x[0]-x[2]) * y[0] + \
           (X-x[0]) * (X-x[2]) / (x[1]-x[0]) / (x[1]-x[2]) * y[1] + \
           (X-x[0]) * (X-x[1]) / (x[2]-x[0]) / (x[2]-x[1]) * y[2]

#####################################################################

# A dense set of x-coordinates where we  will plot both of the
# approximations to see what they look like.  This row-vector should
# begin at the first value of the vector x, and should end at the
# last value of the vector x, but should also include at least 100
# equally spaced values in between.  Hint: X = np.linspace()

X = np.linspace(x[0], x[-1], 200)

#####################################################################

# Method 2 (indirect but more flexible and generalizable)

# Make a row-vector of three ones.  Hint: e = np.ones()
e = np.ones(np.shape(x))

# Make the Vandermonde matrix.  Hint: np.vstack().T
A = np.vstack((e, x, x**2)).T

# The column vector "lam" tells you how much of each basis function
# that you need in order to match the known data.  This line solves
# the linear system A*lam=y for lam.  Hint: You will need to solve
# a linear system using lam = np.linalg.solve()
lam = np.linalg.solve(A, y)

# Evaluate the approximation at each coordinate in X
Q = lam[0] + lam[1]*X + lam[2]*X**2

#####################################################################

# Plot the two quadratic functions at all of the x-coordinates
# in the vector X, and make sure that the two methods give the same
# result.  In order to not mix up the two graphs, plot the first one
# using a solid line, and plot the second one using a dashed line.
# Also, plot the original data on the same graph

plt.figure()
plt.plot(X, quadratic(X), '-')
plt.plot(X, Q, '--')
plt.plot(x, y, '*')
plt.show()

#####################################################################