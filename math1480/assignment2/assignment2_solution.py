import numpy as np                    #import the numpy library as np
import matplotlib.pyplot as plt   #import the plotting library as plt

#####################################################################
"""
ASSIGNMENT 2
MATH 1480
Spring 2020
Due Friday, February 28th at 4:00 PM

What to do:  Fill in this template with python code.  At the end of
your script, your code should produce the correct plot, showing the
graph of the least squares polynomial approximation for the data
supplied by the user.

---------------------------------------------------------------------

PRELIMINARY INFORMATION

How to use this python script: In the code-block labeled
USER INPUT, enter the desired x and y coordinates of n known
points, as arrays.

Result: The script will come up with the least squares polynomial
function of degree d for the data, where d is also specified by the
user.

In Method 1 the quadratic function is constructed directly by
using the Lagrange interpolating polynomial.
More information:  https://www.overleaf.com/read/rccncxhyfvky

In Method 2 we assume that the interpolant is of the form
Q(X) = lam[0] + lam[1]*X + lam[2]*X**2, and then solve for the three
unknown quantities lam[0], lam[1], and lam[2].
More information:  https://www.overleaf.com/read/qhtygvjbxccd

After the Polynomial function has been calculated, plot it, and also
plot the original data as discrete points, to see how close the
polynomial function is to each point.
"""
#####################################################################

# USER INPUT

# The two input vectors, x and y.  These define the known data that
# will be approximated by the least squares polynomial function.
x = np.array([-2., -1., 0., 1., 2.])                   #x-coordinates
y = np.array([1., 3., 0.])                             #y-coordinates

# The desired degree d of the polynomial function.
d = 4

#####################################################################

# Method 1 (build the quadratic function directly)
# Hint: Refer to the description of Method 1 in the block comments
# and overleaf link at the beginning of the assignment.

def quadratic(X):
    q = (X-x[1]) * (X-x[2]) / (x[0]-x[1]) / (x[0]-x[2]) * y[0] + \
        (X-x[0]) * (X-x[2]) / (x[1]-x[0]) / (x[1]-x[2]) * y[1] + \
        (X-x[0]) * (X-x[1]) / (x[2]-x[0]) / (x[2]-x[1]) * y[2]
    return q

#####################################################################

# A dense set of x-coordinates where we  will plot both of the
# approximations to see what they look like.  This row-vector should
# begin at the first value of the vector x, and should end at the
# last value of the vector x, but should also include at least 25
# equally spaced values in between.
# Hint: X = np.linspace()

X = np.linspace(x[0], x[-1], 500)

#####################################################################

# Method 2 (indirect but more flexible and generalizable)
# Hint: Refer to the description of Method 2 in the block comments
# and overleaf link at the beginning of the assignment.

# Make a row-vector of ones that is the same length as x.
# Hint: e = np.ones(np.shape())
e = np.ones(np.shape(x))

# Make the 3-column Vandermonde matrix (columns are 1, x, and x**2).
# Hint: A = np.vstack(()).T
# The ".T" at the end takes the transpose of a matrix.
A = np.vstack((e, x, x**2)).T

# The column vector "lam" (short for lambda) tells you how much of
# each basis function (1, x, x**2) that you need in order to match
# the known data.  Solve the linear system A*lam=y for lam.
# Hint: lam = np.linalg.solve()
lam = np.linalg.solve(A, y)

# Evaluate the approximation at each coordinate in X.
Q = lam[0] + lam[1]*X + lam[2]*X**2
# B = np.vstack((np.ones(np.shape(X)), X, X**2)).T
# Q = B.dot(lam)

#####################################################################

# Plot the two quadratic functions at all of the x-coordinates
# in the vector X, and make sure that the two methods give the same
# result.  In order to not mix up the two graphs, plot the first one
# using a solid line, and plot the second one using a dashed line.
# Also, plot the original data on the same graph.
# Hint: plt.figure() initializes a figure
#       plt.plot(a, b, '-') plots points with x-coordinates in vector
#           a and y-coordinates in vector b, connecting the points
#           by straight line segments.
#       plt.show() renders the figure at the end so can see it.

plt.figure()
plt.plot(X, quadratic(X), "-", linewidth=3, color="red")
plt.plot(X, Q, "--", linewidth=3, color="green")
plt.plot(x, y, ".", markersize=10, color="black")
plt.legend(["Method 1", "Method 2", "Original Points"])
plt.show()

#####################################################################
