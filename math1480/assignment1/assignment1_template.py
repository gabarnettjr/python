                                      #import the numpy library as np
                                  #import the plotting library as plt

#####################################################################
"""
ASSIGNMENT 1
MATH 1480
Spring 2020
Due Friday, February 21st at 3:00 PM

What to do:  Fill in this template with python code.  At the end of
your script, your code should produce the correct plot, showing the
graph of a quadratic function that passes through three points
specified by the user.

---------------------------------------------------------------------

PRELIMINARY INFORMATION

How to use this python script: In the code-block labeled
USER INPUT, enter the desired x and y coordinates of three known
points, as arrays.

Result: The script will come up with a quadratic function that
passes through all three points, and two different methods will be
used to come up with this quadratic function.

In Method 1 the quadratic function is constructed directly by
using the Lagrange interpolating polynomial.
More information:  https://www.overleaf.com/read/rccncxhyfvky

In Method 2 we assume that the interpolant is of the form
Q(X) = lam[0] + lam[1]*X + lam[2]*X**2, and then solve for the three
unknown quantities lam[0], lam[1], and lam[2].
More information:  https://www.overleaf.com/read/qhtygvjbxccd

After the quadratic interpolant has been calculated using the two
methods, the results are plotted for comparison in a single viewing
window, with the known data points plotted as well.
"""
#####################################################################

# USER INPUT

# The two input vectors, x and y.  These define the known data that
# must be matched by the quadratic function.
# Hint: x = np.array([])
#       y = np.array([])

                                                       #x-coordinates
                                                       #y-coordinates

#####################################################################

# Method 1 (build the quadratic function directly)
# Hint: Refer to the description of Method 1 in the block comments
# and overleaf link at the beginning of the assignment.

def quadratic(X):
    q = 
    return q

#####################################################################

# A dense set of x-coordinates where we  will plot both of the
# approximations to see what they look like.  This row-vector should
# begin at the first value of the vector x, and should end at the
# last value of the vector x, but should also include at least 25
# equally spaced values in between.
# Hint: X = np.linspace()



#####################################################################

# Method 2 (indirect but more flexible and generalizable)
# Hint: Refer to the description of Method 2 in the block comments
# and overleaf link at the beginning of the assignment.

# Make a row-vector of ones that is the same length as x.
# Hint: e = np.ones(np.shape())


# Make the 3-column Vandermonde matrix (columns are 1, x, and x**2).
# Hint: A = np.vstack(()).T
# The ".T" at the end takes the transpose of a matrix.


# The column vector "lam" (short for lambda) tells you how much of
# each basis function (1, x, x**2) that you need in order to match
# the known data.  Solve the linear system A*lam=y for lam.
# Hint: lam = np.linalg.solve()


# Evaluate the approximation at each coordinate in X.


#####################################################################

# Plot the two quadratic functions at all of the x-coordinates
# in the vector X, and make sure that the two methods give the same
# result.  In order to not mix up the two graphs, plot the first one
# using a solid line, and plot the second one using a dashed line.
# Also, plot the original data on the same graph.
# Hint: plt.figure() initializes a figure
#       plt.plot(a, b, '-') plots points with x-coordinate in vector
#           a and y-coordinates in vector b, connecting the points
#           by straight line segments.
#       plt.show() renders the figure at the end so can see it.



#####################################################################