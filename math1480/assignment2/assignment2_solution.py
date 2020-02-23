# import the numpy library as np:
import numpy as np

# import the plotting library as plt:
import matplotlib.pyplot as plt

#####################################################################
"""
ASSIGNMENT 2
MATH 1480
Spring 2020
Due Friday, March 6th at 3:00 PM

What to do:  Fill in this template with python code.  At the end of
your script, your code should produce the correct plot, showing the
graph of the numerical solution to the initial value problem, as well
as the graph of the exact solution (if known).  Also, if the exact
solution is available, a plot of the difference between the numerical
and the exact solution will be displayed in a subfigure.

---------------------------------------------------------------------

PRELIMINARY INFORMATION

How to use this python script: In the code-block labeled
USER INPUT, enter these six things:
initial time           t0
final time             tf
number of time steps   N
ODE function           f(t,y)
initial condition      y0
True or False boolean  exactSolutionKnown
exact solution         Y(t)

Result: The script will apply Euler's Method to come up with an
approximate solution at each of the equally-spaced times in the
time interval.  After computing the approximate solution, at the end
of the script, a plot will be displayed with t on the horizontal
axis and y on the vertical axis.

Please have a look at this Overleaf document for an overview of
initial value problems and Euler's Method:

https://www.overleaf.com/read/pfqfsdjzqjty
"""
#####################################################################

# USER INPUT

# initial time:
t0 = 0

# final time:
tf = 2

# total number of sub-intervals (time-steps):
N = 20

# ODE function (RHS):
def f(t,y):
    return -y

# initial condition:
y0 = 1

# True or False boolean variable:
exactSolutionKnown = True

# exact solution:
if exactSolutionKnown:
    def Y(t):
        return np.exp(-t)

#####################################################################

# Initialize arrays t and y, and set first element of y equal to
# the initial value.  Keep in mind that if there are N time-steps,
# then the arrays t and y should have N+1 elements.

t = np.linspace(t0, tf, N+1)
y = np.zeros(np.shape(t))
y[0] = y0

#####################################################################

# Define the time-step h by taking the width of the full time
# interval and dividing by the total number of time-steps N:

h = (tf - t0) / N

#####################################################################

# Main time-stepping loop (for-loop)
# This is where the elements of the array y will be "filled in" using
# Euler's Method.

for n in range(N):
    y[n+1] = y[n] + h * f(t[n], y[n])

#####################################################################

# Plot the numerical solution and the exact solution (if available)
# on the same set of axes.  If the exact solution is available, then
# plot the difference between numerical and exact in a subplot.

plt.figure()

if exactSolutionKnown:
    
    # Plot numerical and exact on same axes in 1st subfigure:
    plt.subplot(1,2,1)
    plt.plot(t, y)
    plt.plot(t, Y(t))
    plt.legend(['numerical solution', 'exact solution'])
    plt.title('Numerical and Exact Solution')
    plt.xlabel('t')
    plt.ylabel('y')
    
    # Plot difference of numerical and exact in 2nd subfigure:
    plt.subplot(1,2,2)
    plt.plot(t, y-Y(t))
    plt.title('Numerical minus Exact')
    plt.xlabel('t')
    plt.ylabel('difference')
    
else:

    # Plot the numerical solution (no exact solution available):
    plt.plot(t, y, '-')
    plt.title('Numerical Solution')
    plt.xlabel('t')
    plt.ylabel('y')

plt.show()

#####################################################################