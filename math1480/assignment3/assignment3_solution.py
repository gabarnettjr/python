# import the numpy library as np:
import numpy as np

# import the plotting library as plt:
import matplotlib.pyplot as plt

# import the Runge-Kutta library (written by you):
import rungeKutta

#####################################################################
"""
ASSIGNMENT 3
MATH 1480
Spring 2020
Due Friday, March 13th at 3:00 PM

What to do:  Fill in this template with python code, and write a
separate library called "rungeKutta.py" to be imported.  At the end
of your script, your code should produce the correct plot, showing
the graph of the numerical solution of the initial value problem, as
well as the graph of the exact solution (if known).  Also, if the
exact solution is available, a plot of the difference between the
numerical and the exact solution will be displayed in a subfigure.

---------------------------------------------------------------------

PRELIMINARY INFORMATION

How to use this python script: In the code-block labeled
USER INPUT, enter these eight things:
initial time                  t0
final time                    tf
total number of time steps    N
ODE function                  f(t,y)
initial condition             y0
True or False boolean         exactSolutionKnown
exact solution                Y(t)
number of Runge-Kutta stages  rkStages [1, 2, 3, or 4]

Result: The script will apply the Runge-Kutta method to come up with
an approximate solution at each of the equally-spaced times in the
time interval.  After computing the approximate solution, at the end
of the script, a plot will be displayed with t on the horizontal
axis and y on the vertical axis.

Please have a look at this Overleaf document for an overview of
initial value problems, Euler's Method, and Runge-Kutta methods:

https://www.overleaf.com/read/pfqfsdjzqjty
"""
#####################################################################

# USER INPUT

# Initial time:
t0 = 0

# Final time:
tf = 5

# Total number of sub-intervals (time-steps):
N = 10

# ODE function (RHS):
def f(t,y):
    return -y

# Initial condition:
y0 = 1

# True or False boolean variable:
exactSolutionKnown = True

# Exact solution:
if exactSolutionKnown:
    def Y(t):
        return np.exp(-t)

# Number of Runge-Kutta stages to use (1, 2, 3, or 4):
rkStages = 3

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

# Define the Runge-Kutta method that will be used based on the user
# input of the variable rkStages.  This is where your "rungeKutta"
# library will be used:

if rkStages == 1:
    rk = rungeKutta.rk1
elif rkStages == 2:
    rk = rungeKutta.rk2
elif rkStages == 3:
    rk = rungeKutta.rk3
elif rkStages == 4:
    rk = rungeKutta.rk4
else:
    print()
    print("ERROR: Please choose 1, 2, 3, or 4 for the")
    print("       number of Runge-Kutta stages (rkStages).")
    exit()

#####################################################################

# Main time-stepping loop (for-loop)
# This is where the elements of the array y will be "filled in" using
# the Runge-Kutta method:

for n in range(N):
    y[n+1] = rk(t[n], y[n], f, h)

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
    plt.plot(t, y)
    plt.title('Numerical Solution')
    plt.xlabel('t')
    plt.ylabel('y')

plt.show()

#####################################################################