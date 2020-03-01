# import the numpy library as np:


# import the plotting library as plt:


# import the Runge-Kutta library (written by you):


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
t0 = 

# Final time:
tf = 

# Total number of sub-intervals (time-steps):
N = 

# ODE function (RHS):
def f(t,y):
    

# Initial condition:
y0 = 

# True or False boolean variable:
exactSolutionKnown = 

# Exact solution:
if exactSolutionKnown:
    def Y(t):
        

# Number of Runge-Kutta stages to use (1, 2, 3, or 4):
rkStages = 

#####################################################################

# Initialize arrays t and y, and set first element of y equal to
# the initial value.  Keep in mind that if there are N time-steps,
# then the arrays t and y should have N+1 elements.



#####################################################################

# Define the time-step h by taking the width of the full time
# interval and dividing by the total number of time-steps N:



#####################################################################

# Define the Runge-Kutta method that will be used based on the user
# input of the variable rkStages.  This is where your "rungeKutta"
# library will be used, and you will need if-elif statements.  If the
# user entered something other than 1, 2, 3, or 4 for the variable
# rkStages, then you should catch it here, print an error statement,
# and exit from the program.



#####################################################################

# Main time-stepping loop (for-loop)
# This is where the elements of the array y will be "filled in" using
# the Runge-Kutta method:

for n in range(N):
    

#####################################################################

# Plot the numerical solution and the exact solution (if available)
# on the same set of axes.  If the exact solution is available, then
# plot the difference between numerical and exact in a subplot.

plt.figure()

if exactSolutionKnown:
    
    # Plot numerical and exact on same axes in 1st subfigure:
    plt.subplot(1,2,1)
    
    
    # Plot difference of numerical and exact in 2nd subfigure:
    plt.subplot(1,2,2)
    
    
else:

    # Plot the numerical solution (no exact solution available):
    

plt.show()

#####################################################################