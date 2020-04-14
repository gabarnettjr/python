## import the numpy library as np
import numpy as np

## import the plotting library as plt
import matplotlib.pyplot as plt

## import the factorial function from the math library
from math import factorial

###########################################################################

def getWeights( z, x, m ) :
    
    n = len(x)
    
    ## Shift the nodes so that the evaluation point is at zero
    x = x - z
    
    ## Construct the Vandermonde matrix A
    A = np.zeros(( n, n ))
    for i in range(n):
        A[i,:] = x**i
    
    ## Create the RHS vector b which has the derivative of each polynomial
    ## function evaluated at zero
    b = np.zeros(( n, 1 ))
    b[m] = factorial( m )
    
    ## Solve for the finite difference weights w
    w = np.linalg.solve( A, b )
    
    return w

###########################################################################