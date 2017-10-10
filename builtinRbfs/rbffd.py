import numpy as np
from scipy.special import xlogy

def getMN() :
    mn = np.array( \
         [ [0,0] \
         , [1,0] \
         , [0,1] \
         , [2,0] \
         , [1,1] \
         , [0,2] \
         , [3,0] \
         , [2,1] \
         , [1,2] \
         , [0,3] \
         ] )
    return mn

def phi( rad, x, y, rbfParam ) :
    if np.mod( rbfParam, 2 ) == 0 :
        z = np.sqrt( x*x + y*y );
        z = xlogy( z**rbfParam, z );
    else :
        z = ( x**2 + y**2 ) ** (rbfParam/2)
    z = z / rad**rbfParam
    return z