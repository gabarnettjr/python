#rbffd.py

import numpy as np
from scipy.special import xlogy

def getMN() :

    mn = np.array( 

         [ [0,0]

         , [1,0]
         , [0,1]

         , [2,0]
         , [1,1]
         , [0,2]

         , [3,0]
         , [2,1]
         , [1,2]
         , [0,3]

         , [4,0]
         , [3,1]
         , [2,2]
         , [1,3]
         , [0,4] ] )

    return mn

def phi( rad, x, y, rbfParam ) :
    # if np.mod( rbfParam, 2 ) == 0 :
        # z = np.sqrt( x*x + y*y );
        # z = xlogy( z**rbfParam, z );
    # else :
    z = ( x*x + y*y ) ** (rbfParam/2)
    z = z / rad**rbfParam
    return z
    
def phi_x( rad, x, y, rbfParam ) :
    z = rbfParam*x * ( x*x + y*y ) ** ((rbfParam-2)/2)
    z = z / rad**rbfParam
    return z
    
def phi_y( rad, x, y, rbfParam ) :
    z = rbfParam*y * ( x*x + y*y ) ** ((rbfParam-2)/2)
    z = z / rad**rbfParam
    return z
