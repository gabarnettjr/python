import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import sys
import time

sys.path.append('../../../site-packages')
from gab import semiImplicit, rk

###########################################################################

testCase = "bubble"
dx = 400.
dz = 400.
FD = 4
dt = 1./8.

plotNodes   = 0
spyMatrices = 0

saveDel = 50

###########################################################################

Cp, Cv, Rd, g = semiImplicit.getConstants()

t = 0.
tf = semiImplicit.getTfinal( testCase )
nTimesteps = np.int( np.round(tf/dt) + 1e-12 )

xLeft, xRight, zTop, nLev, nCol, N, xx, zz, x, z \
= semiImplicit.getSpaceDomain( testCase, dx, dz )

if plotNodes == 1 :
    plt.plot(x,z,'.')
    plt.axis('equal')
    plt.show()

Lx, Lz, HVx, HVz, gamma = semiImplicit.getDerivativeOperators( nCol, nLev \
, FD, dx, dz )

U, thetaBar, piBar, dthetabarDz, dpibarDz \
= semiImplicit.getInitialConditions( testCase, nLev, nCol, x, z \
, Cp, Cv, Rd, g )

###########################################################################

Bc1, Bc2 = semiImplicit.getBoundaryConditionMatrices( thetaBar \
, nLev, nCol, g, Cp )

A = semiImplicit.getBlockMatrix( Lx, Lz, Bc1, nLev, nCol \
, thetaBar, piBar \
, Cp, Cv, Rd )

if spyMatrices == 1 :
    plt.spy(A)
    plt.show()

###########################################################################

def odeFun( t, U ) :
    return semiImplicit.odeFun( t, U \
    , Lx, Lz, HVx, HVz, Bc1, Bc2 \
    , thetaBar, piBar, dthetabarDz, dpibarDz \
    , N, gamma, Rd, Cv, Cp, g ) 

def rungeKutta( t, U ) :
    return rk.rk3( t, U, odeFun, dt )

###########################################################################

et = semiImplicit.printInfo( U, time.clock(), t, N )

for i in range(nTimesteps+1) :
    
    U = rungeKutta( t, U )
    t = t + dt
    
    semiImplicit.printInfo( U, et , t, N )
    time.sleep(1)

###########################################################################
