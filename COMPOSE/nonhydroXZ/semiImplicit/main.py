import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import sys
import time

sys.path.append('../../../site-packages')
from gab import semiImplicit, rk
from gab.nonhydro import setFigAndContourLevels

###########################################################################

testCase = "igw"
dx       = 500.
dz       = 500.
FD       = 4
dt       = 1./2.
saveDel  = 100

saveArrays  = 0
savePlots   = 1
var         = 3
plotNodes   = 0
spyMatrices = 0

###########################################################################

t = 0.

saveString = './results/' + testCase + '/' \
+ 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
+ 'dz' + '{0:1d}'.format(np.int(np.round(dz)+1e-12)) + '/'

###########################################################################

Cp, Cv, Rd, g = semiImplicit.getConstants()

tf = semiImplicit.getTfinal( testCase )
nTimesteps = np.int( np.round(tf/dt) + 1e-12 )

xLeft, xRight, zTop, nLev, nCol, N, xx, zz, x, z \
= semiImplicit.getSpaceDomain( testCase, dx, dz )

if plotNodes == 1 :
    plt.plot( x, z, '.' )
    plt.axis( 'equal' )
    plt.show()

U, thetaBar, piBar, dthetabarDz, dpibarDz \
= semiImplicit.getInitialConditions( testCase, nLev, nCol, x, z \
, Cp, Cv, Rd, g )

Lx, Lz, HVx, HVz, gamma = semiImplicit.getDerivativeOperators( nCol, nLev \
, FD, dx, dz )

Bc1, Bc2 = semiImplicit.getBoundaryConditionMatrices( thetaBar \
, nLev, nCol, g, Cp )

###########################################################################

# fig, CL = setFigAndContourLevels( testCase )
# tmp = Lz.u.dot( (z-3200.)**2. )
# exact = 2.*(z-3200.)
# # tmp = Lx.dot( 3200.*np.sin(np.pi*x/3200) )
# # exact = np.pi * np.cos(np.pi*x/3200.)
# tmp = exact - tmp
# print()
# print([np.min(tmp),np.max(tmp)])
# tmp = np.reshape( tmp, (nCol,nLev) )
# tmp = np.transpose( tmp )
# plt.contourf( xx, zz, tmp )
# # plt.axis( 'equal' )
# plt.colorbar()
# plt.savefig( '{0:04d}'.format(np.int(np.round(t)+1e-12))+'.png', bbox_inches = 'tight' )
# plt.clf()

###########################################################################

A = semiImplicit.getBlockMatrix( Lx, Lz, Bc1, nLev, nCol \
, thetaBar, piBar \
, Cp, Cv, Rd )

if spyMatrices == 1 :
    plt.spy(Lx)
    plt.show()
    sys.exit("\nStop here for now.\n")

###########################################################################

V = np.zeros(( 4*N ))

def odeFunction( t, U ) :
    return semiImplicit.odeFun( t, U \
    , Lx, Lz, HVx, HVz, Bc1, Bc2 \
    , thetaBar, piBar, dthetabarDz, dpibarDz \
    , N, gamma, Rd, Cv, Cp, g, V )

def rungeKutta( t, U ) :
    t, U = rk.rk3( t, U, odeFunction, dt )
    return t, U

fig, CL = setFigAndContourLevels( testCase )

def saveContourPlot( U, t ) :
    semiImplicit.saveContourPlot( U, t \
    , testCase, var, nLev, nCol, N \
    , xx, zz, CL \
    , xLeft, xRight, zTop, dx, dz )

###########################################################################

et = semiImplicit.printInfo( U, time.clock(), t, N )

for i in range(nTimesteps+1) :

    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        if saveArrays == 1 :
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        else :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        et = semiImplicit.printInfo( U, et, t, N )
        if savePlots == 1 :
            saveContourPlot( U, t )
    
    if saveArrays == 1 :
        t, U = rungeKutta( t, U )
    else :
        t = t + dt

###########################################################################