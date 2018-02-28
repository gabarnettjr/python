import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import sys
import time
import os

sys.path.append('../../../site-packages')
from gab import semiImplicit, rk
from gab.nonhydro import setFigAndContourLevels

###########################################################################

#testCase = "igw", "bubble", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent"

#dtExp is the time-step for the explicit time-stepper (RK3)
#dtImp is the time-step for the implicit time-stepper (leapfrog)

#solution will be saved or plotted every saveDel seconds

#implicit and directSolve should be either 0 or 1

#if implicit=1 and directSolve=1, it will take a while because it needs to
#do the sparse LU factorization before time-stepping.

#gx and gz are the hyperviscosity coefficients (lateral and vertical)

#var determines which variable will be plotted:
#var=0: u
#var=1: w
#var=2: potential temperature perturbation
#var=3: exner pressure perturbation

#FD determines the order of the lateral finite differences (set to 4)

###########################################################################

#HOW TO RUN:
#python main.py

testCase = "igw"
dx       = 500.
dz       = 500.
dtExp    = 1./1.
dtImp    = 10.
saveDel  = 100

implicit    = 1 
directSolve = 1

gx = -1./12. * 20.
gz =  1./2.  * .003

saveArrays = 1
savePlots  = 0
var        = 2

plotNodes   = 0
spyMatrices = 0

FD = 4

###########################################################################

t = 0.

if implicit != 1 :
    dtImp = dtExp

if implicit == 1 :
    saveString = './implicit/' + testCase + '/' \
    + 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
    + 'dz' + '{0:1d}'.format(np.int(np.round(dz)+1e-12)) + '/'
else :
    saveString = './explicit/' + testCase + '/' \
    + 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
    + 'dz' + '{0:1d}'.format(np.int(np.round(dz)+1e-12)) + '/'

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )

if not os.path.exists( saveString ) :
    os.makedirs( saveString )

###########################################################################

Cp, Cv, Rd, g = semiImplicit.getConstants()

tf = semiImplicit.getTfinal( testCase )
nTimesteps = np.int( np.round(tf/dtImp) + 1e-12 )

xLeft, xRight, zTop, nLev, nCol, N, xx, zz, x, z \
= semiImplicit.getSpaceDomain( testCase, dx, dz )

if plotNodes == 1 :
    plt.plot( x, z, '.' )
    plt.axis( 'equal' )
    plt.show()
    sys.exit( "\nStop here for now.\n" )

U0, thetaBar, piBar, dthetabarDz, dpibarDz \
= semiImplicit.getInitialConditions( testCase, nLev, nCol, x, z \
, Cp, Cv, Rd, g )

Lx, Lz, HVx, HVz = semiImplicit.getDerivativeOperators( nCol, nLev \
, FD, dx, dz, gx, gz )

Bc1, Bc2 = semiImplicit.getBoundaryConditionMatrices( thetaBar \
, nLev, nCol, g, Cp )

###########################################################################

if ( implicit == 1 ) & ( saveArrays == 1 ) :
    
    A = semiImplicit.getBlockMatrix( Lx, Lz, HVx, HVz, Bc1, Bc2, nLev, nCol \
    , thetaBar, piBar, dthetabarDz, gz \
    , Cp, Cv, Rd, g )
    
    L = sparse.csc_matrix( sparse.eye(4*N) + dtImp*A )
    R = sparse.csc_matrix( sparse.eye(4*N) - dtImp*A )
    
    if directSolve == 1 :
        L = splu(L)

if spyMatrices == 1 :
    plt.spy(Lx)
    plt.show()
    sys.exit("\nStop here for now.\n")

###########################################################################

V   = np.zeros(( 4*N ))

def odeFunction( t, U ) :
    return semiImplicit.odeFun( t, U \
    , Lx, Lz, HVx, HVz, Bc1, Bc2 \
    , thetaBar, piBar, dthetabarDz, dpibarDz \
    , N, gz, Cp, Cv, Rd, g, V )

rungeKutta = rk.rk3

if ( implicit == 1 ) & ( saveArrays == 1 ) :
    
    def leapfrogTimestep( t, U0, U1 ) :
        t, U2 = semiImplicit.leapfrogTimestep( t, U0, U1, dtImp, L, R \
        , Lx, Lz, Bc1, directSolve \
        , thetaBar, dthetabarDz \
        , N, Cp, Cv, Rd, g, V )
        return t, U2

if savePlots == 1 :
    
    fig, CL = setFigAndContourLevels( testCase )

    def saveContourPlot( U, t ) :
        semiImplicit.saveContourPlot( U, t \
        , testCase, var, nLev, nCol, N \
        , xx, zz, CL \
        , xLeft, xRight, zTop, dx, dz )

###########################################################################

#Several RK3 time-steps to make one large time-step:

if saveArrays == 1 :
    np.save( saveString+'{0:04d}'.format(np.int(np.round(t)+1e-12))+'.npy', U0 )
else :
    U0 = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)+1e-12))+'.npy' )

if savePlots == 1 :
    saveContourPlot( U0, t )

U1 = U0

et = semiImplicit.printInfo( U1, time.clock(), t, N )

for i in range( np.int( np.round(dtImp/dtExp) + 1e-12 ) ) :
    t, U1 = rungeKutta( t, U1, odeFunction, dtExp )

###########################################################################

#The remaining time-steps are done with semi-implicit method:

for i in range(1,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dtImp)+1e-12) ) == 0 :
        if saveArrays == 1 :
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)+1e-12))+'.npy', U1 )
        else :
            U1 = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)+1e-12))+'.npy' )
        et = semiImplicit.printInfo( U1, et, t, N )
        if savePlots == 1 :
            saveContourPlot( U1, t )
    
    if saveArrays == 1 :
        if implicit == 1 :
            t, U2 = leapfrogTimestep( t, U0, U1 )
        else :
            t, U2 = rungeKutta( t, U1, odeFunction, dtExp )
        U0 = U1
        U1 = U2
    else :
        t = t + dtImp

###########################################################################
