import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gab import nonhydro, rk

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "bubble"

#"exner" (need to fix so that "hydrostaticPressure" also works):
formulation = "hydrostaticPressure"

semiLagrangian = 0                  #Set this to zero.  SL not working yet.
dx = 100.
ds = 100.
FD = 4                                    #Order of lateral FD (2, 4, or 6)
rbfOrder = 3
polyOrder = 1
stencilSize = 9
saveDel = 100
var = 2
plotFromSaved = 0
rkStages = 3
plotNodes = 0

###########################################################################

t = 0.

saveString = './results/' + testCase + '/' \
+ 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
+ 'ds' + '{0:1d}'.format(np.int(np.round(ds)+1e-12)) + '/'

###########################################################################

Cp, Cv, Rd, g, Po = nonhydro.getConstants()

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= nonhydro.getSpaceDomain( testCase, dx, ds, FD )

tf, dt, dtEul, nTimesteps = nonhydro.getTimeDomain( testCase, dx, ds )

s, dsdx, dsdz = nonhydro.getHeightCoordinate( zTop, zSurf, zSurfPrime )

FDo2 = np.int( FD/2 )
ii = np.arange( 1, nLev+1 )
jj = np.arange( FDo2, nCol+FDo2 )
i0 = ii[0]
i1 = ii[-1] + 1
j0 = jj[0]
j1 = jj[-1] + 1

Tx, Tz, Nx, Nz = nonhydro.getTanNorm( zSurfPrime, x[0,jj] )

U, thetaBar, piBar, dpidsBar \
= nonhydro.getInitialConditions( testCase, formulation \
, nLev, nCol, FD, x, z \
, Cp, Cv, Rd, g, Po \
, dsdz )

ind = nonhydro.getIndexes( x, z, xLeft, xRight, zSurf, zTop, FD, nLev, nCol )

if plotNodes == 1 :
    nonhydro.plotNodes( x, z, ind, testCase )
    sys.exit( "\nDone plotting nodes.\n" )
    
###########################################################################

dsdxBottom = dsdx( x[0,jj], zSurf(x[0,jj]) )
dsdzBottom = dsdz( x[0,jj], zSurf(x[0,jj]) )
dsdxAll = dsdx( x, z )
dsdzAll = dsdz( x, z )
dsdxVec = dsdxAll . flatten()
dsdzVec = dsdzAll . flatten()
dsdxEul = dsdx( x[ii,:][:,jj], z[ii,:][:,jj] )
dsdzEul = dsdz( x[ii,:][:,jj], z[ii,:][:,jj] )

###########################################################################

#Define finite difference (FD) weights for derivative approximation:

if FD == 2 :
    wx = np.array( [ -1./2., 0., 1./2. ] )
    wxhv = np.array( [ 1., -2., 1. ] )
    gamma = 1./2.
elif FD == 4 :
    wx = np.array( [ 1./12., -2./3., 0., 2./3., -1./12. ] )
    wxhv = np.array( [ 1., -4., 6., -4., 1. ] )
    gamma = -1./12.
elif FD == 6 :
    wx = np.array( [ -1./60., 3./20., -3./4., 0., 3./4, -3./20., 1./60. ] )
    wxhv = np.array( [ 1., -6., 15., -20., 15., -6., 1. ] )
    gamma = 1./60.
else :
    sys.exit( "\nError: FD should be 2, 4, or 6.\n" )

ws = np.array( [ -1./2., 0., 1./2. ] )
wshv = np.array( [ 1., -2., 1. ] )

###########################################################################

bigTx = np.tile( Tx, (2,1) )
bigTz = np.tile( Tz, (2,1) )

normGradS = np.sqrt( dsdxBottom**2. + dsdzBottom**2. )

###########################################################################

#Define functions for approximating derivatives:

def Dx( U ) :
    return nonhydro.LxFD( U, wx, jj, dx, FD, FDo2 )

def Ds( U ) :
    return nonhydro.LsFD( U, ws, ii, ds )

def HVx( U ) :
    return nonhydro.LxFD( U, wxhv, jj, dx, FD, FDo2 )

def HVs( U ) :
    return nonhydro.LsFD( U, wshv, ii, ds )

###########################################################################

#Important functions for time stepping, which may be chosen by user:

if formulation == "exner" :
    
    def setGhostNodes( U ) :
        U = nonhydro.setGhostNodes1( U \
        , Tx, Tz, Nx, Nz, bigTx, bigTz \
        , nLev, nCol, thetaBar, g, Cp \
        , normGradS, ds, dsdxBottom, dsdzBottom \
        , wx, jj, dx, FD, FDo2 )
        P = []
        return U, P
    
    def odefun( t, U ) :
        return nonhydro.odefun1( t, U \
        , setGhostNodes, Dx, Ds, HVx, HVs \
        , ii, jj, i0, i1, j0, j1 \
        , dsdxEul, dsdzEul \
        , Cp, Cv, Rd, g, gamma )
    
elif formulation == "hydrostaticPressure" :
    
    def setGhostNodes( U ) :
        U, P = nonhydro.setGhostNodes2( U \
        , Tx, Tz, Nx, Nz, bigTx, bigTz \
        , nLev, nCol, thetaBar, dpidsBar, g, Cp, Po, Rd, Cv \
        , normGradS, ds, dsdxBottom, dsdzBottom, dsdz(x,z) \
        , wx, jj, dx, FD, FDo2 )
        return U, P
    
    def odefun( t, U ) :
        return nonhydro.odefun2( t, U \
        , setGhostNodes, Dx, Ds, HVx, HVs \
        , ii, jj, i0, i1, j0, j1 \
        , dsdxEul, dsdzEul, dsdxAll, dsdzAll \
        , Cp, Cv, Rd, g, gamma )
    
else :
    
    sys.exit( "\nError: formulation should be 'exner' or 'hydrostaticPressure'.\n" )

###########################################################################

if semiLagrangian == 1 :
    
    def semiLagrangianTimestep( Un1, U, alp, bet ) :
        U1, alp, bet = nonhydro.conventionalSemiLagrangianTimestep( Un1, U, alp, bet \
        , setGhostNodes, Dx, Ds \
        , nLev, nCol, FD, FDo2, ds \
        , Cp, Rd, Cv, g, dt \
        , x.flatten(), z.flatten(), dsdx, dsdz \
        , ind.m, i0, i1, j0, j1 \
        , rbfOrder, polyOrder, stencilSize )
        return U1, alp, bet
        # return nonhydro.mySemiLagrangianTimestep( Un1, U \
        # , setGhostNodes, Dx, Ds \
        # , x.flatten(), z.flatten(), ind.m, dt \
        # , nLev, nCol, FD, i0, i1, j0, j1 \
        # , Cp, Rd, Cv, g, dsdxVec[ind.m], dsdzVec[ind.m] \
        # , rbfOrder, polyOrder, stencilSize )

###########################################################################

#functions that will not be changed by user:

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit( "\nError: rkStages should be 3 or 4.  rk2 is not stable for this problem.\n" )

def printInfo( U, et, t ) :
    return nonhydro.printInfo( U, et , t \
    , formulation \
    , thetaBar, piBar, dpidsBar )

#Figure size and contour levels for plotting:
fig, CL = nonhydro.setFigAndContourLevels( testCase )

def saveContourPlot( U, t ) :
    nonhydro.saveContourPlot( U, t \
    , formulation, testCase, var, fig \
    , x, z, thetaBar, piBar, dpidsBar, CL, FDo2 \
    , xLeft, xRight, zTop, dx, ds )

###########################################################################

#Eulerian time-stepping for first large time step:

print()
print("dt =",dt)
print("dtEul =",dtEul)
print()

#Save initial conditions and contour of first frame:
U, P = setGhostNodes( U )
np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
Un1 = U
et = printInfo( U, time.clock(), t )
saveContourPlot( U, t )

#The actual Eulerian time-stepping from t=0 to t=dt:
for i in range( np.int( np.round(dt/dtEul) + 1e-12 ) ) :
    U = rk( t, U, odefun, dtEul )
    t = t + dtEul

U, P = setGhostNodes( U )
alp = 0.
bet = 0.

###########################################################################

#The rest of the time-stepping:

for i in range(1,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        if plotFromSaved == 0 :
            U, P = setGhostNodes( U )
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        elif plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        
        et = printInfo( U, et, t )
        saveContourPlot( U, t )
        
    if plotFromSaved == 0 :
        if semiLagrangian == 0 :
            U = rk( t, U, odefun, dt )
        elif semiLagrangian == 1 :
            U1, alp, bet = semiLagrangianTimestep( Un1, U, alp, bet )
            Un1 = U
            U = U1
        else :
            sys.exit( "\nError: semiLagrangian should be 0 or 1.\n" )
    
    t = t + dt

############################################################################