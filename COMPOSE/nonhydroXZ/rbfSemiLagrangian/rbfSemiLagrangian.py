import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gab import nonhydro, rk

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "bubble"

#"exner" or "hydrostaticPressure"
formulation = "exner"

semiLagrangian = 0
dx = 200.
ds = 200.
FD = 4                                    #Order of lateral FD (2, 4, or 6)
rbforder = 3
polyorder = 1
stencilSize = 9
saveDel = 100
var = 2
plotFromSaved = 0
rkStages = 3
plotNodes = 0

###########################################################################

t = 0.

saveString = './results/' + testCase + '/dx' \
+ '{0:1d}'.format(np.int(dx)) + 'ds' + '{0:1d}'.format(np.int(ds)) + '/'

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

U, thetaBar, piBar = nonhydro.getInitialConditions( testCase, formulation \
, nLev, nCol, FD, x, z \
, Cp, Cv, Rd, g, Po \
, dsdz )

###########################################################################

dsdxBottom = dsdx( x[0,jj], zSurf(x[0,jj]) )
dsdzBottom = dsdz( x[0,jj], zSurf(x[0,jj]) )
dsdx = dsdx( x[ii,:][:,jj], z[ii,:][:,jj] )
dsdz = dsdz( x[ii,:][:,jj], z[ii,:][:,jj] )

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

#Important functions for time stepping:

def setGhostNodes( U ) :
    return nonhydro.setGhostNodesFD( U \
    , Tx, Tz, Nx, Nz, bigTx, bigTz \
    , nLev, nCol, thetaBar, g, Cp \
    , normGradS, ds, dsdxBottom, dsdzBottom \
    , wx, jj, dx, FD, FDo2 )

def odefun( t, U ) :
    return nonhydro.odefunFD( t, U \
    , setGhostNodes \
    , dx, ds, wx, ws, wxhv, wshv \
    , ii, jj, i0, i1, j0, j1 \
    , dsdx, dsdz, FD, FDo2 \
    , Cp, Cv, Rd, g, gamma )

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit( "\nError: rkStages should be 3 or 4.\n" )

#Figure size and contour levels for plotting:
fig, CL = nonhydro.setFigAndContourLevels( testCase )

def saveContourPlot( U ) :
    return nonhydro.saveContourPlot( U \
    , testCase, var, fig \
    , x, z, thetaBar, piBar, CL, FDo2 \
    , xLeft, xRight, zTop, dx, ds, t )

###########################################################################

#Eulerian time-stepping for first large time step:

#Save initial conditions and contour of first frame:
U = setGhostNodes(U)
saveContourPlot( U )

#The actual Eulerian time-stepping
for i in range( np.int(dt/dtEul) ) :
    U = rk( t, U, odefun, dtEul )
    t = t + dtEul

###########################################################################

#Semi-Lagrangian time-stepping (still Eulerian for now):

for i in range(1,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        if plotFromSaved == 0 :
            U = setGhostNodes(U)
        elif plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        
        saveContourPlot( U )
        
    if plotFromSaved == 0 :
        U = rk( t, U, odefun, dt )
    
    t = t + dt

###########################################################################

if plotNodes == 1 :

    #Plot some nodes:

    x = x.flatten()
    z = z.flatten()

    ms = 12

    plt.plot( x, z, ".", color="black" )
    plt.plot( x[ind.m],  z[ind.m],  "o", color="black",  fillstyle="none", markersize=10 )
    plt.plot( x[ind.gl], z[ind.gl], "o", color="red",    fillstyle="none", markersize=ms )
    plt.plot( x[ind.r],  z[ind.r],  "s", color="red",    fillstyle="none", markersize=ms )
    plt.plot( x[ind.gr], z[ind.gr], "o", color="blue",   fillstyle="none", markersize=ms )
    plt.plot( x[ind.l],  z[ind.l],  "s", color="blue",   fillstyle="none", markersize=ms )
    plt.plot( x[ind.gb], z[ind.gb], "^", color="orange", fillstyle="none", markersize=ms )
    plt.plot( x[ind.b],  z[ind.b],  "v", color="orange", fillstyle="none", markersize=ms )
    plt.plot( x[ind.gt], z[ind.gt], "v", color="green",  fillstyle="none", markersize=ms )
    plt.plot( x[ind.t],  z[ind.t],  "^", color="green",  fillstyle="none", markersize=ms )
    if testCase != "igw" :
        plt.axis( 'equal' )
    plt.show()

############################################################################