import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

sys.path.append( '../../../site-packages' )
from gab import nonhydro, rk

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "bubble"

#"exner" or "hydrostaticPressure":
formulation  = "exner"

semiImplicit = 1
gmresTol     = 1e-5

dx    = 200.
ds    = 200.
dtExp = 1./5.
dtImp = 2.

FD = 4                                    #Order of lateral FD (2, 4, or 6)
gx = 1                                     #avg lateral velocity (estimate)
gs = 1                                    #avg vertical velocity (estimate)

rkStages  = 3
plotNodes = 0                               #if 1, plot nodes and then exit
saveDel   = 50                            #print/save every saveDel seconds

var           = 2                        #determines what to plot (0,1,2,3)
saveArrays    = 1 
saveContours  = 1
plotFromSaved = 0                   #if 1, results are loaded, not computed

###########################################################################

t = 0.

if semiImplicit == 1 :
    saveString = './semiImplicitResults/'
elif semiImplicit == 0 :
    dtImp = dtExp
    saveString = './explicitResults/'
else :
    sys.exit( "Error: semiImplicit should be 0 or 1." )

saveString = saveString + testCase + '/' \
+ 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
+ 'ds' + '{0:1d}'.format(np.int(np.round(ds)+1e-12)) + '/'

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )

if not os.path.exists( saveString ) :
    os.makedirs( saveString )

###########################################################################

Cp, Cv, Rd, g, Po = nonhydro.getConstants()

tf = nonhydro.getTfinal( testCase )
nTimesteps = np.int( np.round(tf/dtImp) + 1e-12 )

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= nonhydro.getSpaceDomain( testCase, dx, ds, FD )

s, dsdx, dsdz = nonhydro.getHeightCoordinate( zTop, zSurf, zSurfPrime )

FDo2 = np.int( FD/2 + 1e-12 )
ii = np.arange( 1, nLev+1 )
jj = np.arange( FDo2, nCol+FDo2 )
i0 = ii[0]
i1 = ii[-1] + 1
j0 = jj[0]
j1 = jj[-1] + 1

Tx, Tz, Nx, Nz = nonhydro.getTanNorm( zSurfPrime, x[0,jj] )

U0, thetaBar, piBar, dthetabarDz, dpidsBar \
= nonhydro.getInitialConditions( testCase, formulation \
, nLev, nCol, FD, x, z \
, Cp, Cv, Rd, g, Po \
, dsdz )

thetaBarBot = ( thetaBar[0,   jj] + thetaBar[1,     jj] ) / 2.
thetaBarTop = ( thetaBar[nLev,jj] + thetaBar[nLev+1,jj] ) / 2.

if plotNodes == 1 :
    
    ind = nonhydro.getIndexes( x, z, xLeft, xRight, zSurf, zTop, FD \
    , nLev, nCol )
    
    nonhydro.plotNodes( x, z, ind, testCase )
    
    sys.exit( "\nDone plotting nodes.\n" )
    
###########################################################################

#Derivatives of height coordinate function s, and stuff on interior only:

dsdxBot = dsdx( x[0,jj], zSurf(x[0,jj]) )
dsdzBot = dsdz( x[0,jj], zSurf(x[0,jj]) )
dsdxEul = dsdx( x[ii,:][:,jj], z[ii,:][:,jj] )
dsdzEul = dsdz( x[ii,:][:,jj], z[ii,:][:,jj] )

thetaBarEul    = thetaBar   [ii,:][:,jj]
piBarEul       = piBar      [ii,:][:,jj]
dthetabarDzEul = dthetabarDz[ii,:][:,jj]

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
wx   = wx / dx
wxhv = gamma * gx * wxhv / dx

ws = np.array( [ -1./2., 0., 1./2. ] )
wshv = np.array( [ 1., -2., 1. ] )
ws   = ws / ds
wshv = 1./2. * gs * wshv / ds

###########################################################################

bigTx = np.tile( Tx, (2,1) )
bigTz = np.tile( Tz, (2,1) )

normGradS = np.sqrt( dsdxBot**2. + dsdzBot**2. )

###########################################################################

#Define functions for approximating derivatives:

def Dx( U ) :
    return nonhydro.LxFD_2D( U, wx,   j0, j1, dx, FD, FDo2 )

def Ds( U ) :
    return nonhydro.LsFD_2D( U, ws,   i0, i1, ds )

def HVx( U ) :
    return nonhydro.LxFD_2D( U, wxhv, j0, j1, dx, FD, FDo2 )

def HVs( U ) :
    return nonhydro.LsFD_2D( U, wshv, i0, i1, ds )

###########################################################################

#Important functions for time stepping, which may be chosen by user:

if formulation == "exner" :
    
    def setGhostNodes( U ) :
        U = nonhydro.setGhostNodes1( U \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, jj \
        , nLev, nCol, ds, thetaBarBot, thetaBarTop \
        , g, Cp, normGradS, dsdxBot, dsdzBot \
        , wx, j0, j1, dx, FD, FDo2 )
        P = []
        return U, P
    
    def implicitPart( U ) :
        return nonhydro.implicitPart( U \
        , Dx, Ds, HVx, HVs \
        , nLev, nCol, i0, i1, j0, j1, Cp, Cv, Rd, g \
        , dsdxEul, dsdzEul, thetaBarEul, piBarEul, dthetabarDzEul )
    
    def explicitPart( U ) :
        return nonhydro.explicitPart( U \
        , Dx, Ds \
        , nLev, nCol, i0, i1, j0, j1, Cp, Cv, Rd \
        , dsdxEul, dsdzEul )
    
    def odefun( t, U ) :
        U, P = setGhostNodes( U )
        V = np.zeros(( 4, nLev+2, nCol+FD ))
        V[:,i0:i1,j0:j1] = implicitPart(U) + explicitPart(U)
        return V
    
    if semiImplicit == 1 :
        
        def ell( U ) :
            return nonhydro.L( U \
            , dtImp, setGhostNodes, implicitPart, nLev, nCol, FD \
            , i0, i1, j0, j1 )
        
        L = LinearOperator( ( 4*(nLev+2)*(nCol+FD), 4*(nLev+2)*(nCol+FD) ), matvec=ell, dtype=float )
        
        def leapfrogTimestep( t, U0, U1, dt ) :
            t, U2 = nonhydro.leapfrogTimestep( t, U0, U1, dt \
            , nLev, nCol, FD, i0, i1, j0, j1 \
            , L, implicitPart, explicitPart, gmresTol )
            return t, U2
    
elif formulation == "hydrostaticPressure" :
    
    def setGhostNodes( U ) :
        U, P = nonhydro.setGhostNodes2( U \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, jj \
        , nLev, nCol, thetaBar, dpidsBar, g(), Cp(), Po(), Rd(), Cv() \
        , normGradS, ds, dsdxBottom, dsdzBottom, dsdz(x,z) \
        , wx, j0, j1, dx, FD, FDo2 )
        return U, P
    
    def odefun( t, U ) :
        return nonhydro.odefun2( t, U \
        , setGhostNodes, Dx, Dx2D, Ds, Ds2D, HVx, HVs \
        , ii, jj, i0, i1, j0, j1 \
        , dsdxEul, dsdzEul, dsdxAll, dsdzAll \
        , Cp(), Cv(), Rd(), g(), gamma )
    
else :
    
    sys.exit( "\nError: formulation should be 'exner' or 'hydrostaticPressure'.\n" )

###########################################################################

#Functions that will not be changed by user:

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit( "\nError: rkStages should be 3 or 4.  rk2 is not stable for this problem.\n" )

def printInfo( U, et, t ) :
    return nonhydro.printInfo( U, et , t, formulation )

#Figure size and contour levels for plotting:
if ( saveContours == 1 ) | ( plotFromSaved == 1 ) :
    fig, CL = nonhydro.setFigAndContourLevels( testCase )

def saveContourPlot( U, t ) :
    nonhydro.saveContourPlot( U, t \
    , formulation, testCase, var, fig \
    , x, z, CL, FDo2 \
    , xLeft, xRight, zTop, dx, ds )

###########################################################################

#Eulerian time-stepping for first large time step

#Save initial conditions and contour of first frame:
U0, P = setGhostNodes( U0 )
if saveArrays == 1 :
    np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U0 )
U1 = U0
et = printInfo( U1, time.clock(), t )
if saveContours == 1 :
    saveContourPlot( U1, t )

#The actual Eulerian time-stepping from t=0 to t=dt:
for i in range( np.int( np.round(dtImp/dtExp) + 1e-12 ) ) :
    t, U1 = rk( t, U1, odefun, dtExp )

U1, P = setGhostNodes( U1 )

###########################################################################

#The rest of the time-stepping:

for i in range(1,nTimesteps+1) :
    
    # print()
    # print(i)
    
    if np.mod( i, np.int(np.round(saveDel/dtImp)) ) == 0 :
        
        if plotFromSaved == 0 :
            U1, P = setGhostNodes( U1 )
            if saveArrays == 1 :
                np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U1 )
        elif plotFromSaved == 1 :
            U1 = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        
        et = printInfo( U1, et, t )
        if saveContours == 1 :
            saveContourPlot( U1, t )
        
    if plotFromSaved == 0 :
        if semiImplicit == 0 :
            t, U2 = rk( t, U1, odefun, dtImp )
        elif semiImplicit == 1 :
            t, U2 = leapfrogTimestep( t, U0, U1, dtImp )
            U2, P = setGhostNodes( U2 )
        else :
            sys.exit( "\nError: semiImplicit should be 0 or 1.\n" )
        U0 = U1
        U1 = U2
    else :
        t = t + dtImp

############################################################################
