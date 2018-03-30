import sys
import os
import numpy as np
import time
from scipy.sparse.linalg import LinearOperator

sys.path.append( '../../../site-packages' )
from gab import rk
from gab.nonhydro import common

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "igw"
gx       = 20.                             #avg lateral velocity (estimate)
gs       = .003                           #avg vertical velocity (estimate)

#"theta_pi" or "T_rho_P" or "theta_rho_P":
formulation  = "theta_pi"

semiImplicit = 0
gmresTol     = 1e-9                                          #default: 1e-5

dx    = 500.
ds    = 100.
dtExp = 1./2.                                           #explicit time-step
dtImp = 5.                                              #implicit time-step

FD = 6                                    #Order of lateral FD (2, 4, or 6)

rkStages  = 4
plotNodes = 0                               #if 1, plot nodes and then exit
saveDel   = 100                           #print/save every saveDel seconds

var           = 0                        #determines what to plot (0,1,2,3)
saveArrays    = 0
saveContours  = 1
plotFromSaved = 1                   #if 1, results are loaded, not computed

###########################################################################

t = 0.

if semiImplicit == 1 :
    saveString = './semiImplicitResults/'
elif semiImplicit == 0 :
    dtImp = dtExp
    saveString = './explicitResults/'
else :
    sys.exit( "Error: semiImplicit should be 0 or 1." )

saveString = saveString + formulation + '/' + testCase + '/' \
+ 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
+ 'ds' + '{0:1d}'.format(np.int(np.round(ds)+1e-12)) + '/'

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )

if not os.path.exists( saveString ) :
    os.makedirs( saveString )

###########################################################################

Cp, Cv, Rd, g, Po = common.getConstants()

tf = common.getTfinal( testCase )
nTimesteps = np.int( np.round(tf/dtImp) + 1e-12 )

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= common.getSpaceDomain( testCase, dx, ds, FD )

s, dsdx, dsdz = common.getHeightCoordinate( zTop, zSurf, zSurfPrime )

FDo2 = np.int( FD/2 + 1e-12 )

i0 = 1                                            #first interior row index
i1 = nLev+1                                        #last interior row index
j0 = FDo2                                      #first interior column index
j1 = nCol+FDo2                                  #last interior column index

Tx, Tz, Nx, Nz = common.getTanNorm( zSurfPrime, x[0,j0:j1] )

U0, thetaBar, piBar, dpidsBar, Pbar, rhoBar, Tbar \
, dthetaBarDz, dpiBarDz, dTbarDz, dPbarDz, drhoBarDz \
= common.getInitialConditions( testCase, formulation \
, nLev, nCol, FD, x, z \
, Cp, Cv, Rd, g, Po \
, dsdz )

thetaBarBot = ( thetaBar[0,   j0:j1] + thetaBar[1,     j0:j1] ) / 2.
thetaBarTop = ( thetaBar[nLev,j0:j1] + thetaBar[nLev+1,j0:j1] ) / 2.

if plotNodes == 1 :
    ind = common.getIndexes( x, z, xLeft, xRight, zSurf, zTop, FD \
    , nLev, nCol )
    common.plotNodes( x, z, ind, testCase )
    sys.exit( "\nDone plotting nodes." )
    
###########################################################################

#Derivatives of height coordinate function s, and stuff on interior only:

dsdxBot = dsdx( x[0,j0:j1], zSurf(x[0,j0:j1]) )
dsdzBot = dsdz( x[0,j0:j1], zSurf(x[0,j0:j1]) )
dsdxInt = dsdx( x[i0:i1,j0:j1], z[i0:i1,j0:j1] )
dsdzInt = dsdz( x[i0:i1,j0:j1], z[i0:i1,j0:j1] )
dsdzAll = dsdz( x, z )

thetaBarInt    = thetaBar   [i0:i1,j0:j1]
piBarInt       = piBar      [i0:i1,j0:j1]
dthetaBarDzInt = dthetaBarDz[i0:i1,j0:j1]
rhoBarInt      = rhoBar     [i0:i1,j0:j1]
TbarInt        = Tbar       [i0:i1,j0:j1]
drhoBarDzInt   = drhoBarDz  [i0:i1,j0:j1]
dTbarDzInt     = dTbarDz    [i0:i1,j0:j1]

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
    return common.LxFD_2D( U, wx,   j0, j1, FD, FDo2 )

def Ds( U ) :
    return common.LsFD_2D( U, ws,   i0, i1 )

def HVx( U ) :
    return common.LxFD_2D( U, wxhv, j0, j1, FD, FDo2 )

def HVs( U ) :
    return common.LsFD_2D( U, wshv, i0, i1 )

###########################################################################

#Important functions for time stepping:

if formulation == "theta_pi" :
    
    from gab.nonhydro import theta_pi
    
    def setGhostNodes( U ) :
        U = theta_pi.setGhostNodes( U, Dx \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, j0, j1 \
        , nLev, nCol, FD, FDo2, ds, thetaBarBot, thetaBarTop \
        , g, Cp, normGradS, dsdxBot, dsdzBot )
        P = []
        return U, P
    
    def implicitPart( U, P ) :
        return theta_pi.implicitPart( U \
        , Dx, Ds, HVx, HVs \
        , nLev, nCol, i0, i1, j0, j1, Cp, Cv, Rd, g \
        , dsdxInt, dsdzInt, thetaBarInt, piBarInt, dthetaBarDzInt )
    
    def explicitPart( U, P ) :
        return theta_pi.explicitPart( U \
        , Dx, Ds \
        , nLev, nCol, i0, i1, j0, j1, Cp, Cv, Rd \
        , dsdxInt, dsdzInt )
    
elif formulation == "T_rho_P" :
    
    from gab.nonhydro import T_rho_P
    
    def setGhostNodes( U ) :
        U, P = T_rho_P.setGhostNodes( U, Dx \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, j0, j1 \
        , nLev, nCol, FD, FDo2, ds, Pbar, rhoBar, Tbar \
        , g, Rd, normGradS, dsdxBot, dsdzBot )
        return U, P
    
    def implicitPart( U, P ) :
        return T_rho_P.implicitPart( U, P \
        , Dx, Ds, HVx, HVs \
        , nLev, nCol, i0, i1, j0, j1, Cp, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt, TbarInt, drhoBarDzInt, dTbarDzInt )
    
    def explicitPart( U, P ) :
        return T_rho_P.explicitPart( U, P \
        , Dx, Ds \
        , nLev, nCol, i0, i1, j0, j1, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt )
    
elif formulation == "theta_rho_P" :
    
    from gab.nonhydro import theta_rho_P
    
    def setGhostNodes( U ) :
        U, P = theta_rho_P.setGhostNodes( U, Dx \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, j0, j1 \
        , nLev, nCol, FD, FDo2, ds, Pbar, rhoBar, thetaBar \
        , g, Rd, Cp, Cv, Po, normGradS, dsdxBot, dsdzBot )
        return U, P
    
    def implicitPart( U, P ) :
        return theta_rho_P.implicitPart( U, P \
        , Dx, Ds, HVx, HVs \
        , nLev, nCol, i0, i1, j0, j1, Cp, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt, drhoBarDzInt, dthetaBarDzInt )
    
    def explicitPart( U, P ) :
        return theta_rho_P.explicitPart( U, P \
        , Dx, Ds \
        , nLev, nCol, i0, i1, j0, j1, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt )
    
else :
    
    sys.exit( "\nError: formulation should be 'theta_pi', 'T_rho_P' or 'theta_rho_P'.\n" )

def odefun( t, U ) :
    U, P = setGhostNodes( U )
    V = np.zeros(( 4, nLev+2, nCol+FD ))
    V[:,i0:i1,j0:j1] = implicitPart(U,P) + explicitPart(U,P)
    return V

###########################################################################

if semiImplicit == 1 :
    
    def ell( U ) :
        return common.L( U \
        , dtImp, setGhostNodes, implicitPart, nLev, nCol, FD \
        , i0, i1, j0, j1 )
    
    L = LinearOperator( ( 4*(nLev+2)*(nCol+FD), 4*(nLev+2)*(nCol+FD) ), matvec=ell, dtype=float )
    
    def leapfrogTimestep( t, U0, P0, U1, P1, dt ) :
        t, U2 = common.leapfrogTimestep( t, U0, P0, U1, P1, dt \
        , nLev, nCol, FD, i0, i1, j0, j1 \
        , L, implicitPart, explicitPart, gmresTol )
        return t, U2

###########################################################################

#Other functions:

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit( "\nError: rkStages should be 3 or 4.  rk2 is not stable for this problem.\n" )

def getStandardVariables( U ) :
    return common.getStandardVariables( U \
    , setGhostNodes, formulation, Tbar, Pbar, thetaBar, piBar, dpidsBar, dsdzAll \
    , Po, Rd, Cp, Cv, g )

def printInfo( U, P, et, t ) :
    U = getStandardVariables( U )
    return common.printInfo( U, P, et , t )

#Figure size and contour levels for plotting:
if ( saveContours == 1 ) | ( plotFromSaved == 1 ) :
    fig, CL = common.setFigAndContourLevels( testCase )

def saveContourPlot( U, t ) :
    U = getStandardVariables( U )
    common.saveContourPlot( U, t \
    , testCase, var, fig \
    , x, z, CL, FDo2 \
    , xLeft, xRight, zTop, dx, ds )

###########################################################################

#Eulerian time-stepping for first large time step

#Save initial conditions and contour of first frame:
U0, P0 = setGhostNodes( U0 )
if saveArrays == 1 :
    np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U0 )
U1 = U0
et = printInfo( U0, P0, time.clock(), t )
if saveContours == 1 :
    saveContourPlot( U0, t )

#The actual Eulerian time-stepping from t=0 to t=dtImp:
for i in range( np.int( np.round(dtImp/dtExp) + 1e-12 ) ) :
    t, U1 = rk( t, U1, odefun, dtExp )

U1, P1 = setGhostNodes( U1 )

###########################################################################

#The rest of the time-stepping:

for i in range(1,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dtImp)) ) == 0 :
        
        if plotFromSaved == 0 :
            if saveArrays == 1 :
                np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U1 )
        elif plotFromSaved == 1 :
            U1 = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        
        et = printInfo( U1, P1, et, t )
        
        if saveContours == 1 :
            saveContourPlot( U1, t )
        
    if plotFromSaved == 0 :
        if semiImplicit == 0 :
            t, U2 = rk( t, U1, odefun, dtImp )
        elif semiImplicit == 1 :
            t, U2 = leapfrogTimestep( t, U0, P0, U1, P1, dtImp )
        else :
            sys.exit( "\nError: semiImplicit should be 0 or 1.\n" )
        U2, P2 = setGhostNodes( U2 )
        U0 = U1
        U1 = U2
        P0 = P1
        P1 = P2
    else :
        t = t + dtImp

############################################################################
