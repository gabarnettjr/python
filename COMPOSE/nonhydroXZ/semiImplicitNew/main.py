import sys
import os
import numpy as np
import time
from scipy.sparse.linalg import LinearOperator

sys.path.append( '../../../site-packages' )
from gab import rk, phs1
from gab.nonhydro import common

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "igw"

#"theta_pi" or "T_rho_P" or "theta_rho_P":
formulation  = "T_rho_P"

semiImplicit = 0
gmresTol     = 1e-6                                          #default: 1e-5

dx    = 500.
ds    = 125.
dtExp = 1./2.                                           #explicit time-step
dtImp = 1./1.                                           #implicit time-step

phs = 5
pol = 3
stc = 7

rkStages  = 3
plotNodes = 0                               #if 1, plot nodes and then exit
saveDel   = 100                           #print/save every saveDel seconds

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

saveString = saveString + formulation + '/' + testCase + '/'    \
+ 'phs' + '{0:1d}'.format(phs)                                  \
+ 'pol' + '{0:1d}'.format(pol)                                  \
+ 'stc' + '{0:1d}'.format(stc)                                  \
+ 'dx'  + '{0:1d}'.format(np.int(np.round(dx)+1e-12))           \
+ 'ds'  + '{0:1d}'.format(np.int(np.round(ds)+1e-12))           \
+ 'dti' + '{0:1d}'.format(np.int(np.round(1./dtExp)+1e-12))     \
+ '/'

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )

if not os.path.exists( saveString ) :
    os.makedirs( saveString )

###########################################################################

Cp, Cv, Rd, g, Po = common.getConstants()

tf = common.getTfinal( testCase )
nTimesteps = np.int( np.round(tf/dtImp) + 1e-12 )

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= common.getSpaceDomain( testCase, dx, ds )

sFunc, dsdx, dsdz = common.getHeightCoordinate( zTop, zSurf, zSurfPrime )

Tx, Tz, Nx, Nz = common.getTanNorm( zSurfPrime, x[0,:] )

U0, thetaBar, piBar, dpidsBar, Pbar, rhoBar, Tbar \
, dthetaBarDz, dpiBarDz, dTbarDz, dPbarDz, drhoBarDz \
= common.getInitialConditions( testCase, formulation \
, x, z \
, Cp, Cv, Rd, g, Po \
, dsdz )

thetaBarBot = ( thetaBar[0,   :] + thetaBar[1,     :] ) / 2.
thetaBarTop = ( thetaBar[nLev,:] + thetaBar[nLev+1,:] ) / 2.

if plotNodes == 1 :
    common.plotNodes( x, z, testCase )
    sys.exit( "\nDone plotting nodes." )
    
###########################################################################

#Derivatives of height coordinate function s, and stuff on interior only:

dsdxBot = dsdx( x[0,:], zSurf(x[0,:]) )
dsdzBot = dsdz( x[0,:], zSurf(x[0,:]) )
dsdxInt = dsdx( x[1:-1,:], z[1:-1,:] )
dsdzInt = dsdz( x[1:-1,:], z[1:-1,:] )
dsdxAll = dsdx( x, z )
dsdzAll = dsdz( x, z )

thetaBarInt    = thetaBar   [1:-1,:]
piBarInt       = piBar      [1:-1,:]
dthetaBarDzInt = dthetaBarDz[1:-1,:]
rhoBarInt      = rhoBar     [1:-1,:]
TbarInt        = Tbar       [1:-1,:]
drhoBarDzInt   = drhoBarDz  [1:-1,:]
dTbarDzInt     = dTbarDz    [1:-1,:]

s = sFunc( x[:,0], z[:,0] )

###########################################################################

#Define finite difference (FD) weights for derivative approximation:

if phs == 3 :
    alp = 2.**-7. * 300.
elif phs == 5 :
    alp = -2.**-5. * 300.
elif phs == 7 :
    alp = 2.**-9. * 300.
elif phs == 9 :
    alp = -2.**-13. * 300.
else :
    sys.exit("\nError: phs should be 3, 5, 7, or 9.\n")

Ws = phs1.getDM( x=s, X=s, m=1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvs = alp * ds**(phs-2) * Whvs

phsL = 9
polL = 7
stcL = 17
alpL = -2.**-13. * 300.

Wa = phs1.getPeriodicDM( period=xRight-xLeft, x=x[0,:], X=x[0,:], m=1 \
, phsDegree=phsL, polyDegree=polL, stencilSize=stcL )

Wa = np.transpose( Wa )

# alpDxPol = alpL * dx**(phsL-2)
Whva = phs1.getPeriodicDM( period=xRight-xLeft, x=x[0,:], X=x[0,:], m=phsL-1 \
, phsDegree=phsL, polyDegree=polL, stencilSize=stcL )
Whva = alpL * dx**(phsL-2) * Whva
Whva = np.transpose( Whva )

###########################################################################

#Define finite difference (FD) weights for boundary condition enforcement:

wIbot = phs1.getWeights( 0.,   s[0:stc],   m=0, phsDegree=phs, polyDegree=pol )
wEbot = phs1.getWeights( s[0], s[1:stc+1], m=0, phsDegree=phs, polyDegree=pol )
wDbot = phs1.getWeights( 0.,   s[0:stc],   m=1, phsDegree=phs, polyDegree=pol )

wItop = phs1.getWeights( zTop,  s[-1:-(stc+1):-1], m=0, phsDegree=phs, polyDegree=pol )
wEtop = phs1.getWeights( s[-1], s[-2:-(stc+2):-1], m=0, phsDegree=phs, polyDegree=pol )
wDtop = phs1.getWeights( zTop,  s[-1:-(stc+1):-1], m=1, phsDegree=phs, polyDegree=pol )

###########################################################################

# if FD == 2 :
    # wx = np.array( [ -1./2., 0., 1./2. ] )
    # wxhv = np.array( [ 1., -2., 1. ] )
    # gamma = 1./2.
# elif FD == 4 :
    # wx = np.array( [ 1./12., -2./3., 0., 2./3., -1./12. ] )
    # wxhv = np.array( [ 1., -4., 6., -4., 1. ] )
    # gamma = -1./12.
# elif FD == 6 :
    # wx = np.array( [ -1./60., 3./20., -3./4., 0., 3./4, -3./20., 1./60. ] )
    # wxhv = np.array( [ 1., -6., 15., -20., 15., -6., 1. ] )
    # gamma = 1./60.
# else :
    # sys.exit( "\nError: FD should be 2, 4, or 6.\n" )
# wx   = wx / dx
# wxhv = gamma * gx * wxhv / dx

# ws = np.array( [ -1./2., 0., 1./2. ] )
# wshv = np.array( [ 1., -2., 1. ] )
# ws   = ws / ds
# wshv = 1./2. * gs * wshv / ds

###########################################################################

TxBot = np.tile( Tx, (stc,1) )
TzBot = np.tile( Tz, (stc,1) )
NxBot = np.tile( Nx, (stc-1,1) )
NzBot = np.tile( Nz, (stc-1,1) )

TxTop = np.ones(( stc, nCol ))
TzTop = np.zeros(( stc, nCol ))
NxTop = np.zeros(( stc-1, nCol ))
NzTop = np.ones(( stc-1, nCol ))

normGradS = np.sqrt( dsdxBot**2. + dsdzBot**2. )

###########################################################################

#Define functions for approximating derivatives:

def Da( U ) :
    return U[1:-1,:] @ Wa

def Ds( U ) :
    return Ws[1:-1,:] @ U

def HV( U ) :
    # #Lateral HV:
    # HVL = ( U @ Wa ) + dsdxAll * ( Ws @ U )
    # for i in range( polL ) :
        # HVL = ( HVL @ Wa ) + dsdxAll * ( Ws @ HVL )
    # HVL = alpDxPol * HVL[1:-1,:]
    # #Total HV:
    # return dsdzInt*(Whvs@U) + HVL
    return ( U[1:-1,:] @ Whva ) + ( Whvs @ U )

###########################################################################

#Important functions for time stepping:

if formulation == "theta_pi" :
    
    from gab.nonhydro import theta_pi
    
    def setGhostNodes( U ) :
        U = theta_pi.setGhostNodes( U, Wa \
        , TxBot, TzBot, NxBot, NzBot \
        , TxTop, TzTop, NxTop, NzTop \
        , wIbot, wEbot, wDbot \
        , wItop, wEtop, wDtop \
        , nLev, nCol, stc, ds, thetaBarBot, thetaBarTop \
        , g, Cp, normGradS, dsdxBot, dsdzBot )
        P = []
        return U, P
    
    def implicitPart( U, P ) :
        return theta_pi.implicitPart( U \
        , Da, Ds, HV \
        , nLev, nCol, Cp, Cv, Rd, g \
        , dsdxInt, dsdzInt, thetaBarInt, piBarInt, dthetaBarDzInt )
    
    def explicitPart( U, P ) :
        return theta_pi.explicitPart( U \
        , Da, Ds \
        , nLev, nCol, Cp, Cv, Rd \
        , dsdxInt, dsdzInt )
    
elif formulation == "T_rho_P" :
    
    from gab.nonhydro import T_rho_P
    
    def setGhostNodes( U ) :
        U, P = T_rho_P.setGhostNodes( U, Wa \
        , TxBot, TzBot, NxBot, NzBot \
        , TxTop, TzTop, NxTop, NzTop \
        , wIbot, wEbot, wDbot \
        , wItop, wEtop, wDtop \
        , nLev, nCol, stc, ds, Pbar, rhoBar, Tbar \
        , g, Rd, normGradS, dsdxBot, dsdzBot )
        return U, P
    
    def implicitPart( U, P ) :
        return T_rho_P.implicitPart( U, P \
        , Da, Ds, HV \
        , nLev, nCol, Cp, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt, TbarInt, drhoBarDzInt, dTbarDzInt )
    
    def explicitPart( U, P ) :
        return T_rho_P.explicitPart( U, P \
        , Da, Ds \
        , nLev, nCol, Cv, Rd, g \
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
    V = np.zeros(( 4, nLev+2, nCol ))
    V[:,1:-1,:] = implicitPart(U,P) + explicitPart(U,P)
    return V

###########################################################################

if semiImplicit == 1 :
    
    def ell( U ) :
        return common.L( U \
        , dtImp, setGhostNodes, implicitPart, nLev, nCol )
    
    L = LinearOperator( ( 4*(nLev+2)*nCol, 4*(nLev+2)*nCol ), matvec=ell, dtype=float )
    
    def leapfrogTimestep( t, U0, P0, U1, P1, dt ) :
        t, U2 = common.leapfrogTimestep( t, U0, P0, U1, P1, dt \
        , nLev, nCol \
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
    , x, z, CL \
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
