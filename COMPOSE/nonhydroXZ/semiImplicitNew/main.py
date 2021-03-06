#!/usr/bin/python3
import sys
import os
import numpy as np
import time
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt

sys.path.append( '../../../site-packages' )
from gab import rk, phs1
from gab.nonhydro import common

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "bubble"

#"theta_pi" or "T_rho_P" or "theta_rho_P" or "HOMMEstyle":
formulation  = "HOMMEstyle"

VL = np.int64(sys.argv[1])              #if 1 then do vertically lagrangian
semiImplicit = 0                  #if 1 then do semi-implicit time-stepping
gmresTol     = 1e-6         #only matters if semiImplicit=1.  Default: 1e-5

dx    = np.float64(sys.argv[2])                         #horizontal spacing
ds    = np.float64(sys.argv[3])                           #vertical spacing
dtExp = 1./np.float64(sys.argv[4])                      #explicit time-step
dtImp = 1./2.                                           #implicit time-step

phs = 5                  #exponent of polyharmonic spline RBF (odd integer)
pol = 3                           #highest degree of polynomials to include
stc = 7                                                    #1D stencil-size

rkStages  = 3                        #number of Runge-Kutta stages (3 or 4)
plotNodes = 0                               #if 1, plot nodes and then exit
saveDel   = 100                           #print/save every saveDel seconds

var           = 2                        #determines what to plot (0,1,2,3)
saveArrays    = 1                   #if 1 then save arrays, if 0 then don't
saveContours  = 1                 #if 1 then save contours, if 0 then don't
plotFromSaved = 0           #if 1 then load results, if 0 then compute them

###########################################################################

if plotFromSaved == 1 :
    saveContours = 1

t = 0.

if semiImplicit == 1 :
    saveString = './semiImplicitResults'
elif semiImplicit == 0 :
    dtImp = dtExp
    saveString = './explicitResults'
else :
    sys.exit( "Error: semiImplicit should be 0 or 1." )

saveString = saveString + '/' + formulation + '/' + testCase + '/' \
+ 'phs' + '{0:1d}'.format(phs)                                     \
+ 'pol' + '{0:1d}'.format(pol)                                     \
+ 'stc' + '{0:1d}'.format(stc)                                     \
+ 'dx'  + '{0:1d}'.format(np.int(np.round(dx)+1e-12))              \
+ 'ds'  + '{0:1d}'.format(np.int(np.round(ds)+1e-12))              \
+ 'dti' + '{0:1d}'.format(np.int(np.round(1./dtExp)+1e-12))        \
+ '/'

if ( saveArrays == 1 ) & ( plotFromSaved == 0 ) :
    if os.path.exists( saveString + '*.npy' ) :
        os.remove( saveString + '*.npy' )
    if not os.path.exists( saveString ) :
        os.makedirs( saveString )

#Remove old *.png files:
if saveContours :
    tmp = os.listdir( os.getcwd() )
    for item in tmp :
        if item.endswith(".png") :
            os.remove( os.path.join( os.getcwd(), item ) )

###########################################################################

Cp, Cv, Rd, g, Po = common.getConstants()

tf = common.getTfinal( testCase )
# tf = 20.
nTimesteps = np.int( np.round(tf/dtImp) + 1e-12 )

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= common.getSpaceDomain( testCase, dx, ds )

zBot = zSurf(x[0,:])

sFunc, dsdx, dsdz = common.getHeightCoordinate( zTop, zSurf, zSurfPrime )

Tx, Tz, Nx, Nz = common.getTanNorm( zSurfPrime, x[0,:] )

U0, thetaBar, piBar, dpidsBar, Pbar, rhoBar, Tbar \
, dthetaBarDz, dpiBarDz, dTbarDz, dPbarDz, drhoBarDz \
= common.getInitialConditions( testCase, formulation \
, x, z \
, Cp, Cv, Rd, g, Po \
, dsdz )

backgroundStates = np.zeros(( 4, nLev+2, nCol ))
backgroundStates[0,:,:] = thetaBar
backgroundStates[1,:,:] = dpidsBar
backgroundStates[2,:,:] = Pbar
backgroundStates[3,:,:] = dthetaBarDz

thetaBarBot = ( thetaBar[0,   :] + thetaBar[1,     :] ) / 2.
thetaBarTop = ( thetaBar[nLev,:] + thetaBar[nLev+1,:] ) / 2.

if plotNodes == 1 :
    common.plotNodes( x, z, testCase )
    sys.exit( "\nDone plotting nodes." )
    
###########################################################################

#Derivatives of height coordinate function s, and stuff on interior only:

dsdxBot = dsdx( x[0,:], zSurf(x[0,:]) )
dsdzBot = dsdz( x[0,:], zSurf(x[0,:]) )
dsdxTop = dsdx( x[0,:], zTop )
dsdzTop = dsdz( x[0,:], zTop )
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
dpidsBarInt    = dpidsBar   [1:-1,:]
PbarInt        = Pbar       [1:-1,:]

s = sFunc( x[:,0], z[:,0] )

###########################################################################

#Define vertical (FD) weights for derivative approximation:

if phs == 3 :
    # alp = 2.**-1. * 300.
    alp = 2.**-7. * 300.
elif phs == 5 :
    alp = -2.**-5. * 300.
# elif phs == 7 :
#     alp = 2.**-9. * 300.
# elif phs == 9 :
#     alp = -2.**-13. * 300.
else :
    sys.exit("\nError: phs should be 3 or 5.\n")

Ws = phs1.getDM( x=s, X=s, m=1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )
Whvs = alp * ds**(phs-2) * Whvs

###########################################################################

#same thing but lateral:

phsL = 7
polL = 5
stcL = 13
alpL = 2.**-10. * 300.

Wa = phs1.getPeriodicDM( period=xRight-xLeft, x=x[0,:], X=x[0,:], m=1 \
, phsDegree=phsL, polyDegree=polL, stencilSize=stcL )

# alpDxPol = alpL * dx**(phsL-2)
Whva = phs1.getPeriodicDM( period=xRight-xLeft, x=x[0,:], X=x[0,:], m=phsL-1 \
, phsDegree=phsL, polyDegree=polL, stencilSize=stcL )
Whva = alpL * dx**(phsL-2) * Whva

###########################################################################

#Define finite difference (FD) weights for boundary condition enforcement:

wIbot = phs1.getWeights( 0.,   s[0:stc],   m=0, phsDegree=phs, polyDegree=pol )
wEbot = phs1.getWeights( s[0], s[1:stc+1], m=0, phsDegree=phs, polyDegree=pol )
wDbot = phs1.getWeights( 0.,   s[0:stc],   m=1, phsDegree=phs, polyDegree=pol )
wHbot = phs1.getWeights( 0.,   s[1:stc+1], m=0, phsDegree=phs, polyDegree=pol )

wItop = phs1.getWeights( zTop,  s[-1:-(stc+1):-1], m=0, phsDegree=phs, polyDegree=pol )
wEtop = phs1.getWeights( s[-1], s[-2:-(stc+2):-1], m=0, phsDegree=phs, polyDegree=pol )
wDtop = phs1.getWeights( zTop,  s[-1:-(stc+1):-1], m=1, phsDegree=phs, polyDegree=pol )
wHtop = phs1.getWeights( zTop,  s[-2:-(stc+2):-1], m=0, phsDegree=phs, polyDegree=pol )

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

def Da(U) :
    return Wa.dot(U[1:-1,:].T).T

def Ds(U) :
    return Ws[1:-1,:].dot(U)

def HV(U) :
    # #Lateral HV:
    # HVL = ( U @ Wa ) + dsdxAll * ( Ws @ U )
    # for i in range( polL ) :
    #     HVL = ( HVL @ Wa ) + dsdxAll * ( Ws @ HVL )
    # HVL = alpDxPol * HVL[1:-1,:]
    # #Total HV:
    # return dsdzInt*(Whvs@U) + HVL
    return Whva.dot(U[1:-1,:].T).T + Whvs.dot(U)

###########################################################################

#Important functions for time stepping:

def verticalRemap( U, z, Z ) :                           #used only if VL=1
    z = np.tile( z, (np.shape(U)[0],1,1) )
    Z = np.tile( Z, (np.shape(U)[0],1,1) )
    V = np.zeros( np.shape(U) )
    # #linear on bottom:
    # z0 = z[:,0,:]
    # z1 = z[:,1,:]
    # ZZ = Z[:,0,:]
    # V[:,0,:] = \
    #   ( ZZ - z1 ) * U[:,0,:] / ( z0 - z1 ) \
    # + ( ZZ - z0 ) * U[:,1,:] / ( z1 - z0 )
    # #linear on interior:
    # z0 = z[:,0:nLev+0,:]
    # z1 = z[:,2:nLev+2,:]
    # ZZ = Z[:,1:nLev+1,:]
    # V[:,1:nLev+1,:] = \
    #   ( ZZ - z1 ) * U[:,0:nLev+0,:] / ( z0 - z1 ) \
    # + ( ZZ - z0 ) * U[:,2:nLev+2,:] / ( z1 - z0 )
    # #linear on top:
    # z0 = z[:,-2,:]
    # z1 = z[:,-1,:]
    # ZZ = Z[:,-1,:]
    # V[:,-1,:] = \
    #   ( ZZ - z1 ) * U[:,-2,:] / ( z0 - z1 ) \
    # + ( ZZ - z0 ) * U[:,-1,:] / ( z1 - z0 )
    #quadratic on bottom:
    z0 = z[:,0,:]
    z1 = z[:,1,:]
    z2 = z[:,2,:]
    ZZ = Z[:,0,:]
    V[:,0,:] = \
      ( ZZ - z1 ) * ( ZZ - z2 ) * U[:,0,:] / ( z0 - z1 ) / ( z0 - z2 ) \
    + ( ZZ - z0 ) * ( ZZ - z2 ) * U[:,1,:] / ( z1 - z0 ) / ( z1 - z2 ) \
    + ( ZZ - z0 ) * ( ZZ - z1 ) * U[:,2,:] / ( z2 - z0 ) / ( z2 - z1 )
    #quadratic on interior:
    z0 = z[:,0:nLev+0,:]
    z1 = z[:,1:nLev+1,:]
    z2 = z[:,2:nLev+2,:]
    ZZ = Z[:,1:nLev+1,:]
    V[:,1:nLev+1,:] = \
      ( ZZ - z1 ) * ( ZZ - z2 ) * U[:,0:nLev+0,:] / ( z0 - z1 ) / ( z0 - z2 ) \
    + ( ZZ - z0 ) * ( ZZ - z2 ) * U[:,1:nLev+1,:] / ( z1 - z0 ) / ( z1 - z2 ) \
    + ( ZZ - z0 ) * ( ZZ - z1 ) * U[:,2:nLev+2,:] / ( z2 - z0 ) / ( z2 - z1 )
    #quadratic on top:
    z0 = z[:,nLev-1,:]
    z1 = z[:,nLev+0,:]
    z2 = z[:,nLev+1,:]
    ZZ = Z[:,nLev+1,:]
    V[:,nLev+1,:] = \
      ( ZZ - z1 ) * ( ZZ - z2 ) * U[:,nLev-1,:] / ( z0 - z1 ) / ( z0 - z2 ) \
    + ( ZZ - z0 ) * ( ZZ - z2 ) * U[:,nLev+0,:] / ( z1 - z0 ) / ( z1 - z2 ) \
    + ( ZZ - z0 ) * ( ZZ - z1 ) * U[:,nLev+1,:] / ( z2 - z0 ) / ( z2 - z1 )
    return V
    #############################
    # #VERY SLOW PHS re-map (to verify fast quadratic one above):
    # V = np.zeros( np.shape(U) )
    # pages = np.shape(U)[0]
    # for j in range( nCol ) :
    #     W = phs1.getDM( x=z[:,j], X=Z[:,j], m=0 \
    #     , phsDegree=phs, polyDegree=pol, stencilSize=stc )
    #     for i in range(pages) :
    #         V[i,:,j] = W.dot( U[i,:,j] )
    # return V

def verticalDerivative( U, z ) :
    z = np.tile( z, (np.shape(U)[0],1,1) )
    V = np.zeros( np.shape(U) )
    #derivative of parabola on bottom:
    z0 = z[:,0,:]
    z1 = z[:,1,:]
    z2 = z[:,2,:]
    ZZ = z[:,1,:]
    V[:,0,:] \
    = ( ( ZZ - z1 ) + ( ZZ - z2 ) ) * U[:,0,:] / ( z0 - z1 ) / ( z0 - z2 ) \
    + ( ( ZZ - z0 ) + ( ZZ - z2 ) ) * U[:,1,:] / ( z1 - z0 ) / ( z1 - z2 ) \
    + ( ( ZZ - z0 ) + ( ZZ - z1 ) ) * U[:,2,:] / ( z2 - z0 ) / ( z2 - z1 )
    #derivative of parabola on interior:
    z0 = z[:,0:nLev+0,:]
    z1 = z[:,1:nLev+1,:]
    z2 = z[:,2:nLev+2,:]
    ZZ = z[:,1:nLev+1,:]
    V[:,1:nLev+1,:] \
    = ( ( ZZ - z1 ) + ( ZZ - z2 ) ) * U[:,0:nLev+0,:] / ( z0 - z1 ) / ( z0 - z2 ) \
    + ( ( ZZ - z0 ) + ( ZZ - z2 ) ) * U[:,1:nLev+1,:] / ( z1 - z0 ) / ( z1 - z2 ) \
    + ( ( ZZ - z0 ) + ( ZZ - z1 ) ) * U[:,2:nLev+2,:] / ( z2 - z0 ) / ( z2 - z1 )
    #derivative of parabola on top:
    z0 = z[:,-3,:]
    z1 = z[:,-2,:]
    z2 = z[:,-1,:]
    ZZ = z[:,-1,:]
    V[:,-1,:] \
    = ( ( ZZ - z1 ) + ( ZZ - z2 ) ) * U[:,-3,:] / ( z0 - z1 ) / ( z0 - z2 ) \
    + ( ( ZZ - z0 ) + ( ZZ - z2 ) ) * U[:,-2,:] / ( z1 - z0 ) / ( z1 - z2 ) \
    + ( ( ZZ - z0 ) + ( ZZ - z1 ) ) * U[:,-1,:] / ( z2 - z0 ) / ( z2 - z1 )
    return V

# #check vertical remap/derivative:
# tmp = ds/4.
# ran = tmp * ( -1. + 2. * np.random.rand(np.shape(U0)[1],np.shape(U0)[2]) )
# ztmp = z + ran
# tmp0 = U0
# tmp = verticalRemap( tmp0, z, ztmp )
# # tmp = verticalDerivative( tmp, ztmp )
# tmp = verticalRemap( tmp, ztmp, z )
# tmpA = tmp[2,:,:]
# tmpB = tmp0[2,:,:]
# # tmpB = verticalDerivative(tmp0,z)[2,:,:]
# tmp = tmpB - tmpA
# fig, ax = plt.subplots( 3, 1 )
# alp = 0.
# bet = .01
# plot0 = ax[0].contourf( x, z, tmpA, np.linspace(alp,bet,20) )
# plot1 = ax[1].contourf( x, z, tmpB, np.linspace(alp,bet,20) )
# plot2 = ax[2].contourf( x, z, tmp, 20 )
# for i in range(len(ax)) :
#     # ax[i].set_aspect('equal')
#     plt.colorbar( eval('plot{0:d}'.format(i)), ax=ax[i] )
# plt.show()
# sys.exit("\nStop here for now.\n")

dUdt  = np.zeros(( np.shape(U0)[0], nLev+2, nCol ))         #used in odefun
dUdti = np.zeros(( np.shape(U0)[0], nLev,   nCol ))     #not being used yet

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
        backgroundStatesTmp = []
        return U, P, backgroundStatesTmp
    
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
    
    def odefun( t, U, dUdt ) :
        U, P, tmp = setGhostNodes( U )
        dUdt[:,1:-1,:] = implicitPart(U,P) + explicitPart(U,P)
        return dUdt
    
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
        backgroundStatesTmp = []
        return U, P, backgroundStatesTmp
    
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
    
    def odefun( t, U, dUdt ) :
        U, P, tmp = setGhostNodes( U )
        dUdt[:,1:-1,:] = implicitPart(U,P) + explicitPart(U,P)
        return dUdt
    
elif formulation == "theta_rho_P" :
    
    from gab.nonhydro import theta_rho_P
    
    def setGhostNodes( U ) :
        U, P = theta_rho_P.setGhostNodes( U, Wa \
        , TxBot, TzBot, NxBot, NzBot \
        , TxTop, TzTop, NxTop, NzTop \
        , wIbot, wEbot, wDbot \
        , wItop, wEtop, wDtop \
        , nLev, nCol, stc, ds, Pbar, rhoBar, thetaBar \
        , g, Rd, Cp, Cv, Po, normGradS, dsdxBot, dsdzBot )
        backgroundStatesTmp = []
        return U, P, backgroundStatesTmp
    
    def implicitPart( U, P ) :
        return theta_rho_P.implicitPart( U, P \
        , Da, Ds, HV \
        , nLev, nCol, Cp, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt, drhoBarDzInt, dthetaBarDzInt )
    
    def explicitPart( U, P ) :
        return theta_rho_P.explicitPart( U, P \
        , Da, Ds \
        , nLev, nCol, Cv, Rd, g \
        , dsdxInt, dsdzInt, rhoBarInt )
    
    def odefun( t, U, dUdt ) :
        U, P, tmp = setGhostNodes( U )
        dUdt[:,1:-1,:] = implicitPart(U,P) + explicitPart(U,P)
        return dUdt

elif formulation == "HOMMEstyle" :
    
    from gab.nonhydro import HOMMEstyle
    
    def setGhostNodes( U ) :
        U, P, backgroundStatesTmp = HOMMEstyle.setGhostNodes( U \
        , verticalRemap, dsdz, testCase \
        , Wa, Ws \
        , TxBot, TzBot, NxBot, NzBot \
        , TxTop, TzTop, NxTop, NzTop \
        , wIbot, wEbot, wDbot, wHbot \
        , wItop, wEtop, wDtop, wHtop \
        , dsdxBot, dsdzBot, dsdxTop, dsdzTop \
        , nLev, nCol, stc, backgroundStates \
        , Po, g, Rd, Cv, Cp, zBot, zTop, x, z, VL )
        return U, P, backgroundStatesTmp
    
    def implicitPart( U, P ) :
        return HOMMEstyle.implicitPart( U, P \
        , nLev, nCol )
    
    def explicitPart( U, P, backgroundStatesTmp ) :
        return HOMMEstyle.explicitPart( U, P, backgroundStatesTmp \
        , Da, Ds, HV, verticalRemap \
        , nLev, nCol, g, wIbot, wItop, stc, z, VL )
    
    
    def odefun( t, U, dUdt ) :
        U, P, backgroundStatesTmp = setGhostNodes( U )
        dUdt[:,1:-1,:] = implicitPart( U, P ) \
        + explicitPart( U, P, backgroundStatesTmp )
        # dUdt[4,0,:]  = -U[0,0,:]  * ( U[4,0,:] @ Wa ) + g*U[1,0,:]
        # dUdt[4,-1,:] = -U[0,-1,:] * ( U[4,-1,:] @ Wa ) + g*U[1,-1,:]
        return dUdt
    
else :
    
    sys.exit( "\nError: formulation should be 'theta_pi', 'T_rho_P' or 'theta_rho_P'.\n" )

###########################################################################

if semiImplicit == 1 :
    
    if formulation == "HOMMEstyle" :
        
        sys.exit("\nError: HOMMEstyle doesn't support semi-implicit yet.\n")
    
    else :
        
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

q1 = dUdt
q2 = np.zeros( np.shape(U0) )
if rkStages == 3 :
    def RK( t, U ):
        t, U = rk.rk3( t, U, odefun, dtExp, q1, q2 )
        return t, U
elif rkStages == 4 :
    q3 = np.zeros( np.shape(U0) )
    q4 = np.zeros( np.shape(U0) )
    def RK( t, U ) :
        t, U = rk.rk4( t, U, odefun, dtExp, q1, q2, q3, q4 )
        return t, U
else :
    sys.exit( "\nError: rkStages should be 3 or 4.  rk2 is not stable for this problem.\n" )

def getStandardVariables( U ) :
    #Regardless of the formulation, this returns U=(u,w,theta,pi,(phi)).
    return common.getStandardVariables( U \
    , setGhostNodes, formulation, Tbar, Pbar, thetaBar, piBar, dpidsBar, dsdzAll \
    , Po, Rd, Cp, Cv, g )

def printInfo( U, P, et, t ) :
    U = getStandardVariables( U )
    return common.printInfo( U, P, et , t )

#Figure size and contour levels for plotting:
if ( saveContours == 1 ) | ( plotFromSaved == 1 ) :
    fig, CL = common.setFigAndContourLevels( testCase )

def saveContourPlot( U, t, z ) :
    U = getStandardVariables( U )
    common.saveContourPlot( U, t, z \
    , testCase, var, fig \
    , x, CL \
    , xLeft, xRight, zTop, dx, ds )

###########################################################################

#Eulerian time-stepping for first large time step

#Save initial conditions and contour of first frame:
U0, P0, tmp = setGhostNodes( U0 )
if saveArrays == 1 :
    np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U0 )
U1 = U0
et = time.time()
et = printInfo( U0, P0, et, t )
if saveContours == 1 :
    saveContourPlot( U0, t, z )

#The actual Eulerian time-stepping from t=0 to t=dtImp:
for i in range( np.int( np.round(dtImp/dtExp) + 1e-12 ) ) :
    t, U1 = RK( t, U1 )

U1, P1, tmp = setGhostNodes( U1 )

###########################################################################

#The rest of the time-stepping:

for i in range( 1, nTimesteps+1 ) :
    
    if np.mod( i, np.int(np.round(saveDel/dtImp)) ) == 0 :
        
        if plotFromSaved == 0 :
            if saveArrays == 1 :
                if ( formulation == "HOMMEstyle" ) & ( VL == 1 ) :
                    U1 = verticalRemap( U1, U1[4,:,:]/g, z )
                    U1, P1, tmp = setGhostNodes( U1 )
                np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U1 )
        elif plotFromSaved == 1 :
            U1 = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        
        if saveContours == 1 :
            if ( formulation == "HOMMEstyle" ) & ( VL == 1 ) :
                U1 = verticalRemap( U1, U1[4,:,:]/g, z )
                U1, P1, tmp = setGhostNodes( U1 )
            saveContourPlot( U1, t, z )
        
        et = printInfo( U1, P1, et, t )

    if plotFromSaved == 0 :
        if semiImplicit == 0 :
            if ( np.mod(i,1) == 0 ) & ( formulation == "HOMMEstyle" ) & ( VL == 1 ) :
                U1 = verticalRemap( U1, U1[4,:,:]/g, z )
                U1, P1, tmp = setGhostNodes( U1 )
            t, U2 = RK( t, U1 )
        elif semiImplicit == 1 :
            t, U2 = leapfrogTimestep( t, U0, P0, U1, P1, dtImp )
        else :
            sys.exit( "\nError: semiImplicit should be 0 or 1.\n" )
        U2, P2, tmp = setGhostNodes( U2 )
        U0 = U1
        U1 = U2
        P0 = P1
        P1 = P2
    else :
        t = t + dtImp

############################################################################
