#!/usr/bin/python3
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.annulus import common, eulerEquations

###########################################################################

args = eulerEquations.parseInput()

#get rid of the args prefix on all the variable names:
temporaryDictionary = vars(args)
for k in temporaryDictionary.keys() :
    exec("{} = args.{}".format(k,k))

dt = 1./dti
halfWidth = eval( halfWidth )

if plotFromSaved :
    saveContours = True

###########################################################################

rSurf, rSurfPrime \
= common.getTopoFunc( innerRadius, outerRadius, topoType \
, amp, frq, steepness, eval(ang1) )

ang1 = eval(ang1)                                  #convert string to float
xc1 = (rSurf(ang1)+(outerRadius-rSurf(ang1))/hf)*np.cos(ang1)      #x-coord
yc1 = (rSurf(ang1)+(outerRadius-rSurf(ang1))/hf)*np.sin(ang1)      #y-coord
if ang2 :
    ang2 = eval(ang2)                              #convert string to float
    xc2 = (rSurf(ang2)+outerRadius)/2.*np.cos(ang2)     #x-coord of GA bell
    yc2 = (rSurf(ang2)+outerRadius)/2.*np.sin(ang2)     #y-coord of GA bell
    if ang3 :
        ang3 = eval(ang3)                          #convert string to float
        xc3 = (rSurf(ang3)+outerRadius)/2.*np.cos(ang3) #x-coord of GA bell
        yc3 = (rSurf(ang3)+outerRadius)/2.*np.sin(ang3) #y-coord of GA bell

def initialCondition( x, y, kx, ky ) :
    #Gaussian:
    z = 1. + np.exp( -steepness*( (kx*(x-xc1))**2. + (ky*(y-yc1))**2. ) )
    if ang2 :
        z = z + np.exp( -steepness*( (x-xc2)**2. + (y-yc2)**2. ) )
        if ang3 :
            z = z + np.exp( -steepness*( (x-xc3)**2. + (y-yc3)**2. ) )
    return z
    # #Wendland function:
    # def wf( xc, yc ) :
    #     r = np.sqrt( 6. * ( (x-xc)**2. + (y-yc)**2. ) )
    #     ind = r<1.
    #     w = np.zeros( np.shape(x) )
    #     w[ind] = ( 1. - r[ind] ) ** 10. * ( 429.*r[ind]**4. + 450.*r[ind]**3. \
    #     + 210.*r[ind]**2. + 50.*r[ind] + 5.  )
    #     return w
    # z = wf(xc1,yc1)
    # if ang2 :
    #     z = z + wf(xc2,yc2)
    #     if ang3 :
    #         z = z + wf(xc3,yc3)
    # return 1. + z/5.

###########################################################################

#Delete old stuff, and set things up for saving:

saveString = eulerEquations.getSavestring( wavesOnly \
, testCase, innerRadius, outerRadius, tf, saveDel, steepness, amp, frq \
, VL, phs, pol, stc, clusterType, clusterStrength, rks, nlv, dti )

if ( saveArrays ) & ( not plotFromSaved ) :
    if os.path.exists( saveString + '*.npy' ) :
        os.remove( saveString + '*.npy' )                 #remove old files
    if not os.path.exists( saveString ) :
        os.makedirs( saveString )                     #make new directories

if saveContours :
    tmp = os.listdir( os.getcwd() )
    for item in tmp :
        if item.endswith(".png") :
            os.remove( os.path.join( os.getcwd(), item ) )

###########################################################################

#Atmospheric constants:
Cp = 1004.
Cv = 717.
Rd = Cp - Cv
g  = 9.81
Po = 10.**5.

###########################################################################

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

###########################################################################

#Get th vector using the prescribed clustering method:

clusterStrength = clusterStrength / 100.

if ( clusterType == "linear" ) | ( clusterType == "geometric" ) :
    th = common.fastAngles( innerRadius, outerRadius, nlv, ang1 \
    , clusterType, clusterStrength )
    nth = len(th)
else :
    #Get regularly spaced angles:
    clusterStrength = 0.
    nth = common.getNth( innerRadius, outerRadius, nlv )    #nmbr of angles
    th  = np.linspace( ang1, ang1+2.*np.pi, nth+1 )   #vector of all angles
    th  = th[0:-1]                       #remove last angle (same as first)

###########################################################################

#Get the main background state functions based on the test case:

e    = np.ones((  nlv, nth ))
null = np.zeros(( nlv, nth ))

exnerPressure, inverseExnerPressure \
, potentialTemperature, potentialTemperatureDerivative \
= eulerEquations.getExnerPressureAndPotentialTemperatureFunctions( testCase \
, innerRadius, Cp, g, e, null )

###########################################################################

#Get s-levels and computational mesh (thth,ss):

if verticalCoordinate == "height" :
    ds = ( outerRadius - innerRadius ) / (nlv-2)          #constant delta s
    s  = np.linspace( innerRadius-ds/2, outerRadius+ds/2, nlv )   #s vector
    thth, ss = np.meshgrid( th, s )            #mesh of s values and angles
    rr = common.getRadiiOnHeightCoordinateLevels( thth, ss \
    , innerRadius, outerRadius, rSurf )                      #mesh of radii
elif verticalCoordinate == "hybridSigma" :
    rr, s, ds \
    = eulerEquations.getRadiiOnPressureCoordinateLevels( nlv, nth, th \
    , rSurf, exnerPressure, inverseExnerPressure \
    , outerRadius, Po, Rd, Cp )
    thth, ss = np.meshgrid( th, s )
else :
    raise ValueError("verticalCoordinate should be 'height' or \
    'hybridSigma'")

###########################################################################

#Load th and s vectors if plotting from saved, otherwise save them:

if plotFromSaved :
    th = np.load( saveString + 'th' + '.npy' )    #load vector of th values
    s  = np.load( saveString + 's'  + '.npy' )     #load vector of s values
else :
    if saveArrays :
        np.save( saveString + 'th' + '.npy', th ) #save vector of th values
        np.save( saveString + 's'  + '.npy', s )   #save vector of s values

tmp = np.hstack(( th[-1]-2.*np.pi, th, th[0]+2.*np.pi ))
dth = ( tmp[2:nth+2] - tmp[0:nth] ) / 2.             #non-constant delta th

###########################################################################

#Get Cartesian mesh and some metric terms:

xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates

cosTh = np.cos(thth)                                                 #dr/dx
sinTh = np.sin(thth)                                                 #dr/dy
cosThOverR = cosTh/rr                                               #dth/dy
sinThOverR = sinTh/rr                                              #-dth/dx

drdx  =  np.cos(thth)
drdy  =  np.sin(thth)
dthdy =  np.cos(thth)/rr
dthdx = -np.sin(thth)/rr

###########################################################################

#Get height coordinate function and its derivatives:

sFunc, dsdth, dsdr \
= common.getHeightCoordinate( innerRadius, outerRadius, rSurf, rSurfPrime )

dsdthAll = dsdth( rr, thth )                 #dsdth values over entire mesh
dsdrAll  = dsdr( rr, thth )                   #dsdr values over entire mesh

###########################################################################

#Metric terms that will be used in Dx() and Dy() functions:

mtx = dsdrAll * drdx + dsdthAll * dthdx             #Dx() metric term
mty = dsdrAll * drdy + dsdthAll * dthdy             #Dy() metric term

###########################################################################

#Set (x,y) and r on the bottom boundary (B) and top boundary (T):

xB = rSurf(th)   * np.cos(th)
yB = rSurf(th)   * np.sin(th)
xT = outerRadius * np.cos(th)
yT = outerRadius * np.sin(th)

rB = rSurf(th)
rT = outerRadius

###########################################################################

#Set font sizes for any plots that might be requested below:

fst = 40                                               #font-size for title
fsc = 30                                            #font-size for colorbar
fsa = 30                                                #font-size for axes

###########################################################################

if plotNodes :
    
    plt.rc( 'font', size=fsa )
    fig = plt.figure()
    # fig = plt.figure( figsize=(13,12) )
    
    #Plot the nodes:
    
    plt.plot( xx.flatten(), yy.flatten(), ".", markersize=10 )
    plt.plot( np.hstack((xB,xB[0])), np.hstack((yB,yB[0])), "k-" \
    , np.hstack((xT,xT[0])), np.hstack((yT,yT[0])), "k-" \
    , linewidth=3 )
    tmp = outerRadius + .2
    plt.axis([-tmp,tmp,-tmp,tmp])
    plt.axis('image')
    # plt.xlabel( 'x' )
    # plt.ylabel( 'y' )
    plt.title( "verticalCoord={0:1s}, clusterType={1:1s}, clusterStrength={2:g}" \
    . format(verticalCoordinate,clusterType,clusterStrength) \
    , fontsize=fst )
    plt.show()

###########################################################################

if plotHeightCoord :
    
    plt.rc( 'font', size=fsa )
    
    #Plot the coordinate transformation functions:
    
    plt.contourf( xx, yy, sFunc(rr,thth), 20 )
    plt.plot( xB, yB, "r-", xT, yT, "r-" )
    plt.axis( 'equal' )
    tmp = plt.colorbar()
    tmp.ax.tick_params( labelsize=fsc )
    plt.title( 's(r,th)', fontsize=fst )
    plt.show()
    
    plt.contourf( xx, yy, dsdthAll, 20 )
    plt.plot( xB, yB, "r-", xT, yT, "r-" )
    plt.axis( 'equal' )
    tmp = plt.colorbar()
    tmp.ax.tick_params( labelsize=fsc )
    plt.title( 'ds/dth', fontsize=fst )
    plt.show()
    
    plt.contourf( xx, yy, dsdrAll, 20 )
    plt.plot( xB, yB, "r-", xT, yT, "r-" )
    plt.axis( 'equal' )
    tmp = plt.colorbar()
    tmp.ax.tick_params( labelsize=fsc )
    plt.title( 'ds/dr', fontsize=fst )
    plt.show()

###########################################################################

if plotRadii :
    
    plt.rc( 'font', size=fsa )
    
    #Plot the perturbed radii:
    
    fig, ax = plt.subplots( 1, 2, figsize=(8,4) )
    ax[0].plot( s0, s0, '-', s0, s, '.' )   #plot of initial vs perturbed s
    ax[0].set_xlabel('s0')
    ax[0].set_ylabel('s')
    ax[1].plot( s[1:-1], ds, '-' )            #plot of s vs non-constant ds
    ax[1].set_xlabel('s')
    ax[1].set_ylabel('ds')
    plt.show()

###########################################################################

if plotNodes or plotHeightCoord or plotRadii :
    
    sys.exit("\nFinished plotting.")

###########################################################################

#Hyperviscosity coefficient (alp) for radial direction:

c = np.sqrt( 287. * 300. )

if noRadialHV :
    alp = 0.                                   #remove radial HV completely
else :
    if pol == 1 :
        alp =  2.**-6.  * c
    elif pol == 3 :
        alp = -2.**-10. * c
    elif pol == 5 :
        alp =  2.**-14. * c
    elif pol == 7 :
        alp = -2.**-18. * c
    #######################
    elif pol == 2 :
        alp =  2.**-6.  * c
    elif pol == 4 :
        alp = -2.**-10. * c
    elif pol == 6 :
        alp =  2.**-14. * c
    elif pol == 8 :
        alp = -2.**-18. * c
    else :
        raise ValueError("1 <= pol <= 8")

###########################################################################

#Parameters for angular approximations:

if angularFD :
    #parameters for conventional FD8 approximation:
    phsA = 9
    polA = 8
    stcA = 9
else :
    #parameters for PHSFD approximation (polA=7 or polA=8):
    phsA = 9
    polA = 7
    stcA = 17

if noAngularHV :
    alpA = 0.                                 #remove angular HV completely
elif polA == 8 :
    alpA = -2.**-35. * c
elif polA == 7 :
    alpA = -2.**-35. * c
else :
    raise ValueError("polA should be 7 or 8 to achieve a high order \
    angular approximation (no boundaries in this direction).")

###########################################################################

#Extra things needed to enforce the Neumann boundary condition for P:

stcB = stc                  #stencil-size for enforcing boundary conditions
# stcB = min( nlv-1, 2*(pol+2)+1 )

NxBot, NyBot, NxTop, NyTop \
, TxBot, TyBot, TxTop, TyTop \
, someFactor, bottomFactor, topFactor \
= common.getTangentsAndNormals( th, stcB, rSurf, dsdr, dsdth, g )

###########################################################################

#Set initial condition for U[2,:,:] (T) and U[3,:,:] (rho):

U, thetaBar, piBar, Tbar, Pbar, rhoBar, phiBar \
, dTbarDr, drhoBarDr \
= eulerEquations.getInitialConditions( testCase, nlv, nth, initialCondition \
, xx, yy, kx, ky, rr, thth, innerRadius, ang1, Cp, Cv, Rd, g, Po )

###########################################################################

#Check the max value of the temperature perturbation on the boundaries:

print()
print( 'min/max(T) on boundaries =' \
, np.min(np.hstack(((U[2,-1,:]+U[2,1,:])/2.,(U[2,-1,:]+U[2,-2,:])/2.))) \
, np.max(np.hstack(((U[2,-1,:]+U[2,1,:])/2.,(U[2,-1,:]+U[2,-2,:])/2.))) )
print()

###########################################################################

#Radial PHS-FD weights arranged in a differentiation matrix:

#Matrix for approximating first derivative in radial direction:
Ws = phs1.getDM( x=s, X=s, m=1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

#Simple (but still correct with dsdr multiplier) radial HV:
Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )
Whvs = alp * ds**(phs-2) * Whvs                   #scaled radial HV matrix
# dsPol = spdiags( ds**(phs-2), np.array([0]), len(ds), len(ds) )
# Whvs = alp * dsPol.dot(Whvs)

###########################################################################

#Angular PHS-FD weights arranged in a differentiation matrix:

#Matrix for approximating first derivative in angular direction:
Wlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=1 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )

#Simple (and technically incorrect) angular HV:
Whvlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=phsA-1 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
# Whvlam = alpA * (ds0/innerRadius)**(phsA-2) * Whvlam     #scaled angular HV
dthPol = spdiags( dth**(phsA-2), np.array([0]), len(dth), len(dth) )
Whvlam = alpA * dthPol.dot(Whvlam)

###########################################################################

#Weights for interpolation to boundary (I), extrapolation to
#ghost-nodes (E), d/ds at boundary (D), and extrapolation to boundary(H):

wIinner = phs1.getWeights( innerRadius, s[0:stcB],   0, phs, pol )
wEinner = phs1.getWeights( s[0],        s[1:stcB+1], 0, phs, pol )
wDinner = phs1.getWeights( innerRadius, s[0:stcB],   1, phs, pol )
wHinner = phs1.getWeights( innerRadius, s[1:stcB+1], 0, phs, pol )

wIouter = phs1.getWeights( outerRadius, s[-1:-stcB-1:-1], 0, phs, pol )
wEouter = phs1.getWeights( s[-1],       s[-2:-stcB-2:-1], 0, phs, pol )
wDouter = phs1.getWeights( outerRadius, s[-1:-stcB-1:-1], 1, phs, pol )
wHouter = phs1.getWeights( outerRadius, s[-2:-stcB-2:-1], 0, phs, pol )

###########################################################################

#Weights to interpolate from perturbed mesh to regular mesh for plotting:

# Wradial = phs1.getDM( x=s, X=s0[1:-1], m=0 \
# , phsDegree=phs, polyDegree=pol, stencilSize=stc )
# 
# Wangular = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th0, m=0 \
# , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )

###########################################################################

#Functions to approximate differential operators and other things:
    
def Ds(U) :
    return Ws.dot(U)

def Dlam(U) :
    return Wlam.dot(U.T).T

# def Dr(U) :                                        #du/dr = (du/ds)*(ds/dr)
#     return Ds(U) * dsdrAll
# 
# def Dth(U) :                           #du/dth = du/dlam + (du/ds)*(ds/dth)
#     return Dlam(U) + Ds(U) * dsdthAll

def Dx(U) :                    #du/dx = (du/dr)*(dr/dx) + (du/dth)*(dth/dx)
    return mtx * Ds(U) + dthdx * Dlam(U)

def Dy(U) :                    #du/dy = (du/dr)*(dr/dy) + (du/dth)*(dth/dy)
    return mty * Ds(U) + dthdy * Dlam(U)

def HV(U) :
    return Whvs.dot(U) + Whvlam.dot(U[1:-1,:].T).T

def fastBackgroundStates( phi ) :
    Pbar, rhoBar, Tbar, drhoBarDr, dTbarDr \
    = eulerEquations.fastBackgroundStates( phi \
    , potentialTemperature, exnerPressure, potentialTemperatureDerivative \
    , g, innerRadius, Po, Cp, Rd )
    return Pbar, rhoBar, Tbar, drhoBarDr, dTbarDr

if mlv == 1 :
    def setGhostNodes( U ) :
        U, Pbar, rhoBar, Tbar, drhoBarDr, dTbarDr \
        = eulerEquations.setGhostNodesMidLevels( U \
        , NxBot, NyBot, NxTop, NyTop \
        , TxBot, TyBot, TxTop, TyTop \
        , fastBackgroundStates, e, null \
        , someFactor, bottomFactor, topFactor \
        , stcB, Wlam, Rd, cosTh[0,:], sinTh[0,:] \
        , wIinner, wEinner, wDinner, wHinner \
        , wIouter, wEouter, wDouter, wHouter \
        , innerRadius, outerRadius, rB, rT, g )
        return U, Pbar, rhoBar, Tbar, drhoBarDr, dTbarDr
elif mlv == 0 :
    raise ValueError("This isn't working for Euler equations yet.")
    # def setGhostNodes(U) :
    #     return eulerEquations.setGhostNodesInterfaces( U \
    #     , TxBot[0,:], TyBot[0,:], TxTop[0,:], TyTop[0,:] \
    #     , someFactor, stcB, Wlam \
    #     , wEinner, wDinner, wEouter, wDouter )
else :
    raise ValueError("Only mlv=1 please.")

dUdt = np.zeros(( 6, nlv, nth ))

if wavesOnly :
    def odefun( t, U, dUdt ) :
        dUdt = eulerEquations.odefunWaves( t, U, dUdt \
        , setGhostNodes, Dx, Dy, HV \
        , drdx, drdy \
        , Tbar, rhoBar, dTbarDr, drhoBarDr, Rd, Cv, g )
        return dUdt
else :
    def odefun( t, U, dUdt ) :
        dUdt = eulerEquations.odefunFast( t, U, dUdt \
        , setGhostNodes, Ds, Dlam, HV \
        , drdx, drdy \
        , phiBar, null \
        , Rd, Cv, g, innerRadius, VL, verticalCoordinate )
        # dUdt = eulerEquations.odefunEuler( t, U, dUdt \
        # , setGhostNodes, Dx, Dy, HV \
        # , drdx, drdy \
        # , Tbar, rhoBar, dTbarDr, drhoBarDr, Rd, Cv, g )
        return dUdt

q1 = dUdt               #let q1 be another reference to the same array dUdt
q2 = np.zeros( np.shape(dUdt) )       #rk3 and rk4 both need a second array

if rks == 3 :
    def RK( t, U ) :
        t, U = rk.rk3( t, U, odefun, dt, q1, q2 )
        return t, U
elif rks == 4 :
    q3 = np.zeros( np.shape(dUdt) )
    q4 = np.zeros( np.shape(dUdt) )
    def RK( t, U ) :
        t, U = rk.rk4( t, U, odefun, dt, q1, q2, q3, q4 )
        return t, U
else :
    raise ValueError("rks should be 3 or 4 in this problem.  \
    rks=1 and rks=2 are unstable.")

if saveContours :
    fig = plt.figure( figsize = (18,14) )

def plotSomething( U, t ) :
    eulerEquations.plotSomething( U, t \
    , testCase, Dx, Dy \
    , whatToPlot, xx, yy, th \
    , Rd, Po, Cp, xB, yB, xT, yT, outerRadius, fig \
    , dynamicColorbar, ang1, halfWidth \
    , Tbar, rhoBar, thetaBar, Pbar, piBar, phiBar )

###########################################################################

#Main time-stepping loop:

nullRemap = np.zeros(( 4, nlv, nth ))

et = time.time()

for i in np.arange( 0, nTimesteps+1 ) :
    
    #Vertical re-map:
    if np.mod(i,4)==0 and VL and not plotFromSaved :
        U, tmp, tmp, tmp, tmp, tmp = setGhostNodes( U )
        U[0:4,:,:] = eulerEquations.verticalRemap( U[0:4,:,:] \
        , U[4,:,:], phiBar, nlv-2, nullRemap )
        U[4,:,:] = phiBar
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        print( "t = {0:5d} | et = {1:6.2f} | maxAbsRho = {2:.2e}" \
        . format( np.int(np.round(t)) \
        , time.time()-et \
        , np.max(np.abs(U[3,:,:])) ) )
        
        et = time.time()
        
        if plotFromSaved :
            U[0:5,:,:] = np.load( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy' )
        
        if saveArrays or saveContours :
            U, tmp, tmp, tmp, tmp, tmp = setGhostNodes( U )
            if VL :
                U[0:4,:,:] = eulerEquations.verticalRemap( U[0:4,:,:] \
                , U[4,:,:], phiBar, nlv-2, nullRemap )
                U[4,:,:] = phiBar
                U, tmp, tmp, tmp, tmp, tmp = setGhostNodes( U )
        
        if saveArrays :
            np.save( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:] )
        
        if saveContours :
            plotSomething( U, t )
    
    if plotFromSaved :
        t = t + dt
    else :
        t, U = RK( t, U )

###########################################################################
