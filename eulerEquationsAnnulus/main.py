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

if plotFromSaved :
    saveContours = True

###########################################################################

rSurf, rSurfPrime \
= common.getTopoFunc( innerRadius, outerRadius, amp, frq )

ang1 = eval(ang1)                                  #convert string to float
xc1 = (rSurf(ang1)+(outerRadius-rSurf(ang1))/3.)*np.cos(ang1)#x-coord of GA
yc1 = (rSurf(ang1)+(outerRadius-rSurf(ang1))/3.)*np.sin(ang1)#y-coord of GA
if ang2 :
    ang2 = eval(ang2)                              #convert string to float
    xc2 = (rSurf(ang2)+outerRadius)/2.*np.cos(ang2)     #x-coord of GA bell
    yc2 = (rSurf(ang2)+outerRadius)/2.*np.sin(ang2)     #y-coord of GA bell
    if ang3 :
        ang3 = eval(ang3)                          #convert string to float
        xc3 = (rSurf(ang3)+outerRadius)/2.*np.cos(ang3) #x-coord of GA bell
        yc3 = (rSurf(ang3)+outerRadius)/2.*np.sin(ang3) #y-coord of GA bell

def initialCondition( x, y ) :
    #Gaussian:
    z = 1. + np.exp( -exp*( (x-xc1)**2. + (y-yc1)**2. ) )
    if ang2 :
        z = z + np.exp( -exp*( (x-xc2)**2. + (y-yc2)**2. ) )
        if ang3 :
            z = z + np.exp( -exp*( (x-xc3)**2. + (y-yc3)**2. ) )
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
, innerRadius, outerRadius, tf, saveDel, exp, amp, frq \
, mlv, phs, pol, stc, clu, pct, rks, nlv, dti )

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

#Get th and s vectors and save them:

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

pct = pct / 100.
if clu == "linear" :
    th = common.fastAngles( innerRadius, outerRadius, nlv, ang1, clu, pct )
    nth = len(th)
elif clu == "geometric" :
    th = common.fastAngles( innerRadius, outerRadius, nlv, ang1, clu, pct )
    nth = len(th)
else :
    #Get regularly spaced angles:
    pct = 0.
    nth = common.getNth( innerRadius, outerRadius, nlv )    #nmbr of angles
    th  = np.linspace( ang1, ang1+2.*np.pi, nth+1 )   #vector of all angles
    th  = th[0:-1]                       #remove last angle (same as first)

th0 = np.linspace( ang1, ang1+2.*np.pi, nth+1 )
th0 = th0[0:-1]

if mlv == 1 :
    ds0 = ( outerRadius - innerRadius ) / (nlv-2)         #constant delta s
    s0  = np.linspace( innerRadius-ds0/2, outerRadius+ds0/2, nlv )#s vector
    # tmp = (ptr/100.) * ds0                    #relative perturbation factor
    # ran = -tmp + 2.*tmp*np.random.rand(nlv)     #random perturbation vector
    s   = s0.copy()                                  #copy regular s vector
    # s   = s + ran                              #s vector after perturbation
elif mlv == 0 :
    ds0 = ( outerRadius - innerRadius ) / (nlv-3)         #constant delta s
    s0  = np.linspace( innerRadius-ds0, outerRadius+ds0, nlv )    #s vector
    # tmp = (ptr/100.) * ds0                    #relative perturbation factor
    # ranBot = -tmp + 2.*tmp*np.random.rand(1)
    # ranMid = -tmp + 2.*tmp*np.random.rand(nlv-4)
    # ranTop = -tmp + 2.*tmp*np.random.rand(1)
    s = s0.copy()                                    #copy regular s vector
    # s[0] = s[0] + ranBot                         #perturb bottom ghost node
    # s[2:-2] = s[2:-2] + ranMid           #perturb interior nodes (no bndry)
    # s[-1] = s[-1] + ranTop                          #perturb top ghost node
else :
    raise ValueError("mlv should be 0 or 1")

if plotFromSaved :
    th = np.load( saveString + 'th' + '.npy' )    #load vector of th values
    s  = np.load( saveString + 's'  + '.npy' )     #load vector of s values
else :
    if saveArrays :
        np.save( saveString + 'th' + '.npy', th ) #save vector of th values
        np.save( saveString + 's'  + '.npy', s )   #save vector of s values

tmp = np.hstack(( th[-1]-2.*np.pi, th, th[0]+2.*np.pi ))
dth = ( tmp[2:nth+2] - tmp[0:nth] ) / 2.             #non-constant delta th
ds  = ( s[2:nlv] - s[0:nlv-2] ) / 2.                  #non-constant delta s

###########################################################################

#Get computational mesh and mesh for contour plotting:

thth, ss = np.meshgrid( th, s )      #mesh of perturbed s values and angles
rr = common.getRadii( thth, ss \
, innerRadius, outerRadius, rSurf )                #mesh of perturbed radii
xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates
xxi = xx[1:-1,:]                                    #x without ghost layers
yyi = yy[1:-1,:]                                    #y without ghost layers
ththi = thth[1:-1,:]                               #th without ghost layers
rri = rr[1:-1,:]                                    #r without ghost layers

cosTh = np.cos(thth)                                                 #dr/dx
sinTh = np.sin(thth)                                                 #dr/dy
cosThOverR = cosTh/rr                                               #dth/dy
sinThOverR = sinTh/rr                                              #-dth/dx

drdxAll  =  np.cos(thth)
drdyAll  =  np.sin(thth)
dthdyAll =  np.cos(thth)/rr
dthdxAll = -np.sin(thth)/rr

thth0, ss0 = np.meshgrid( th0, s0[1:-1] )        #regular mesh for plotting
rr0 = common.getRadii( thth0, ss0 \
, innerRadius, outerRadius, rSurf )                  #mesh of regular radii
xx0 = rr0 * np.cos(thth0)                         #mesh of regular x-coords
yy0 = rr0 * np.sin(thth0)                         #mesh of regular x-coords

###########################################################################

#Get height coordinate s and its derivatives:

sFunc, dsdth, dsdr \
= common.getHeightCoordinate( innerRadius, outerRadius, rSurf, rSurfPrime )

dsdthi = dsdth( rri, ththi )             #interior values of dsdth function
dsdri  = dsdr( rri, ththi )               #interior values of dsdr function

dsdthAll = dsdth( rr, thth )                 #dsdth values over entire mesh
dsdrAll  = dsdr( rr, thth )                   #dsdr values over entire mesh

###########################################################################

#Metric terms that will be used in Dx() and Dy() functions:

mtx = dsdrAll * drdxAll + dsdthAll * dthdxAll             #Dx() metric term
mty = dsdrAll * drdyAll + dsdthAll * dthdyAll             #Dy() metric term

###########################################################################

#Set (x,y) on the bottom boundary (B) and top boundary (T):

xB0 = rSurf(th0) * np.cos(th0)
yB0 = rSurf(th0) * np.sin(th0)
xT0 = outerRadius * np.cos(th0)
yT0 = outerRadius * np.sin(th0)

xT = outerRadius * np.cos(th)
yT = outerRadius * np.sin(th)
xB = rSurf(th)   * np.cos(th)
yB = rSurf(th)   * np.sin(th)

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
    plt.title( "clu={0:1s}, pct={1:1g}, ptr={2:1d}".format(clu,pct,ptr), fontsize=fst )
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
        alp = -2.**-10.  * c
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

#Atmospheric constants:
Cp = 1004.
Cv = 717.
Rd = Cp - Cv
g  = 9.81
Po = 10.**5.

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

U = np.zeros(( 5, nlv, nth ))

#Hydrostatic background states and initial theta perturbation:
thetaBar = 300. * np.ones(( nlv, nth ))
# thetaPrime = np.zeros(( nlv, nth ))
thetaPrime = 2.*( initialCondition(xx,yy) - 1. )
piBar = 1. - g / Cp / thetaBar * ( rr - innerRadius )
piPrime = np.zeros(( nlv, nth ))
Tbar = piBar * thetaBar
Tprime = ( piBar + piPrime ) * ( thetaBar + thetaPrime ) - Tbar
Pbar = Po * piBar ** (Cp/Rd)
Pprime = Po * ( piBar + piPrime ) ** (Cp/Rd) - Pbar
rhoBar = Pbar / Rd / Tbar
rhoPrime = ( Pbar + Pprime ) / Rd / ( Tbar + Tprime ) - rhoBar

#Initial condition for temperature and density perturbations:
U[2,:,:] = Tprime
U[3,:,:] = rhoPrime

#Radial derivatives of hydrostatic background states:
#For derivation, please see "Background States" section of
#https://www.overleaf.com/read/jpddcpggwyhh
dthetaBarDr = np.zeros(( nlv, nth ))
dpiBarDr = -g / Cp / thetaBar
dTbarDr = piBar * dthetaBarDr + thetaBar * dpiBarDr
dPbarDr = Po * Cp/Rd * piBar**(Cp/Rd-1.) * dpiBarDr
drhoBarDr = ( dPbarDr - Rd*rhoBar*dTbarDr ) / ( Rd * Tbar )

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
Whvs = alp * ds0**(phs-2) * Whvs                   #scaled radial HV matrix
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

Wradial = phs1.getDM( x=s, X=s0[1:-1], m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Wangular = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th0, m=0 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )

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
    return mtx * Ds(U) + dthdxAll * Dlam(U)

def Dy(U) :                    #du/dy = (du/dr)*(dr/dy) + (du/dth)*(dth/dy)
    return mty * Ds(U) + dthdyAll * Dlam(U)

def HV(U) :
    return Whvs.dot(U) + Whvlam.dot(U[1:-1,:].T).T

if mlv == 1 :
    def setGhostNodes( U ) :
        U = eulerEquations.setGhostNodesMidLevels( U \
        , NxBot, NyBot, NxTop, NyTop \
        , TxBot, TyBot, TxTop, TyTop \
        , rhoBar, Tbar, Pbar \
        , someFactor, bottomFactor, topFactor \
        , stcB, Wlam, Rd \
        , wIinner, wEinner, wDinner, wHinner \
        , wIouter, wEouter, wDouter, wHouter )
        return U
elif mlv == 0 :
    raise ValueError("This isn't working for Euler equations yet.")
    # def setGhostNodes(U) :
    #     return eulerEquations.setGhostNodesInterfaces( U \
    #     , TxBot[0,:], TyBot[0,:], TxTop[0,:], TyTop[0,:] \
    #     , someFactor, stcB, Wlam \
    #     , wEinner, wDinner, wEouter, wDouter )
else :
    raise ValueError("Only mlv=1 please.")

dUdt = np.zeros(( 5, nlv, nth ))
if wavesOnly :
    def odefun( t, U, dUdt ) :
        dUdt = eulerEquations.odefunWaves( t, U, dUdt \
        , setGhostNodes, Dx, Dy, HV \
        , drdxAll, drdyAll \
        , Tbar, rhoBar, dTbarDr, drhoBarDr, Rd, Cv, g )
        return dUdt
else :
    def odefun( t, U, dUdt ) :
        dUdt = eulerEquations.odefunEuler( t, U, dUdt \
        , setGhostNodes, Dx, Dy, HV \
        , drdxAll, drdyAll \
        , Tbar, rhoBar, dTbarDr, drhoBarDr, Rd, Cv, g )
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
    , Dx, Dy \
    , whatToPlot, xx, yy, th, xx0, yy0, th0, Wradial, Wangular \
    , Rd, Po, Cp, xB, yB, xT, yT, outerRadius, fig \
    , dynamicColorbar, interp, ang1 \
    , Tbar, rhoBar, thetaBar, Pbar, piBar )

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange( 0, nTimesteps+1 ) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        print( "t = {0:5d} | et = {1:6.2f} | maxAbsRho = {2:.2e}" \
        . format( np.int(np.round(t)) \
        , time.time()-et \
        , np.max(np.abs(U[3,:,:])) ) )
        
        et = time.time()
        
        if plotFromSaved :
            U[0:4,:,:] = np.load( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy' )
        
        if saveArrays or saveContours :
            U = setGhostNodes(U)
        
        if saveArrays :
            np.save( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:4,:,:] )
        
        if saveContours :
            plotSomething( U, t )
    
    if plotFromSaved :
        t = t + dt
    else :
        t, U = RK( t, U )

###########################################################################
