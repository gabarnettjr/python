#!/usr/bin/python3
import sys
import os
import time
import numpy as np
# from scipy.sparse import spdiags
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.annulus import common, waveEquation

###########################################################################

args = waveEquation.parseInput()

#get rid of the args prefix on all the variable names:
temporaryDictionary = vars(args)
for k in temporaryDictionary.keys() :
    exec("{} = args.{}".format(k,k))

dt = 1./dti

if plotFromSaved :
    saveContours = True

###########################################################################

rSurf, rSurfPrime \
= common.getTopoFunc( innerRadius, outerRadius, "trig", amp, frq, 0, 0 )

ang1 = eval(ang1)                                  #convert string to float
xc1 = (rSurf(ang1)+outerRadius)/2.*np.cos(ang1)         #x-coord of GA bell
yc1 = (rSurf(ang1)+outerRadius)/2.*np.sin(ang1)         #y-coord of GA bell
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
    # r = np.sqrt( 6 * ( (x-xc1)**2. + (y-yc1)**2. ) )
    # ind = r<1.
    # z = np.zeros( np.shape(x) )
    # z[ind] = ( 1. - r[ind] ) ** 10. * ( 429.*r[ind]**4. + 450.*r[ind]**3. \
    # + 210.*r[ind]**2. + 50.*r[ind] + 5.  )
    # z = 1. + z/5.
    # return z

###########################################################################

#Delete old stuff, and set things up for saving:

saveString = waveEquation.getSavestring( c, innerRadius, outerRadius \
, tf, saveDel, exp, amp, frq \
, mlv, phs, pol, stc, pta, ptr, rks, nlv, dti )

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

#Get th and s vectors and save them (randomly perturbed):

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

nth = common.getNth( innerRadius, outerRadius, nlv )#nmbr of angular levels
dth0 = 2.*np.pi / nth                                 #constant delta theta
th0  = np.linspace( 0., 2.*np.pi, nth+1 )             #vector of all angles
th0  = th0[0:-1]                         #remove last angle (same as first)
tmp = (pta/100.) * dth0                       #relative perturbation factor
ran = -tmp + 2.*tmp*np.random.rand(nth)         #random perturbation vector
th = th0.copy()                                     #copy regular th vector
th = th + ran                                 #th vector after perturbation

if mlv == 1 :
    ds0 = ( outerRadius - innerRadius ) / (nlv-2)         #constant delta s
    s0  = np.linspace( innerRadius-ds0/2, outerRadius+ds0/2, nlv )#s vector
    tmp = (ptr/100.) * ds0                    #relative perturbation factor
    ran = -tmp + 2.*tmp*np.random.rand(nlv)     #random perturbation vector
    s   = s0.copy()                                  #copy regular s vector
    s   = s + ran                              #s vector after perturbation
elif mlv == 0 :
    ds0 = ( outerRadius - innerRadius ) / (nlv-3)         #constant delta s
    s0  = np.linspace( innerRadius-ds0, outerRadius+ds0, nlv )    #s vector
    tmp = (ptr/100.) * ds0                    #relative perturbation factor
    ranBot = -tmp + 2.*tmp*np.random.rand(1)
    ranMid = -tmp + 2.*tmp*np.random.rand(nlv-4)
    ranTop = -tmp + 2.*tmp*np.random.rand(1)
    s = s0.copy()                                    #copy regular s vector
    s[0] = s[0] + ranBot                         #perturb bottom ghost node
    s[2:-2] = s[2:-2] + ranMid           #perturb interior nodes (no bndry)
    s[-1] = s[-1] + ranTop                          #perturb top ghost node
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
rr = common.getRadiiOnHeightCoordinateLevels( thth, ss \
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
rr0 = common.getRadiiOnHeightCoordinateLevels( thth0, ss0 \
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

xB = rSurf(th0) * np.cos(th0)
yB = rSurf(th0) * np.sin(th0)
xT = outerRadius * np.cos(th0)
yT = outerRadius * np.sin(th0)

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
    plt.title( "pta={0:1d}, ptr={1:1d}".format(pta,ptr), fontsize=fst )
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

#Set initial condition for U[0,:,:] (P):

U = np.zeros(( 3, nlv, nth ))
U[0,:,:] = initialCondition(xx,yy)
Po = U[0,1:-1,:].copy()
rhoInv = c**2. / Po

###########################################################################

#Check the max value of the initial condition on the boundaries:

print()
print( 'max value on boundaries =', np.max(np.hstack(( \
initialCondition(xB,yB),                               \
initialCondition(xT,yT) ))) )
print()

###########################################################################

#Hyperviscosity coefficient (alp) for radial direction:

if noRadialHV :
    alp = 0.                                   #remove radial HV completely
else :
    if pol == 1 :
        alp =  2.**-1.  * c
    elif pol == 3 :
        alp = -2.**-5.  * c
    elif pol == 5 :
        alp =  2.**-9.  * c
    elif pol == 7 :
        alp = -2.**-13. * c
    #######################
    elif pol == 2 :
        alp =  2.**-3.  * c
    elif pol == 4 :
        alp = -2.**-7.  * c
    elif pol == 6 :
        alp =  2.**-11. * c
    elif pol == 8 :
        alp = -2.**-15. * c
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
    polA = 8
    stcA = 17

if noAngularHV :
    alpA = 0.                                 #remove angular HV completely
elif polA == 8 :
    alpA = -2.**-15. * c
elif polA == 7 :
    alpA = -2.**-13. * c
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
= common.getTangentsAndNormals( th, stcB, rSurf, dsdr, dsdth, 9.8 )

###########################################################################

#Radial PHS-FD weights arranged in a differentiation matrix:

#Matrix for approximating first derivative in radial direction:
Ws = phs1.getDM( z=s, x=s, m=1, phs=phs, pol=pol, stc=stc )

#Simple (but still correct with dsdr multiplier) radial HV:
Whvs = phs1.getDM( z=s[1:-1], x=s, m=phs-1, phs=phs, pol=pol, stc=stc )
Whvs = alp * ds0**(phs-2) * Whvs                   #scaled radial HV matrix
# dsPol = spdiags( ds**(phs-2), np.array([0]), len(ds), len(ds) )
# Whvs = alp * dsPol.dot(Whvs)

# #Complex radial HV:
# dr = ( rr[2:nlv,:] - rr[0:nlv-2,:] ) / 2.
# alpDrPol = alp * dr**(phs-2)
# alpDrPol = alp * ((outerRadius-innerRadius)/(nlv-2)) ** (phs-2)
# alpDxPol = alp * ( ( dr + ss[1:-1,:]*np.tile(dth,(nlv-2,1)) ) / 2. ) ** (phs-2)
# alpDxPol = alp * ( ( (outerRadius-innerRadius)/(nlv-2) + ss[1:-1,:]*2.*np.pi/nth ) / 2. ) ** (phs-2)
# alpDsPol = alp * ( ( ss[1:-1,:]*np.tile(dth,(nlv-2,1)) + np.transpose(np.tile(ds,(nth,1))) ) / 2. ) ** (phs-2)

###########################################################################

#Angular PHS-FD weights arranged in a differentiation matrix:

#Matrix for approximating first derivative in angular direction:
Wlam = phs1.getPeriodicDM( z=th, x=th, m=1 \
, phs=phsA, pol=polA, stc=stcA, period=2*np.pi )

#Simple (and technically incorrect) angular HV:
Whvlam = phs1.getPeriodicDM( z=th, x=th, m=phsA-1 \
, phs=phsA, pol=polA, stc=stcA, period=2*np.pi )
Whvlam = alpA * dth0**(phsA-2) * Whvlam           #scaled angular HV matrix
# dthPol = spdiags( dth**polA, np.array([0]), len(dth), len(dth) )
# Whvlam = alpA * dthPol.dot(Whvlam)

# #Complex angular HV:
# alpDthPol = alpA * dth0**polA
# # alpDthPol = alpA * ( np.tile(dth,(nlv-2,1)) ) ** polA

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

###########################################################################

#Weights to interpolate from perturbed mesh to regular mesh for plotting:

Wradial = phs1.getDM( z=s0[1:-1], x=s, m=0, phs=phs, pol=pol, stc=stc )

Wangular = phs1.getPeriodicDM( z=th0, x=th, m=0 \
, phs=phsA, pol=polA, stc=stcA, period=2*np.pi )

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
    # return Dr(U) * drdxAll + Dth(U) * dthdxAll

def Dy(U) :                    #du/dy = (du/dr)*(dr/dy) + (du/dth)*(dth/dy)
    return mty * Ds(U) + dthdyAll * Dlam(U)
    # return Dr(U) * drdyAll + Dth(U) * dthdyAll

# def L(U) :
#     # Us = Ws@U
#     # return Ws@Us + ssInv*Us + ssInv2*((U@Wlam)@Wlam)
#     Us = Ws@U
#     Uth = (U@Wlam) + Us*dsdthAll
#     return Ws@(Us*dsdrAll)*dsdrAll + rrInv*Us*dsdrAll \
#     +rrInv2 * ( Uth@Wlam + (Ws@Uth)*dsdthAll )
# 
# def HV(U) :
#     for i in range(np.int(np.round((phs-1)/2))) :
#         U = L(U)
#     return alpDrPol * U[1:-1,:]

def HV(U) :
    # #Angular HV:
    # HVth = ( U @ Wlam ) + dsdthAll * ( Ws @ U )
    # for i in range( polA ) :
        # HVth = ( HVth @ Wlam ) + dsdthAll * ( Ws @ HVth )
    # HVth = alpDthPol * HVth[1:-1,:]
    # #Total HV:
    # return dsdri*(Whvs@U) + HVth
    #Simple (incorrect) method:
    return Whvs.dot(U) + Whvlam.dot(U[1:-1,:].T).T

if mlv == 1 :
    def setGhostNodes(U) :
        return waveEquation.setGhostNodesMidLevels( U \
        , NxBot, NyBot, NxTop, NyTop \
        , TxBot, TyBot, TxTop, TyTop \
        , someFactor, stcB, Wlam \
        , wIinner, wEinner, wDinner, wHinner, wIouter, wEouter, wDouter )
elif mlv == 0 :
    def setGhostNodes(U) :
        return waveEquation.setGhostNodesInterfaces( U \
        , TxBot[0,:], TyBot[0,:], TxTop[0,:], TyTop[0,:] \
        , someFactor, stcB, Wlam \
        , wEinner, wDinner, wEouter, wDouter )
else :
    raise ValueError("Only mlv=0 and mlv=1 are currently supported.")

dUdt = np.zeros( np.shape(U) )
def odefun( t, U, dUdt ) :
    dUdt = waveEquation.odefun( t, U, dUdt \
    , setGhostNodes, Ds, Dlam, HV \
    , Po, rhoInv, dsdthAll, dsdrAll \
    , cosTh, sinTh, cosThOverR, sinThOverR, mtx, mty )
    # dUdt = waveEquation.odefunCartesian( t, U, dUdt \
    # , setGhostNodes, Dx, Dy, HV \
    # , Po, rhoInv )
    return dUdt

q1 = dUdt               #let q1 be another reference to the same array dUdt
q2 = np.zeros( np.shape(U) )          #rk3 and rk4 both need a second array
if rks == 3 :
    def RK( t, U ) :
        t, U = rk.rk3( t, U, odefun, dt, q1, q2 )
        return t, U
elif rks == 4 :
    q3 = np.zeros( np.shape(U) )
    q4 = np.zeros( np.shape(U) )
    def RK( t, U ) :
        t, U = rk.rk4( t, U, odefun, dt, q1, q2, q3, q4 )
        return t, U
else :
    raise ValueError("rks should be 3 or 4 in this problem.  \
    rks=1 and rks=2 are unstable.")

if saveContours :
    fig = plt.figure( figsize = (18,14) )

def plotSomething( U, t ) :
    waveEquation.plotSomething( U, t \
    , Dx, Dy \
    , whatToPlot, xx, yy, xx0, yy0, Wradial, Wangular \
    , c, xB, yB, xT, yT, outerRadius, fig \
    , dynamicColorbar, noInterp )

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange( 0, nTimesteps+1 ) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        print( "t =", np.int(np.round(t)), ",  et =", time.time()-et )
        et = time.time()
        
        if plotFromSaved :
            U = np.load( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy' )
        elif saveArrays :
            np.save( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', setGhostNodes(U) )
        
        if saveContours :
            plotSomething( setGhostNodes(U), t )
        
        if np.max(np.abs(U)) > 5. :
            raise ValueError("Solution greater than 5 in magnitude.  Unstable in time.")
    
    if plotFromSaved :
        t = t + dt
    else :
        t, U = RK( t, U )

###########################################################################
