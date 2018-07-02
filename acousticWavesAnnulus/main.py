#!/usr/bin/python3
import sys
import os
import time
import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.annulus import common, waveEquation

###########################################################################

args = waveEquation.parseInput()
#get rid of the args prefix on all the variable names:
d = vars(args)
for k in d.keys() :
    exec("{} = args.{}".format(k,k))

dt = 1./dti

if plotFromSaved :
    saveContours = True

###########################################################################

rSurf, rSurfPrime \
= common.getTopoFunc( innerRadius, outerRadius, amp, frq )

tmp = 17./18.*np.pi
xc1 = (rSurf(tmp)+outerRadius)/2.*np.cos(tmp)                #x-coord of IC
yc1 = (rSurf(tmp)+outerRadius)/2.*np.sin(tmp)                #y-coord of IC
tmp = 5./18.*np.pi
xc2 = (rSurf(tmp)+outerRadius)/2.*np.cos(tmp)                #x-coord of IC
yc2 = (rSurf(tmp)+outerRadius)/2.*np.sin(tmp)                #y-coord of IC
def initialCondition( x, y ) :
    # #Wendland function:
    # r = np.sqrt( 6 * ( (x-xc1)**2. + (y-yc1)**2. ) )
    # ind = r<1.
    # z = np.zeros( np.shape(x) )
    # z[ind] = ( 1. - r[ind] ) ** 10. * ( 429.*r[ind]**4. + 450.*r[ind]**3. \
    # + 210.*r[ind]**2. + 50.*r[ind] + 5.  )
    # z = 1. + z/5.
    # return z
    #Gaussian:
    return 1. + np.exp( -exp*( (x-xc1)**2. + (y-yc1)**2. ) ) \
              + np.exp( -exp*( (x-xc2)**2. + (y-yc2)**2. ) )

###########################################################################

#Set things up for saving:

saveString = waveEquation.getSavestring( c, innerRadius, outerRadius \
, tf, saveDel, exp, amp, frq \
, mlv, phs, pol, stc, ptb, rks, nlv, dti )

if ( saveArrays ) & ( not plotFromSaved ) :
    if os.path.exists( saveString + '*.npy' ) :
        os.remove( saveString + '*.npy' )                 #remove old files
    if not os.path.exists( saveString ) :
        os.makedirs( saveString )                     #make new directories

###########################################################################

#Get th and s vectors and save them (randomly perturbed):

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

nth = common.getNth( innerRadius, outerRadius, nlv ) #nmbr of angular levels
dth0 = 2.*np.pi / nth                                 #constant delta theta
th0  = np.linspace( 0., 2.*np.pi, nth+1 )             #vector of all angles
th0  = th0[0:-1]                         #remove last angle (same as first)
tmp = (ptb/100.) * dth0                       #relative perturbation factor
ran = -tmp + 2.*tmp*np.random.rand(nth)         #random perturbation vector
th = th0.copy()                                     #copy regular th vector
th = th + ran                                 #th vector after perturbation

if mlv == 1 :
    ds0 = ( outerRadius - innerRadius ) / (nlv-2)          #constant delta s
    s0  = np.linspace( innerRadius-ds0/2, outerRadius+ds0/2, nlv ) #s vector
    tmp = (ptb/100.) * ds0                     #relative perturbation factor
    ran = -tmp + 2.*tmp*np.random.rand(nlv)      #random perturbation vector
    s   = s0.copy()                                  #copy regular s vector
    s   = s + ran                              #s vector after perturbation
elif mlv == 0 :
    ds0 = ( outerRadius - innerRadius ) / (nlv-3)          #constant delta s
    s0  = np.linspace( innerRadius-ds0, outerRadius+ds0, nlv )     #s vector
    tmp = (ptb/100.) * ds0                     #relative perturbation factor
    ranBot = -tmp + 2.*tmp*np.random.rand(1)
    ranMid = -tmp + 2.*tmp*np.random.rand(nlv-4)
    ranTop = -tmp + 2.*tmp*np.random.rand(1)
    s = s0.copy()                                    #copy regular s vector
    s[0] = s[0] + ranBot                         #perturb bottom ghost node
    s[2:-2] = s[2:-2] + ranMid           #perturb interior nodes (no bndry)
    s[-1] = s[-1] + ranTop                          #perturb top ghost node
else :
    sys.exit("\nError: mlv should be 0 or 1.\n")

if plotFromSaved :
    th = np.load( saveString + 'th' + '.npy' )    #load vector of th values
    s  = np.load( saveString + 's'  + '.npy' )     #load vector of s values
else :
    if saveArrays :
        np.save( saveString + 'th' + '.npy', th ) #save vector of th values
        np.save( saveString + 's'  + '.npy', s )   #save vector of s values

tmp = np.hstack(( th[-1]-2.*np.pi, th, th[0]+2.*np.pi ))
dth = ( tmp[2:nth+2] - tmp[0:nth] ) / 2.             #non-constant delta th
ds  = ( s[2:nlv] - s[0:nlv-2] ) / 2.                    #non-constant delta s

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

# #Plot the coordinate transformation functions and then exit:
# 
# plt.contourf( xxi, yyi, sFunc(rri,ththi), 20 )
# plt.axis( 'equal' )
# plt.colorbar()
# plt.title( 's' )
# plt.show()
# 
# plt.contourf( xxi, yyi, dsdthi, 20 )
# plt.axis( 'equal' )
# plt.colorbar()
# plt.title( 'ds/dth' )
# plt.show()
# 
# plt.contourf( xxi, yyi, dsdri, 20 )
# plt.axis( 'equal' )
# plt.colorbar()
# plt.title( 'ds/dr' )
# plt.show()
# 
# sys.exit("\nStop here for now.\n")

###########################################################################

# #Plot the perturbed radii and then exit:
# 
# fig, ax = plt.subplots( 1, 2, figsize=(8,4) )
# ax[0].plot( s0, s0, '-', s0, s, '.' )       #plot of initial vs perturbed s
# ax[0].set_xlabel('s0')
# ax[0].set_ylabel('s')
# ax[1].plot( s[1:-1], ds, '-' )                #plot of s vs non-constant ds
# ax[1].set_xlabel('s')
# ax[1].set_ylabel('ds')
# plt.show()
# 
# sys.exit("\nStop here for now.\n")

###########################################################################

#Set initial condition for U[0,:,:] (P):

U = np.zeros(( 3, nlv, nth ))
U[0,:,:] = initialCondition(xx,yy)
Po = U[0,1:-1,:]
rhoInv = c**2. / Po

###########################################################################

#Check the value of the initial condition on boundaries:

xB = rSurf(th0) * np.cos(th0)
yB = rSurf(th0) * np.sin(th0)
xT = outerRadius * np.cos(th0)
yT = outerRadius * np.sin(th0)
print()
print( 'max value on boundaries =', np.max(np.hstack(( \
initialCondition(xB,yB),                             \
initialCondition(xT,yT) ))) )
print()

###########################################################################

# #Plot the nodes and then exit:
# 
# plt.plot( xx.flatten(), yy.flatten(), "." \
# , xB, yB, "-" \
# , xT, yT, "-" )
# plt.axis('equal')
# tmp = outerRadius + .2
# plt.axis([-tmp,tmp,-tmp,tmp])
# plt.xlabel( 'x' )
# plt.ylabel( 'y' )
# plt.show()
# 
# sys.exit("\nStop here for now.\n")

###########################################################################

#Hyperviscosity coefficient (alp) for radial direction:
if ( pol == 1 ) | ( pol == 2 ) :
    alp = 2.**-1. * c
elif ( pol == 3 ) | ( pol == 4 ) :
    alp = -2.**-5. * c
elif ( pol == 5 ) | ( pol == 6 ) :
    alp = 2.**-9. * c
elif ( pol == 7 ) | ( pol == 8 ) :
    alp = -2.**-13. * c
else :
    sys.exit("\nError: 1 <= pol <= 8\n")

if uhv != 1 :
    alp = 0.                                   #remove radial HV completely

###########################################################################

#Parameters for angular approximations:

phsA = 9
polA = 7
stcA = 17
alpA = -2.**-13. * c

###########################################################################

#Extra things needed to enforce the Neumann boundary condition for P:

stcB = stc
# stcB = min( nlv-1, 2*(pol+2)+1 )

NxBot, NyBot, NxTop, NyTop \
, TxBot, TyBot, TxTop, TyTop \
, someFactor \
= common.getTangentsAndNormals( th, stcB, rSurf, dsdr, dsdth )

###########################################################################

#Radial PHS-FD weights arranged in a differentiation matrix:

Ws = phs1.getDM( x=s, X=s, m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

#Simple (but still correct with dsdr multiplier) radial HV:
Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )
Whvs = alp * ds0**pol * Whvs
# dsPol = spdiags( ds**pol, np.array([0]), len(ds), len(ds) )
# Whvs = alp * dsPol.dot(Whvs)                       #scaled radial HV matrix

# #Complex radial HV:
# dr = ( rr[2:nlv,:] - rr[0:nlv-2,:] ) / 2.
# alpDrPol = alp * dr**pol
# alpDrPol = alp * ((outerRadius-innerRadius)/(nlv-2)) ** pol
# alpDxPol = alp * ( ( dr + ss[1:-1,:]*np.tile(dth,(nlv-2,1)) ) / 2. ) ** pol
# alpDxPol = alp * ( ( (outerRadius-innerRadius)/(nlv-2) + ss[1:-1,:]*2.*np.pi/nth ) / 2. ) ** pol
# alpDsPol = alp * ( ( ss[1:-1,:]*np.tile(dth,(nlv-2,1)) + np.transpose(np.tile(ds,(nth,1))) ) / 2. ) ** pol

###########################################################################

#Angular PHS-FD weights arranged in a differentiation matrix:

Wlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=1 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )

#Simple (and incorrect) angular HV:
Whvlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=phsA-1 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
Whvlam = alpA * dth0**polA * Whvlam
# dthPol = spdiags( dth**polA, np.array([0]), len(dth), len(dth) )
# Whvlam = alpA * dthPol.dot(Whvlam)                #scaled angular HV matrix

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

wIouter = phs1.getWeights( outerRadius, s[-1:-(stcB+1):-1], 0, phs, pol )
wEouter = phs1.getWeights( s[-1],       s[-2:-(stcB+2):-1], 0, phs, pol )
wDouter = phs1.getWeights( outerRadius, s[-1:-(stcB+1):-1], 1, phs, pol )

###########################################################################

#Interpolation from perturbed mesh to regular mesh for plotting:

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

def Dr(U) :                                        #du/dr = (du/ds)*(ds/dr)
    return Ds(U) * dsdrAll

def Dth(U) :                           #du/dth = du/dlam + (du/ds)*(ds/dth)
    return Dlam(U) + Ds(U) * dsdthAll

def Dx(U) :                    #du/dx = (du/dr)*(dr/dx) + (du/dth)*(dth/dx)
    return Dr(U) * drdxAll + Dth(U) * dthdxAll

def Dy(U) :                    #du/dy = (du/dr)*(dr/dy) + (du/dth)*(dth/dy)
    return Dr(U) * drdyAll + Dth(U) * dthdyAll

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
    def setGhostNodes( U ) :
        return waveEquation.setGhostNodesMidLevels( U \
        , NxBot, NyBot, NxTop, NyTop \
        , TxBot, TyBot, TxTop, TyTop \
        , someFactor, stcB, Wlam \
        , wIinner, wEinner, wDinner, wHinner, wIouter, wEouter, wDouter )
else :
    def setGhostNodes( U ) :
        return waveEquation.setGhostNodesInterfaces( U \
        , TxBot[0,:], TyBot[0,:], TxTop[0,:], TyTop[0,:] \
        , someFactor, stcB \
        , Wlam, wEinner, wDinner, wEouter, wDouter )

def odefun( t, U ) :
    return waveEquation.odefun( t, U \
    , setGhostNodes, Ds, Dlam, HV \
    , Po, rhoInv, dsdthAll, dsdrAll \
    , cosTh, sinTh, cosThOverR, sinThOverR )
    # return waveEquation.odefunCartesian( t, U \
    # , setGhostNodes, Dx, Dy, HV          \
    # , Po, rhoInv )

if rks == 3 :
    rk = rk.rk3
elif rks == 4 :
    rk = rk.rk4
else :
    sys.exit("\nError: rks should be 3 or 4 in this problem.\n")

###########################################################################

#Main time-stepping loop:

if saveContours :
    fig = plt.figure( figsize = (18,14) )
et = time.time()

for i in np.arange( 0, nTimesteps+1 ) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        print( "t =", np.int(np.round(t)), ",  et =", time.time()-et )
        et = time.time()
        
        if plotFromSaved :
            U = np.load( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy' )
        else :
            if saveArrays :
                U = setGhostNodes( U )
                np.save( saveString \
                + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U )
        
        if saveContours :
            waveEquation.plotSomething( U, t \
            , Dx, Dy \
            , whatToPlot, xx0, yy0, Wradial, Wangular \
            , c, xB, yB, xT, yT, outerRadius, fig )
        
        if np.max(np.abs(U)) > 5. :
            sys.exit("\nUnstable in time.\n")
        
    if plotFromSaved :
        t = t + dt
    else :
        [ t, U ] = rk( t, U, odefun, dt )

###########################################################################
