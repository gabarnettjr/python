import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.annulus import common, waveEquation

###########################################################################

c           = .03                                     #wave speed (c**2=RT)
innerRadius = 1.
outerRadius = 2.
tf          = 100.                                              #final time
saveDel     = 10                           #time interval to save snapshots
exp         = 200.                 #controls steepness of initial condition
amp         = .10                 #amplitude of trigonometric topo function
frq         = 5                   #frequency of trigonometric topo function

plotFromSaved = 0                            #if 1, load instead of compute
saveContours  = 1                       #switch for saving contours as pngs

mlv      = np.int64(sys.argv[1])                #0:interfaces, 1:mid-levels
phs      = np.int64(sys.argv[2])             #PHS RBF exponent (odd number)
pol      = np.int64(sys.argv[3])                         #polynomial degree
stc      = np.int64(sys.argv[4])                              #stencil size
ptb      = np.float64(sys.argv[5])   #random radial perturbation percentage
rkStages = np.int64(sys.argv[6])     #number of Runge-Kutta stages (3 or 4)
ns       = np.int64(sys.argv[7])                  #total number of s levels
dt       = 1./np.float64(sys.argv[8])                              #delta t

rSurf, rSurfPrime \
= common.getTopoFunc( innerRadius, outerRadius, amp, frq )

tmp = 17./18.*np.pi
xc1 = (rSurf(tmp)+outerRadius)/2.*np.cos(tmp)           #x-coord of GA bell
yc1 = (rSurf(tmp)+outerRadius)/2.*np.sin(tmp)           #y-coord of GA bell
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
    return 1. + np.exp( -exp*( (x-xc1)**2. + (y-yc1)**2. ) )

###########################################################################

#Set things up for saving:

saveString = waveEquation.getSavestring( c, innerRadius, outerRadius, tf, saveDel, exp, amp, frq \
, mlv, phs, pol, stc, ptb, rkStages, ns, dt )

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )                       #remove old files

if not os.path.exists( saveString ) :
    os.makedirs( saveString )                         #make new directories

###########################################################################

#Get th and s vectors and save them (randomly perturbed):

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

nth = common.getNth( innerRadius, outerRadius, ns ) #nmbr of angular levels
dth0 = 2.*np.pi / nth                                 #constant delta theta
th0  = np.linspace( 0., 2.*np.pi, nth+1 )             #vector of all angles
th0  = th0[0:-1]                         #remove last angle (same as first)
tmp = ptb * dth0                              #relative perturbation factor
ran = -tmp + 2.*tmp*np.random.rand(nth)         #random perturbation vector
th = th0.copy()                                     #copy regular th vector
th = th + ran                                 #th vector after perturbation

if mlv == 1 :
    ds0  = ( outerRadius - innerRadius ) / (ns-2)         #constant delta s
    s0  = np.linspace( innerRadius-ds0/2, outerRadius+ds0/2, ns ) #s vector
    tmp = ptb * ds0                           #relative perturbation factor
    ran = -tmp + 2.*tmp*np.random.rand(ns)      #random perturbation vector
    s   = s0.copy()                                  #copy regular s vector
    s   = s + ran                              #s vector after perturbation
elif mlv == 0 :
    ds0 = ( outerRadius - innerRadius ) / (ns-3)          #constant delta s
    s0 = np.linspace( innerRadius-ds0, outerRadius+ds0, ns )      #s vector
    tmp = ptb * ds0                           #relative perturbation factor
    ranBot = -tmp + 2.*tmp*np.random.rand(1)
    ranMid = -tmp + 2.*tmp*np.random.rand(ns-4)
    ranTop = -tmp + 2.*tmp*np.random.rand(1)
    s = s0.copy()                                    #copy regular s vector
    s[0] = s[0] + ranBot                         #perturb bottom ghost node
    s[2:-2] = s[2:-2] + ranMid           #perturb interior nodes (no bndry)
    s[-1] = s[-1] + ranTop                          #perturb top ghost node
else :
    sys.exit("\nError: mlv should be 0 or 1.\n")

if plotFromSaved == 1 :
    th = np.load( saveString+'th'+'.npy' )        #load vector of th values
    s = np.load( saveString+'s'+'.npy' )           #load vector of s values
else :
    np.save( saveString+'th'+'.npy', th )         #save vector of th values
    np.save( saveString+'s'+'.npy', s )            #save vector of s values

tmp = np.hstack(( th[-1]-2.*np.pi, th, th[0]+2.*np.pi ))
dth = ( tmp[2:nth+2] - tmp[0:nth] ) / 2.             #non-constant delta th
ds  = ( s[2:ns] - s[0:ns-2] ) / 2.                    #non-constant delta s

###########################################################################

#Get computational mesh and mesh for contour plotting:

thth, ss = np.meshgrid( th, s )      #mesh of perturbed s values and angles
rr = common.getRadii( thth, ss \
, innerRadius, outerRadius, rSurf )                #mesh of perturbed radii
xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates
xxi = xx[1:-1,:]                                      #without ghost layers
yyi = yy[1:-1,:]                                      #without ghost layers
ththi = thth[1:-1,:]                                  #without ghost layers
rri = rr[1:-1,:]                                      #without ghost layers

cosTh = np.cos(ththi)                                                #dr/dx
sinTh = np.sin(ththi)                                                #dr/dy
cosThOverR = cosTh/rri                                              #dth/dy
sinThOverR = sinTh/rri                                             #-dth/dx

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

#Plot the coordinate transformation functions and then exit:

# plt.contourf( xxi, yyi, sFunc(rri,ththi), 20 )
# plt.axis( 'equal' )
# plt.colorbar()
# plt.title( 's' )
# plt.show()

# plt.contourf( xxi, yyi, dsdthi, 20 )
# plt.axis( 'equal' )
# plt.colorbar()
# plt.title( 'ds/dth' )
# plt.show()

# plt.contourf( xxi, yyi, dsdri, 20 )
# plt.axis( 'equal' )
# plt.colorbar()
# plt.title( 'ds/dr' )
# plt.show()

# sys.exit("\nStop here for now.\n")

###########################################################################

#Plot the perturbed radii and then exit:

# fig, ax = plt.subplots( 1, 2, figsize=(8,4) )
# ax[0].plot( s0, s0, '-', s0, s, '.' )       #plot of initial vs perturbed s
# ax[0].set_xlabel('s0')
# ax[0].set_ylabel('s')
# ax[1].plot( s[1:-1], ds, '-' )                #plot of s vs non-constant ds
# ax[1].set_xlabel('s')
# ax[1].set_ylabel('ds')
# plt.show()

# sys.exit("\nStop here for now.\n")

###########################################################################

#Set initial condition for U[0,:,:] (P):

U = np.zeros(( 3, ns, nth ))
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

#Plot the nodes and then exit:

# plt.plot( xx.flatten(), yy.flatten(), "." \
# , xB, yB, "-" \
# , xT, yT, "-" )
# plt.axis('equal')
# tmp = outerRadius + .2
# plt.axis([-tmp,tmp,-tmp,tmp])
# plt.xlabel( 'x' )
# plt.ylabel( 'y' )
# plt.show()

# sys.exit("\nStop here for now.\n")

###########################################################################

#Hyperviscosity coefficient (alp) for radial and angular directions:

if ( pol == 3 ) | ( pol == 4 ) :
    alp = -2.**-10.
elif ( pol == 5 ) | ( pol == 6 ) :
    alp = 2.**-14.
else :
    sys.exit("\nError: pol should be 3, 4, 5, or 6.\n")

if stc == pol+1 :
    alp = 0.                    #remove radial HV if using only polynomials

###########################################################################

#Parameters for angular approximations:

phsA = 9
polA = 7
stcA = 17
alpA = -2.**-18.

###########################################################################

#Extra things needed to enforce the Neumann boundary condition for P:

# stcB = stc
stcB = min( ns-1, stc+4 )
# if ( pol==5 | pol==6 ) & ( ns == 14 ) :
    # stcB = 13
# else :
    # stcB = 2*(pol+2)+1

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

#Complex radial HV:
# dr = ( rr[2:ns,:] - rr[0:ns-2,:] ) / 2.
# alpDrPol = alp * dr**pol
# alpDrPol = alp * ((outerRadius-innerRadius)/(ns-2)) ** pol
# alpDxPol = alp * ( ( dr + ss[1:-1,:]*np.tile(dth,(ns-2,1)) ) / 2. ) ** pol
# alpDxPol = alp * ( ( (outerRadius-innerRadius)/(ns-2) + ss[1:-1,:]*2.*np.pi/nth ) / 2. ) ** pol
# alpDsPol = alp * ( ( ss[1:-1,:]*np.tile(dth,(ns-2,1)) + np.transpose(np.tile(ds,(nth,1))) ) / 2. ) ** pol

###########################################################################

#Angular PHS-FD weights arranged in a differentiation matrix:

Wlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=1 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
Wlam = np.transpose( Wlam )                #work on rows instead of columns

# #Simple (and incorrect) angular HV:
# Whvlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=phsA-1 \
# , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
# Whvlam = alpA * dth0**polA * Whvlam
# # dthPol = spdiags( dth**polA, np.array([0]), len(dth), len(dth) )
# # Whvlam = alpA * dthPol.dot(Whvlam)                #scaled angular HV matrix
# Whvlam = np.transpose( Whvlam )             #work on rows instead of column

#Complex angular HV:
alpDthPol = alpA * dth0**polA
# alpDthPol = alpA * ( np.tile(dth,(ns-2,1)) ) ** polA

###########################################################################

#Weights for interpolation to boundary (I), extrapolation to
#ghost-nodes (E), and d/ds at boundary (D):

wIinner = phs1.getWeights( innerRadius, s[0:stcB],   0, phs, pol+1 )
wEinner = phs1.getWeights( s[0],        s[1:stcB+1], 0, phs, pol+1 )
wDinner = phs1.getWeights( innerRadius, s[0:stcB],   1, phs, pol+1 )

wIouter = phs1.getWeights( outerRadius, s[-1:-(stcB+1):-1], 0, phs, pol+1 )
wEouter = phs1.getWeights( s[-1],       s[-2:-(stcB+2):-1], 0, phs, pol+1 )
wDouter = phs1.getWeights( outerRadius, s[-1:-(stcB+1):-1], 1, phs, pol+1 )

###########################################################################

#Interpolation from perturbed mesh to regular mesh for plotting:

Wradial = phs1.getDM( x=s, X=s0[1:-1], m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Wangular = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th0, m=0 \
, phsDegree=phsA, polyDegree=polA, stencilSize=stcA )

Wangular = np.transpose( Wangular )               #act on rows, not columns

###########################################################################

#Functions to approximate differential operators and other things:
    
def Ds(U) :
    return Ws[1:-1,:] @ U

def Dlam(U) :
    return U[1:-1,:] @ Wlam

# def Dr(U) :
    # return Ds(U) * dsdri

# def Dth(U) :
    # return Dlam(U) + Ds(U) * dsdthi

# def Dx(U) :
    # return cosTh * Dr(U) - sinThOverR * Dth(U)

# def Dy(U) :
    # return sinTh * Dr(U) + cosThOverR * Dth(U)

# def L(U) :
    # # Us = Ws@U
    # # return Ws@Us + ssInv*Us + ssInv2*((U@Wlam)@Wlam)
    # Us = Ws@U
    # Uth = (U@Wlam) + Us*dsdthAll
    # return Ws@(Us*dsdrAll)*dsdrAll + rrInv*Us*dsdrAll \
    # +rrInv2 * ( Uth@Wlam + (Ws@Uth)*dsdthAll )

# def HV(U) :
    # for i in range(np.int(np.round((phs-1)/2))) :
        # U = L(U)
    # return alpDrPol * U[1:-1,:]

def HV(U) :
    #Angular HV:
    HVth = ( U @ Wlam ) + dsdthAll * ( Ws @ U )
    for i in range( polA ) :
        HVth = ( HVth @ Wlam ) + dsdthAll * ( Ws @ HVth )
    HVth = alpDthPol * HVth[1:-1,:]
    #Total HV:
    return dsdri*(Whvs@U) + HVth
    # #Simple (incorrect) method:
    # return ( Whvs @ U ) + ( U[1:-1,:] @ Whvlam )

if mlv == 1 :
    def setGhostNodes( U ) :
        return waveEquation.setGhostNodesMidLevels( U \
        , NxBot, NyBot, NxTop, NyTop \
        , TxBot, TyBot, TxTop, TyTop \
        , someFactor, stcB \
        , Wlam, wIinner, wEinner, wDinner, wIouter, wEouter, wDouter )
else :
    def setGhostNodes( U ) :
        return waveEquation.setGhostNodesInterfaces( U \
        , TxBot[0,:], TyBot[0,:], TxTop[0,:], TyTop[0,:] \
        , someFactor, stcB \
        , Wlam, wEinner, wDinner, wEouter, wDouter )

def odefun( t, U ) :
    return waveEquation.odefun( t, U \
    , setGhostNodes, Ds, Dlam, HV \
    , Po, rhoInv, dsdthi, dsdri \
    , cosTh, sinTh, cosThOverR, sinThOverR )
    # return waveEquation.odefunCartesian( t, U \
    # , setGhostNodes, Dx, Dy, HV          \
    # , Po, rhoInv )

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit("\nError: rkStages should be 3 or 4 in this problem.\n")

###########################################################################

#Main time-stepping loop:

fig = plt.figure( figsize = (18,14) )
et = time.clock()

for i in np.arange( 0, nTimesteps+1 ) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        print( "t =", np.int(np.round(t)), ",  et =", time.clock()-et )
        et = time.clock()
        
        if plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            U = setGhostNodes( U )
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        
        if saveContours == 1 :
            tmp = Wradial @ ( U[0,:,:] ) @ Wangular
            # plt.contourf( xx0, yy0, tmp, 20 )
            plt.contourf( xx0, yy0, tmp, np.arange(1.-.15,1.+.17,.02) )
            plt.plot( xB, yB, "k-", xT, yT, "k-" )
            plt.axis('equal')
            tmp = outerRadius + .2
            plt.axis([-tmp,tmp,-tmp,tmp])
            plt.colorbar()
            fig.savefig( '{0:04d}'.format(np.int(np.round(t)+1e-12))+'.png', bbox_inches = 'tight' )
            plt.clf()
        
        if np.max(np.abs(U)) > 5. :
            sys.exit("\nUnstable in time.\n")
        
    if plotFromSaved == 1 :
        t = t + dt
    else :
        [ t, U ] = rk( t, U, odefun, dt )

###########################################################################