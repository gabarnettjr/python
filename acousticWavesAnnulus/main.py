import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

sys.path.append( '../site-packages' )

from gab import rk, phs1, phs2
from gab.annulus import common, waveEquation

###########################################################################

c           = .01                                     #wave speed (c**2=RT)
innerRadius = 1.
outerRadius = 2.
tf          = 100.                                              #final time
saveDel     = 10                           #time interval to save snapshots
exp         = 100.                 #controls steepness of initial condition
amp         = .10                 #amplitude of trigonometric topo function
frq         = 9                   #frequency of trigonometric topo function

plotFromSaved = 0                            #if 1, load instead of compute
saveContours  = 0                       #switch for saving contours as pngs

dimSplit = np.int64(sys.argv[1])               #0:none, 1:some, 2:fullSplit
phs      = np.int64(sys.argv[2])             #PHS RBF exponent (odd number)
pol      = np.int64(sys.argv[3])                         #polynomial degree
stc      = np.int64(sys.argv[4])                              #stencil size
ptb      = np.float64(sys.argv[5])   #random radial perturbation percentage
rkStages = np.int64(sys.argv[6])     #number of Runge-Kutta stages (3 or 4)
ns       = np.int64(sys.argv[7])+2                #total number of s levels
dt       = 1./np.float64(sys.argv[8])                              #delta t

rSurf, rSurfPrime \
= common.getTopoFunc( innerRadius, outerRadius, amp, frq, phs, pol, stc )
sFunc, dsdth, dsdr \
= common.getHeightCoordinate( innerRadius, outerRadius, rSurf, rSurfPrime )

tmp = 17./18.*np.pi
xc1 = (rSurf(tmp)+outerRadius)/2.*np.cos(tmp)           #x-coord of GA bell
yc1 = (rSurf(tmp)+outerRadius)/2.*np.sin(tmp)           #y-coord of GA bell
def initialCondition( x, y ) :
    return 1. + np.exp( -exp*( (x-xc1)**2. + (y-yc1)**2. ) )

###########################################################################

#Set things up for saving:

saveString = waveEquation.getSavestring( c, innerRadius, outerRadius, tf, saveDel, exp, amp, frq \
, dimSplit, phs, pol, stc, ptb, rkStages, ns, dt )

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )                       #remove old files

if not os.path.exists( saveString ) :
    os.makedirs( saveString )                         #make new directories

###########################################################################

#Get th and s vectors and save them (randomly perturbed):

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

nth = common.getNth( innerRadius, outerRadius, ns ) #nmbr of angular levels
dth = 2.*np.pi / nth                                  #constant delta theta
th0  = np.linspace( 0., 2.*np.pi, nth+1 )             #vector of all angles
th0  = th0[0:-1]                         #remove last angle (same as first)
tmp = ptb * dth                               #relative perturbation factor
ran = -tmp + 2.*tmp*np.random.rand(nth)         #random perturbation vector
th = th0.copy()                                     #copy regular th vector
th = th + ran                                 #th vector after perturbation

ds  = ( outerRadius - innerRadius ) / (ns-2)              #constant delta s
s0  = np.linspace( innerRadius-ds/2, outerRadius+ds/2, ns )       #s vector
tmp = ptb * ds                                #relative perturbation factor
ran = -tmp + 2.*tmp*np.random.rand(ns)          #random perturbation vector
s   = s0.copy()                                      #copy regular s vector
s   = s + ran                                  #s vector after perturbation

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
dsdthi = dsdth( rri, ththi )             #interior values of dsdth function
dsdri  = dsdr( rri, ththi )               #interior values of dsdr function

cosTh = np.cos(ththi)
sinTh = np.sin(ththi)
cosThOverR = cosTh/rri
sinThOverR = sinTh/rri

thth0, ss0 = np.meshgrid( th0, s0[1:-1] )        #regular mesh for plotting
rr0 = common.getRadii( thth0, ss0 \
, innerRadius, outerRadius, rSurf )                  #mesh of regular radii
xx0 = rr0 * np.cos(thth0)                         #mesh of regular x-coords
yy0 = rr0 * np.sin(thth0)                         #mesh of regular x-coords

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

#Extra things needed to enforce the Neumann boundary condition for P:

NxBot, NyBot, NxTop, NyTop \
, TxBot, TyBot, TxTop, TyTop, someFactor \
= common.getTangentsAndNormals( th, stc, rSurf, dsdr, dsdth )

###########################################################################

#Check the value of the initial condition on inner boundary:

xB = rSurf(th) * np.cos(th)
yB = rSurf(th) * np.sin(th)
xT = outerRadius * np.cos(th)
yT = outerRadius * np.sin(th)
print()
print( 'max value on inner boundary =', np.max(initialCondition(xB,yB)) )
print()

###########################################################################

#Plot the nodes and then exit:

# plt.plot( xx.flatten(), yy.flatten(), "." \
# , xB, yB, "-" \
# , xT, yT, "-" )
# plt.axis('equal')
# plt.xlabel( 'x' )
# plt.ylabel( 'y' )
# plt.show()

# sys.exit("\nStop here for now.\n")

###########################################################################

#Hyperviscosity coefficient (alp) for radial and angular directions:

if ( pol == 3 ) | ( pol == 4 ) :
    alp = -2.**-10.
elif ( pol == 5 ) | ( pol == 6 ) :
    alp = 2.**-13.
else :
    sys.exit("\nError: pol should be 3, 4, 5, or 6.\n")

if stc == pol+1 :
    alp = 0.                           #remove HV if using only polynomials

###########################################################################

if dimSplit != 2 :
    
    if plotFromSaved != 1 :
    
        #Get fully 2D Cartesian DMs:
        
        stencils = phs2.getStencils( xx.flatten(), yy.flatten() \
        , xxi.flatten(), yyi.flatten(), stc )
        
        if dimSplit == 1 :
            A = phs2.getAmatrices( stencils, phs, pol )
            Wx = phs2.getWeights( stencils, A, "1",  0 )
            Wy = phs2.getWeights( stencils, A, "2",  0 )
        elif dimSplit == 0 :
            e1 = np.transpose( np.vstack(( -yyi.flatten(), xxi.flatten() )) )
            nm = np.sqrt( e1[:,0]**2. + e1[:,1]**2. )
            e1 = e1 / np.transpose(np.tile(nm,(2,1)))
            e2 = np.transpose( np.vstack((  xxi.flatten(), yyi.flatten() )) )
            nm = np.sqrt( e2[:,0]**2. + e2[:,1]**2. )
            e2 = e2 / np.transpose(np.tile(nm,(2,1)))
            stencils = phs2.rotateStencils( stencils, e1, e2 )
            A = phs2.getAmatrices( stencils, phs, pol )
            Wx1 = phs2.getWeights( stencils, A, "1", 0 )
            Wx2 = phs2.getWeights( stencils, A, "2", 0 )
            Wx = Wx1 * stencils.dx1dx + Wx2 * stencils.dx2dx
            Wy = Wx1 * stencils.dx1dy + Wx2 * stencils.dx2dy
        
        Wth = np.transpose(np.tile(-yyi.flatten(),(stc,1))) * Wx \
            + np.transpose(np.tile( xxi.flatten(),(stc,1))) * Wy
        
        # K = np.int( np.round( (phs-1)/2 ) )
        # Whv = phs2.getWeights( stencils, A, "hv", K )
        # Whv = alp * stencils.h**(2*K-1) * Whv
    
    if pol == 3 :
        stc = 7
    elif pol == 5 :
        stc = 13
    else :
        sys.exit("\nOnly using pol=3 and pol=5 in this case.\n")

###########################################################################

#Radial FD weights arranged in a differentiation matrix:

Ws   = phs1.getDM( x=s, X=s[1:-1], m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

dsPol = spdiags( ds**pol, np.array([0]), len(ds), len(ds) )
Whvs = alp * dsPol.dot(Whvs)                       #scaled radial HV matrix

###########################################################################

#Angular FD weights arranged in a differentiation matrix:

Wlam   = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvlam = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th, m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

dthPol = spdiags( dth**pol, np.array([0]), len(dth), len(dth) )
Whvlam = alp * dthPol.dot(Whvlam)                 #scaled angular HV matrix

Wlam = np.transpose( Wlam )                #work on rows instead of columns
Whvlam = np.transpose( Whvlam )            #work on rows instead of columns

###########################################################################

#Weights for interpolation to boundary, extrapolation to ghost-nodes,
#and d/ds at boundary:

wIinner = phs1.getWeights( innerRadius, s[0:stc],   0, phs, pol )
wEinner = phs1.getWeights( s[0],        s[1:stc+1], 0, phs, pol )
wDinner = phs1.getWeights( innerRadius, s[0:stc],   1, phs, pol )

wIouter = phs1.getWeights( outerRadius, s[-1:-(stc+1):-1], 0, phs, pol )
wEouter = phs1.getWeights( s[-1],       s[-2:-(stc+2):-1], 0, phs, pol )
wDouter = phs1.getWeights( outerRadius, s[-1:-(stc+1):-1], 1, phs, pol )

wIinner = np.transpose( np.tile( wIinner, (nth,1) ) )
wEinner = np.transpose( np.tile( wEinner, (nth,1) ) )
wDinner = np.transpose( np.tile( wDinner, (nth,1) ) )
wIouter = np.transpose( np.tile( wIouter, (nth,1) ) )
wEouter = np.transpose( np.tile( wEouter, (nth,1) ) )
wDouter = np.transpose( np.tile( wDouter, (nth,1) ) )

###########################################################################

#Interpolation from perturbed mesh to regular mesh for plotting:

Wradial = phs1.getDM( x=s, X=s0[1:-1], m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Wangular = phs1.getPeriodicDM( period=2*np.pi, x=th, X=th0, m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Wangular = np.transpose( Wangular )               #act on rows, not columns

###########################################################################

#Functions to approximate differential operators and other things:

if dimSplit == 2 :
    
    def Ds(U) :
        return Ws @ U
    
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
    
    def HV(U) :
        return ( Whvs @ U ) + ( U[1:-1,:] @ Whvlam )
    
elif ( dimSplit == 1 ) | ( dimSplit == 0 ) :
    
    def Ds(U) :
        return Ws @ U
    
    def Dr(U) :
        return Ds(U) * dsdri
    
    def Dth(U) :
        U = U.flatten()
        U = np.sum( Wth*U[stencils.idx], axis=1 )
        return np.reshape( U, (ns-2,nth) )
    
    def Dx(U) :
        return cosTh * Dr(U) - sinThOverR * Dth(U)
    
    def Dy(U) :
        return sinTh * Dr(U) + cosThOverR * Dth(U)
    
    def HV(U) :
        return ( Whvs @ U ) + ( U[1:-1,:] @ Whvlam )

# elif dimSplit == 0 :
    
    # def Dx(U) :
        # U = U.flatten()
        # U = np.sum( Wx*U[stencils.idx], axis=1 )
        # return np.reshape( U, (ns-2,nth) )
    
    # def Dy(U) :
        # U = U.flatten()
        # U = np.sum( Wy*U[stencils.idx], axis=1 )
        # return np.reshape( U, (ns-2,nth) )
    
    # def HV(U) :
        # U = U.flatten()
        # U = np.sum( Whv*U[stencils.idx], axis=1 )
        # return np.reshape( U, (ns-2,nth) )

else :
    
    sys.exit("\nError: dimSplit should be 0, 1, or 2.\n")

def setGhostNodes( U ) :
    return waveEquation.setGhostNodesNeumann( U \
    , NxBot, NyBot, NxTop, NyTop \
    , TxBot, TyBot, TxTop, TyTop \
    , someFactor, stc \
    , Wlam, wIinner, wEinner, wDinner, wIouter, wEouter, wDouter )
    # return waveEquation.setGhostNodes( U \
    # , rhoB, rhoT, wIinner, wEinner, wIouter, wEouter, stc )

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
            plt.contourf( xx0, yy0, tmp, np.arange(1.-.19,1.+.21,.02) )
            plt.axis('equal')
            plt.colorbar()
            fig.savefig( '{0:04d}'.format(np.int(np.round(t)+1e-12))+'.png', bbox_inches = 'tight' )
            plt.clf()
        
        if np.max(np.abs(U)) > 10. :
            sys.exit("\nUnstable in time.\n")
        
    if plotFromSaved == 1 :
        t = t + dt
    else :
        [ t, U ] = rk( t, U, odefun, dt )

###########################################################################