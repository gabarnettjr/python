import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.acousticWaveEquation import annulus

###########################################################################

innerRadius   = 1.
outerRadius   = 2.
rkStages      = 3                    #number of Runge-Kutta stages (3 or 4)
saveDel       = 10                         #time interval to save snapshots
plotFromSaved = 0                            #if 1, load instead of compute

c  = 1./10.                                                     #wave speed
nr = 32+2                                    #total number of radial levels
dt = 1./20.                                                        #delta t
tf = 100.                                                       #final time

phs = 7                                      #PHS RBF exponent (odd number)
pol = 5                                                  #polynomial degree
stc = 15                                                      #stencil size
ptb = .30                            #random radial perturbation percentage

xc1 = 0.                                                #x-coord of GA bell
yc1 = ( innerRadius + outerRadius ) / 2.                #y-coord of GA bell
def initialCondition( x, y ) :
    return np.exp( -20.*( (x-xc1)**2. + (y-yc1)**2. ) )

###########################################################################

saveString = './shortResults/'    \
+ 'nr'   + '{0:1d}'.format(nr-2)  \
+ '_phs' + '{0:1d}'.format(phs)   \
+ '_pol' + '{0:1d}'.format(pol)   \
+ '_stc' + '{0:1d}'.format(stc)   \
+ '_ptb' + '{0:1.2f}'.format(ptb) \
+ '/'

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )

if not os.path.exists( saveString ) :
    os.makedirs( saveString )

###########################################################################

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

nth = np.int(np.round(  \
2*np.pi * innerRadius * \
(nr-2)/(outerRadius-innerRadius) ))               #number of angular levels
dth = 2.*np.pi / nth                                  #constant delta theta
th = np.linspace( 0., 2.*np.pi, nth+1 )               #vector of all angles
th = th[0:-1]                            #remove last angle (same as first)

dr = ( outerRadius - innerRadius ) / (nr-2)               #constant delta r
r0 = np.linspace( innerRadius-dr/2, outerRadius+dr/2, nr )   #radius vector
ptb = ptb * dr
ran = -ptb + 2*ptb*np.random.rand(len(r0))             #perturbation vector
r = np.linspace( innerRadius-dr/2, outerRadius+dr/2, nr )
r = r + ran                               #radius vector after perturbation
dr = (r[2:len(r)]-r[0:len(r)-2])/2.                   #non-constant delta r

thth, rr = np.meshgrid( th, r )                   #mesh of radii and angles
xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates

thth0, rr0 = np.meshgrid( th, r0[1:-1] )         #regular mesh for plotting
xx0 = rr0 * np.cos(thth0)
yy0 = rr0 * np.sin(thth0)

###########################################################################

#Plot showing how much the radii have been perturbed:

fig, ax = plt.subplots( 1, 2, figsize=(10,5) )
ax[0].plot( r0, r0, '-', r0, r, '.' )   #plot of initial vs perturbed radii
plt.xlabel('r0')
plt.ylabel('r')
ax[1].plot( r[1:-1], dr, '-' )                #plot of r vs non-constant dr
plt.xlabel('r')
plt.ylabel('dr')
plt.show()

###########################################################################

#Set initial condition and fixed Dirichlet BC for U[0,:,:] (rho):

U = np.zeros(( 3, nr, nth ))
U[0,:,:] = initialCondition( xx, yy )

xB = innerRadius * np.cos(th)
yB = innerRadius * np.sin(th)
rhoB = initialCondition( xB, yB )

xT = outerRadius * np.cos(th)
yT = outerRadius * np.sin(th)
rhoT = initialCondition( xT, yT )

###########################################################################

#Plot the nodes:

# plt.plot( xx.flatten(), yy.flatten(), "." \
# , xB, yB, "-" \
# , xT, yT, "-" )
# plt.axis('equal')
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

#Radial hyperviscosity coefficient (alp):

if pol == 1 :
    alp = 2^-9000
elif pol == 3 :
    alp = -2.**-10.
elif pol == 5 :
    alp = 2**-12
elif pol == 7 :
    alp = -2**-14
else :
    alp = 0.

###########################################################################

#Radial weights arranged in a differentiation matrix:

Wr   = phs1.getDM( x=r, X=r[1:-1], m=1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvr = phs1.getDM( x=r, X=r[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

###########################################################################

#Modify the radial HV matrix to take into account the varying radii:

drPol = spdiags( dr**pol, np.array([0]), len(dr), len(dr) )
Whvr = alp * drPol.dot(Whvr)

###########################################################################

#Angular FD weights arranged in a differentiation matrix:

Wth   = phs1.getPeriodicDM( period=2*np.pi, X=th, m=1  \
, phsDegree=3,  polyDegree=10, stencilSize=11 )

Whvth = phs1.getPeriodicDM( period=2*np.pi, X=th, m=10 \
, phsDegree=11, polyDegree=10, stencilSize=11 )

###########################################################################

#Weights for interpolation to boundary and extrapolation to ghost-nodes:

wIinner = phs1.getWeights( innerRadius, r[0:stc],   0, phs, pol )
wEinner = phs1.getWeights( r[0],        r[1:stc+1], 0, phs, pol )

wIouter = phs1.getWeights( outerRadius, r[-1:-stc-1:-1], 0, phs, pol )
wEouter = phs1.getWeights( r[-1],       r[-2:-stc-2:-1], 0, phs, pol )

###########################################################################

#Interpolation from perturbed nodes to regular nodes for plotting:

W = phs1.getDM( x=r, X=r0[1:-1], m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

###########################################################################

#Functions to approximate differential operators and other things:

def Dr( U ) :
    return Wr.dot( U )

def Dth( U ) :
    return np.transpose( Wth.dot( np.transpose(U) ) )

def HVr( U ) :
    return Whvr.dot( U )

def HVth( U ) :
    np.transpose( Whvth.dot( np.transpose(U) ) )

def setGhostNodes( U ) :
    return annulus.setGhostNodes( U \
    , rhoB, rhoT, wIinner, wEinner, wIouter, wEouter )

def odefun( t, U ) :
    return annulus.odefun( t, U \
    , setGhostNodes, Dr, Dth, HVr, HVth \
    , thth[1:-1,:], rr[1:-1,:], c )

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
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        
        # Uplot = np.zeros(( 3, nr-2, nth ))
        # for k in range(3) :
            # Uplot[k,:,:] = W.dot(U[k,:,:])
        
        plt.contourf( xx0, yy0, W.dot(U[0,:,:]), np.arange(-.255,.255+.01,.01) )
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