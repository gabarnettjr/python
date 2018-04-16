import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.acousticWaveEquation import annulus
from gab.pseudospectral import periodic

###########################################################################

innerRadius   = 1.
outerRadius   = 2.
tf            = 100.                                            #final time

rkStages      = 3                    #number of Runge-Kutta stages (3 or 4)
saveDel       = 10                         #time interval to save snapshots
plotFromSaved = 0                            #if 1, load instead of compute

c  = 1./10.                                                     #wave speed
ns = 128+2                                        #total number of s levels
dt = 1./48.                                                        #delta t

phs = 7                                      #PHS RBF exponent (odd number)
pol = 5                                                  #polynomial degree
stc = 13                                                      #stencil size
ptb = .30                            #random radial perturbation percentage

xc1 = 0.                                                #x-coord of GA bell
yc1 = ( innerRadius + outerRadius ) / 2.                #y-coord of GA bell
def initialCondition( x, y ) :
    return np.exp( -50.*( (x-xc1)**2. + (y-yc1)**2. ) )

k = innerRadius
amp = .02
def rSurf( th ) :
    return innerRadius
    # return innerRadius + amp*np.sin(k*th)
def rSurfPrime( th ) :
    return 0.
    # return amp*k*np.cos(k*th)

###########################################################################

saveString = './results/'    \
+ 'ns'   + '{0:1d}'.format(ns-2)  \
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

nth = annulus.getNth( innerRadius, outerRadius, ns )#nmbr of angular levels
dth = 2.*np.pi / nth                                  #constant delta theta
th = np.linspace( 0., 2.*np.pi, nth+1 )               #vector of all angles
th = th[0:-1]                            #remove last angle (same as first)

ds = ( outerRadius - innerRadius ) / (ns-2)               #constant delta s
s0 = np.linspace( innerRadius-ds/2, outerRadius+ds/2, ns )        #s vector
ptb = ptb * ds
ran = -ptb + 2*ptb*np.random.rand(len(s0))             #perturbation vector
s = np.linspace( innerRadius-ds/2, outerRadius+ds/2, ns )
s = s + ran                                    #s vector after perturbation
ds = (s[2:len(s)]-s[0:len(s)-2])/2.                   #non-constant delta s

np.save( saveString+'s'+'.npy', s )

thth, ss = np.meshgrid( th, s )                   #mesh of radii and angles
xx = ss * np.cos(thth)                               #mesh of x-coordinates
yy = ss * np.sin(thth)                               #mesh of y-coordinates

thth0, ss0 = np.meshgrid( th, s0[1:-1] )         #regular mesh for plotting
xx0 = ss0 * np.cos(thth0)
yy0 = ss0 * np.sin(thth0)

###########################################################################

#Plot showing how much the radii have been perturbed:

# fig, ax = plt.subplots( 1, 2, figsize=(10,5) )
# ax[0].plot( s0, s0, '-', s0, s, '.' )       #plot of initial vs perturbed s
# plt.xlabel('s0')
# plt.ylabel('s')
# ax[1].plot( s[1:-1], ds, '-' )                #plot of s vs non-constant ds
# plt.xlabel('s')
# plt.ylabel('ds')
# plt.show()

###########################################################################

#Set initial condition and fixed Dirichlet BC for U[0,:,:] (rho):

U = np.zeros(( 3, ns, nth ))
U[0,:,:] = initialCondition( xx, yy )

xB = innerRadius * np.cos(th)
yB = innerRadius * np.sin(th)
rhoB = initialCondition( xB, yB )

xT = outerRadius * np.cos(th)
yT = outerRadius * np.sin(th)
rhoT = initialCondition( xT, yT )

###########################################################################

#Plot the nodes and exit:

# plt.plot( xx.flatten(), yy.flatten(), "." \
# , xB, yB, "-" \
# , xT, yT, "-" )
# plt.axis('equal')
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

#Radial hyperviscosity coefficient (alp):

if ( pol == 1 ) | ( pol == 2 ) :
    alp = 2^-9000
elif ( pol == 3 ) | ( pol == 4 ) :
    alp = -2.**-10.
elif ( pol == 5 ) | ( pol == 6 ) :
    alp = 2.**-12.
elif ( pol == 7 ) | ( pol == 8 ) :
    alp = -2**-14

if stc == pol+1 :
    alp = 0.

###########################################################################

#Radial weights arranged in a differentiation matrix:

Ws   = phs1.getDM( x=s, X=s[1:-1], m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

###########################################################################

#Modify the radial HV matrix to take into account the varying radii:

dsPol = spdiags( ds**pol, np.array([0]), len(ds), len(ds) )
Whvs = alp * dsPol.dot(Whvs)

###########################################################################

#Angular FD weights arranged in a differentiation matrix:

# Wth = periodic.getDM( th, th, 1 )
Wth   = phs1.getPeriodicDM( period=2*np.pi, X=th, m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvth = phs1.getPeriodicDM( period=2*np.pi, X=th, m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

###########################################################################

#Weights for interpolation to boundary and extrapolation to ghost-nodes:

wIinner = phs1.getWeights( innerRadius, s[0:stc],   0, phs, pol )
wEinner = phs1.getWeights( s[0],        s[1:stc+1], 0, phs, pol )

wIouter = phs1.getWeights( outerRadius, s[-1:-stc-1:-1], 0, phs, pol )
wEouter = phs1.getWeights( s[-1],       s[-2:-stc-2:-1], 0, phs, pol )

###########################################################################

#Interpolation from perturbed nodes to regular nodes for plotting:

W = phs1.getDM( x=s, X=s0[1:-1], m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

###########################################################################

#Functions to approximate differential operators and other things:

def Ds( U ) :
    return Ws.dot( U )

def Dth( U ) :
    return np.transpose( Wth.dot( np.transpose(U) ) )

def HVs( U ) :
    return Whvs.dot( U )

def HVth( U ) :
    np.transpose( Whvth.dot( np.transpose(U) ) )

def setGhostNodes( U ) :
    return annulus.setGhostNodes( U \
    , rhoB, rhoT, wIinner, wEinner, wIouter, wEouter )

def odefun( t, U ) :
    return annulus.odefun( t, U \
    , setGhostNodes, Ds, Dth, HVs, HVth \
    , thth[1:-1,:], ss[1:-1,:], c )

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