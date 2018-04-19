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

c           = .1                                                #wave speed
innerRadius = 1.
outerRadius = 2.
tf          = 10.                                               #final time
k           = 100.                 #controls steepness of initial condition

rkStages      = 3                    #number of Runge-Kutta stages (3 or 4)
saveDel       = 1                          #time interval to save snapshots
plotFromSaved = 0                            #if 1, load instead of compute

ns = 256+2                                        #total number of s levels
dt = 1./80                                                         #delta t

phs = 7                                      #PHS RBF exponent (odd number)
pol = 5                                                  #polynomial degree
stc = 13                                                      #stencil size
ptb = .00                            #random radial perturbation percentage

xc1 = (innerRadius+outerRadius)/2.*np.cos(np.pi/4)      #x-coord of GA bell
yc1 = (innerRadius+outerRadius)/2.*np.sin(np.pi/4)      #y-coord of GA bell
def initialCondition( x, y ) :
    return np.exp( -k*( (x-xc1)**2. + (y-yc1)**2. ) )

rSurf, dsdth, dsdr \
= annulus.getHeightCoordinate( outerRadius, innerRadius )

###########################################################################

saveString = 'c' + '{0:1.2f}'.format(c)  \
+ '_ri' + '{0:1.0f}'.format(innerRadius) \
+ '_ro' + '{0:1.0f}'.format(outerRadius) \
+ '_tf' + '{0:04.0f}'.format(tf)         \
+ '_k'  + '{0:03.0f}'.format(k)

saveString = saveString + '/'     \
+ 'phs'  + '{0:1d}'.format(phs)   \
+ '_pol' + '{0:1d}'.format(pol)   \
+ '_stc' + '{0:1d}'.format(stc)   \
+ '_ptb' + '{0:1.2f}'.format(ptb) \
+ '_ns'  + '{0:1d}'.format(ns-2)  \
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
ptb = ptb * ds                                #relative perturbation factor
ran = -ptb + 2*ptb*np.random.rand(len(s0))      #random perturbation vector
s = s0.copy()
s = s + ran                                    #s vector after perturbation
ds = ( s[2:len(s)] - s[0:len(s)-2] ) / 2.             #non-constant delta s

np.save( saveString+'s'+'.npy', s )                #save vector of s values

###########################################################################

thth, ss = np.meshgrid( th, s )      #mesh of perturbed s values and angles
rr = annulus.getRadii( thth, ss \
, innerRadius, outerRadius, rSurf )                #mesh of perturbed radii
xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates

thth0, ss0 = np.meshgrid( th, s0[1:-1] )                 #mesh for plotting
rr0 = annulus.getRadii( thth0, ss0 \
, innerRadius, outerRadius, rSurf )                  #mesh of regular radii
xx0 = rr0 * np.cos(thth0)                             #mesh of reg x-coords
yy0 = rr0 * np.sin(thth0)                             #mesh of reg x-coords

###########################################################################

#Plot showing how much the radii have been perturbed:

# fig, ax = plt.subplots( 1, 2, figsize=(8,4) )
# ax[0].plot( s0, s0, '-', s0, s, '.' )       #plot of initial vs perturbed s
# plt.xlabel('s0')
# plt.ylabel('s')
# ax[1].plot( s[1:-1], ds, '-' )                #plot of s vs non-constant ds
# plt.xlabel('s')
# plt.ylabel('ds')
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

#Set initial condition and fixed Dirichlet BC for U[0,:,:] (rho):

U = np.zeros(( 3, ns, nth ))
U[0,:,:] = initialCondition( xx, yy )

xB = rSurf(th) * np.cos(th)
yB = rSurf(th) * np.sin(th)
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
    alp = 0.                           #remove HV if using only polynomials

###########################################################################

#Radial weights arranged in a differentiation matrix:

Ws   = phs1.getDM( x=s, X=s[1:-1], m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvs = phs1.getDM( x=s, X=s[1:-1], m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

dsPol = spdiags( ds**pol, np.array([0]), len(ds), len(ds) )
Whvs = alp * dsPol.dot(Whvs)

###########################################################################

#Angular FD weights arranged in a differentiation matrix:

# Wth = periodic.getDM( th=th, TH=th, m=1 )

Wth   = phs1.getPeriodicDM( period=2*np.pi, X=th, m=1 \
, phsDegree=9, polyDegree=7, stencilSize=17 )

Whvth = phs1.getPeriodicDM( period=2*np.pi, X=th, m=8 \
, phsDegree=9, polyDegree=7, stencilSize=17 )

Whvth = -2.**-15. * dth**7. * Whvth

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
    return np.transpose( Whvth.dot( np.transpose(U) ) )

def setGhostNodes( U ) :
    return annulus.setGhostNodesNoLoop( U \
    , rhoB, rhoT \
    , np.transpose(np.tile(wIinner,(nth,1))) \
    , np.transpose(np.tile(wEinner,(nth,1))) \
    , np.transpose(np.tile(wIouter,(nth,1))) \
    , np.transpose(np.tile(wEouter,(nth,1))) \
    , stc )
    # return annulus.setGhostNodes( U \
    # , rhoB, rhoT, wIinner, wEinner, wIouter, wEouter, stc )

def odefun( t, U ) :
    return annulus.odefun( t, U \
    , setGhostNodes, Ds, Dth, HVs, HVth \
    , thth[1:-1,:], rr[1:-1,:], c \
    , dsdth(rr[1:-1,:],thth[1:-1,:]), dsdr(rr[1:-1,:],thth[1:-1,:]) )

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