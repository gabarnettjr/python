import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import rk, phs1
from gab.acousticWaveEquation import annulus

###########################################################################

innerRadius   = 2.
outerRadius   = 3.
rkStages      = 3                    #number of Runge-Kutta stages (3 or 4)
saveDel       = 20                         #time interval to save snapshots
plotFromSaved = 0                            #if 1, load instead of compute

c    = 1./10.                                                   #wave speed
nr   = 64+2                                  #total number of radial levels
FDr  = 4                                #order of radial FD approx (2 or 4)
FDth = 8                        #order of angular FD approx (2, 4, 6, or 8)
dt   = 1./16.                                                      #delta t
tf   = 500.                                                     #final time

alp = 1.                                  #radial HV coefficient (not used)
bet = 1.                                 #angular HV coefficient (not used)

xc1 = 0.                                                #x-coord of GA bell
yc1 = ( innerRadius + outerRadius ) / 2.                #y-coord of GA bell
def initialCondition( x, y ) :
    return np.exp( -20.*( (x-xc1)**2. + (y-yc1)**2. ) )

###########################################################################

saveString = './results/' + 'nr' + '{0:1d}'.format(nr-2) + '/'

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
th = np.linspace( 0., 2.*np.pi, nth+1 )                       #angle vector
th = th[0:-1]                            #remove last angle (same as first)

dr = ( outerRadius - innerRadius ) / (nr-2)               #constant delta r
r  = np.linspace( innerRadius-dr/2, outerRadius+dr/2, nr )   #radius vector

thth, rr = np.meshgrid( th, r )                   #mesh of radii and angles

xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates

###########################################################################

#Set initial conditions:

U = np.zeros(( 3, nr, nth ))
U[0,:,:] = initialCondition( xx, yy )

xB = ( xx[0,:] + xx[1,:] ) / 2.
yB = ( yy[0,:] + yy[1,:] ) / 2.
rhoB = initialCondition( xB, yB )

xT = ( xx[-2,:] + xx[-1,:] ) / 2.
yT = ( yy[-2,:] + yy[-1,:] ) / 2.
rhoT = initialCondition( xT, yT )

###########################################################################

# plt.plot( xx.flatten(), yy.flatten(), "." \
# , xB, yB, "-" \
# , xT, yT, "-" )
# plt.axis('equal')
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

# wr    = annulus.getFDweights( 1,    FDr  )               #radial derivative
wth   = annulus.getFDweights( 1,    FDth )              #angular derivative
# wHVr  = annulus.getFDweights( FDr,  FDr  )                       #radial HV
wHVth = annulus.getFDweights( FDth, FDth )                      #angular HV

phs = 5
pol = 5
stc = 13
Wr  = phs1.getDM( r, 1, phs, pol, stc )
Wr  = Wr[1:-1,:]

wI = phs1.getWeights( (r[0]+r[1])/2.,    r[0:stc],   0, phs, pol )
wE = phs1.getWeights( r[0],              r[1:stc+1], 0, phs, pol )
# wI, wE = annulus.getInterpExtrapWeights( FDr )             #weights for BCs

ii, jj = annulus.getMainIndex( FDr, FDth, nr, nth )

# app = annulus.Lr( U[0,:,:], FDr, ii, wr, dr )
# plt.contourf( xx[ii,:], yy[ii,:], app, 20 )
# plt.axis('equal')
# plt.colorbar()
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

def Dr( U ) :
    return Wr.dot(U)
    # return annulus.Lr( U \
    # , FDr, ii, wr, dr )

def Dth( U ) :
    return annulus.Lth( U \
    , FDth, jj, wth[np.int(np.round(FDth/2-1)),:], dth )

def HVr( U ) :
    return alp * annulus.Lr( U \
    , FDr, ii, wHVr, dr )

def HVth( U ) :
    return bet * annulus.Lth( U \
    , FDth, jj, wHVth[np.int(np.round(FDth/2-1)),:], dth )

def setGhostNodes( U ) :
    return annulus.setGhostNodes( U \
    , rhoB, rhoT, FDr, wI, wE )

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

for i in np.arange(0,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        print( "t =", np.int(np.round(t)), ",  et =", time.clock()-et )
        et = time.clock()
        if plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        plt.contourf( xx, yy, U[0,:,:], np.arange(-.25,.2625,.0125) )
        plt.axis('equal')
        plt.colorbar()
        fig.savefig( '{0:04d}'.format(np.int(np.round(t)+1e-12))+'.png', bbox_inches = 'tight' )
        plt.clf()
        
    if plotFromSaved == 1 :
        t = t + dt
    else :
        [ t, U ] = rk( t, U, odefun, dt )

###########################################################################