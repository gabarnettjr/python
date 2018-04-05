import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import rk
from gab.finiteDifferences import annulus

###########################################################################

innerRadius   = 1.
outerRadius   = 2.
rkStages      = 3                    #number of Runge-Kutta stages (3 or 4)
saveDel       = 5                          #time interval to save snapshots
plotFromSaved = 1                            #if 1, load instead of compute

c    = 1./10.                                                   #wave speed
nr   = 128+2                                       #number of radial levels
FDr  = 2                                         #order of radial FD approx
FDth = 4                                        #order of angular FD approx
dt   = 1./16.                                                      #delta t
tf   = 100.                                                     #final time

alp = 1.                                             #radial HV coefficient
bet = 1.                                            #angular HV coefficient

###########################################################################

saveString = './results/' + 'nr' + '{0:1d}'.format(nr-2) + '/'

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )

if not os.path.exists( saveString ) :
    os.makedirs( saveString )

###########################################################################

t = 0.
nTimesteps = np.int(np.round( tf / dt ))

nth = np.int(np.round(2*np.pi*innerRadius*(nr-2)))#number of angular levels
dth = 2.*np.pi / nth                                  #constant delta theta
th = np.linspace( 0., 2.*np.pi, nth+1 )                       #angle vector
th = th[0:-1]                            #remove last angle (same as first)

dr = ( outerRadius - innerRadius ) / (nr-2)               #constant delta r
r  = np.linspace( innerRadius-dr/2, outerRadius+dr/2, nr )   #radius vector

thth, rr = np.meshgrid( th, r )                   #mesh of radii and angles

xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates

# plt.plot( xx.flatten(), yy.flatten(), "." )
# plt.axis('equal')
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

#Set initial conditions:

xc = 0.                                                 #x-coord of IC bell
yc = ( innerRadius + outerRadius ) / 2.                 #y-coord of IC bell
U = np.zeros(( 3, nr, nth ))
U[0,:,:] = np.exp( -10.*( (xx-xc)**2. + (yy-yc)**2. ) )             #set IC
rhoB = ( U[0,0,:] + U[0,1,:] ) / 2.
rhoT = ( U[0,-1,:] + U[0,-2,:] ) / 2.

###########################################################################

wr    = annulus.getCenteredWeights( 1,    FDr  )         #radial derivative
wHVr  = annulus.getCenteredWeights( FDr,  FDr  )                 #radial HV
wth   = annulus.getCenteredWeights( 1,    FDth )        #angular derivative
wHVth = annulus.getCenteredWeights( FDth, FDth )                #angular HV

ii, jj = annulus.getMainIndex( FDr, FDth, nr, nth )

# app = annulus.Lr( U[0,:,:], FDr, ii, wr, dr )
# plt.contourf( xx[ii,:], yy[ii,:], app, 20 )
# plt.axis('equal')
# plt.colorbar()
# plt.show()
# sys.exit("\nStop here for now.\n")

###########################################################################

def Dr( U ) :
    return annulus.Lr( U \
    , FDr, ii, wr, dr )

def Dth( U ) :
    return annulus.Lth( U \
    , FDth, jj, wth, dth )

def HVr( U ) :
    return alp * annulus.Lr( U \
    , FDr, ii, wHVr, dr )

def HVth( U ) :
    return bet * annulus.Lth( U \
    , FDth, jj, wHVth, dth )

def odefun( t, U ) :
    return annulus.odefun( t, U \
    , Dr, Dth, HVr, HVth, thth[ii,:], rr[ii,:], ii, c, rhoB, rhoT )

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit("\nError: rkStages should be 3 or 4 in this problem.\n")

###########################################################################

#Main time-stepping loop:

fig = plt.figure( figsize = (18,14) )

for i in np.arange(0,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        print( "t =", t )
        if plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        plt.contourf( xx, yy, U[1,:,:], np.arange(-2,2.1,.1) )
        plt.axis('equal')
        plt.colorbar()
        fig.savefig( '{0:04d}'.format(np.int(np.round(t)+1e-12))+'.png', bbox_inches = 'tight' )
        plt.clf()
        
    if plotFromSaved == 1 :
        t = t + dt
    else :
        [ t, U ] = rk( t, U, odefun, dt )

###########################################################################