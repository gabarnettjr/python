import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

sys.path.append( '../site-packages' )

from gab import rk, phs1, phs2
from gab.acousticWaveEquation import annulus
from gab.pseudospectral import periodic

###########################################################################

c           = .1                                                #wave speed
innerRadius = 2.
outerRadius = 3.
tf          = 50.                                               #final time
saveDel     = 5                            #time interval to save snapshots
exp         = 100.                 #controls steepness of initial condition
amp         = .05        #relative amplitude of trigonometric topo function
frq         = 9                   #frequency of trigonometric topo function

plotFromSaved = 1                            #if 1, load instead of compute

dimSplit = np.int64(sys.argv[1])               #0:none, 1:some, 2:fullSplit
phs      = np.int64(sys.argv[2])             #PHS RBF exponent (odd number)
pol      = np.int64(sys.argv[3])                         #polynomial degree
stc      = np.int64(sys.argv[4])                              #stencil size
ptb      = np.float64(sys.argv[5])   #random radial perturbation percentage
rkStages = np.int64(sys.argv[6])     #number of Runge-Kutta stages (3 or 4)
ns       = np.int64(sys.argv[7])+2                #total number of s levels
dt       = 1./np.float64(sys.argv[8])                              #delta t

rSurf, dsdth, dsdr \
= annulus.getHeightCoordinate( innerRadius, outerRadius, amp, frq )

tmp = np.pi
xc1 = (rSurf(tmp)+outerRadius)/2.*np.cos(tmp)           #x-coord of GA bell
yc1 = (rSurf(tmp)+outerRadius)/2.*np.sin(tmp)           #y-coord of GA bell
def initialCondition( x, y ) :
    return np.exp( -exp*( (x-xc1)**2. + (y-yc1)**2. ) )

###########################################################################

#Set things up for saving:

saveString = annulus.getSavestring( c, innerRadius, outerRadius, tf, saveDel, exp, amp, frq \
, dimSplit, phs, pol, stc, ptb, rkStages, ns, dt )

if os.path.exists( saveString+'*.npy' ) :
    os.remove( saveString+'*.npy' )                       #remove old files

if not os.path.exists( saveString ) :
    os.makedirs( saveString )                         #make new directories

###########################################################################

#Get th and s vectors and save s vector (randomly perturbed):

t = 0.                                                       #starting time
nTimesteps = np.int(np.round( tf / dt ))         #total number of timesteps

nth = annulus.getNth( innerRadius, outerRadius, ns )#nmbr of angular levels
dth = 2.*np.pi / nth                                  #constant delta theta
th  = np.linspace( 0., 2.*np.pi, nth+1 )              #vector of all angles
th  = th[0:-1]                           #remove last angle (same as first)

ds  = ( outerRadius - innerRadius ) / (ns-2)              #constant delta s
s0  = np.linspace( innerRadius-ds/2, outerRadius+ds/2, ns )       #s vector
ptb = ptb * ds                                #relative perturbation factor
ran = -ptb + 2*ptb*np.random.rand(len(s0))      #random perturbation vector
s   = s0.copy()
s   = s + ran                                  #s vector after perturbation
ds  = ( s[2:len(s)] - s[0:len(s)-2] ) / 2.            #non-constant delta s

np.save( saveString+'s'+'.npy', s )                #save vector of s values

###########################################################################

#Get computational mesh and mesh for contour plotting:

thth, ss = np.meshgrid( th, s )      #mesh of perturbed s values and angles
rr = annulus.getRadii( thth, ss \
, innerRadius, outerRadius, rSurf )                #mesh of perturbed radii
xx = rr * np.cos(thth)                               #mesh of x-coordinates
yy = rr * np.sin(thth)                               #mesh of y-coordinates
xxi = xx[1:-1,:]                                       #exclude ghost nodes
yyi = yy[1:-1,:]                                       #exclude ghost nodes

thth = thth[1:-1,:]                                     #remove ghost nodes
rr = rr[1:-1,:]                                         #remove ghost nodes
dsdth = dsdth( thth, rr )                   #overwrite function with values
dsdr  = dsdr( thth, rr )                    #overwrite function with values

cosTh = np.cos(thth)
sinTh = np.sin(thth)
cosThOverR = cosTh/rr
sinThOverR = sinTh/rr

thth0, ss0 = np.meshgrid( th, s0[1:-1] )         #regular mesh for plotting
rr0 = annulus.getRadii( thth0, ss0 \
, innerRadius, outerRadius, rSurf )                  #mesh of regular radii
xx0 = rr0 * np.cos(thth0)                             #mesh of reg x-coords
yy0 = rr0 * np.sin(thth0)                             #mesh of reg x-coords

###########################################################################

#Plot showing how the radii have been perturbed:

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
# plt.xlabel( 'x' )
# plt.ylabel( 'y' )
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

if ( dimSplit != 2 ) & ( plotFromSaved != 1 ) :
    
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
        stc = 9
    elif pol == 5 :
        stc = 21
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

Wlam   = phs1.getPeriodicDM( period=2*np.pi, X=th, m=1     \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvlam = phs1.getPeriodicDM( period=2*np.pi, X=th, m=phs-1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

Whvlam = alp * dth**pol * Whvlam                  #scaled angular HV matrix

Wlam = np.transpose( Wlam )                #work on rows instead of columns
Whvlam = np.transpose( Whvlam )            #work on rows instead of columns

###########################################################################

#Weights for interpolation to boundary and extrapolation to ghost-nodes:

wIinner = phs1.getWeights( innerRadius, s[0:stc],   0, phs, pol )
wEinner = phs1.getWeights( s[0],        s[1:stc+1], 0, phs, pol )

wIouter = phs1.getWeights( outerRadius, s[-1:-stc-1:-1], 0, phs, pol )
wEouter = phs1.getWeights( s[-1],       s[-2:-stc-2:-1], 0, phs, pol )

wIinner = np.transpose( np.tile( wIinner, (nth,1) ) )
wEinner = np.transpose( np.tile( wEinner, (nth,1) ) )
wIouter = np.transpose( np.tile( wIouter, (nth,1) ) )
wEouter = np.transpose( np.tile( wEouter, (nth,1) ) )

###########################################################################

#Interpolation from perturbed s values to regular s values for plotting:

W = phs1.getDM( x=s, X=s0[1:-1], m=0 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc )

###########################################################################

#Functions to approximate differential operators and other things:

if dimSplit == 2 :
    
    def Ds(U) :
        return Ws @ U
    
    def Dlam(U) :
        return U[1:-1,:] @ Wlam
    
    def Dr(U) :
        return Ds(U) * dsdr
    
    def Dth(U) :
        return Dlam(U) + Ds(U) * dsdth
    
    def Dx(U) :
        return cosTh * Dr(U) - sinThOverR * Dth(U)
    
    def Dy(U) :
        return sinTh * Dr(U) + cosThOverR * Dth(U)
    
    def HV(U) :
        return ( Whvs @ U ) + ( U[1:-1,:] @ Whvlam )
    
elif ( dimSplit == 1 ) | ( dimSplit == 0 ) :
    
    def Ds(U) :
        return Ws @ U
    
    def Dr(U) :
        return Ds(U) * dsdr
    
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
    return annulus.setGhostNodes1D( U \
    , rhoB, rhoT, wIinner, wEinner, wIouter, wEouter, stc )

def odefun( t, U ) :
    return annulus.odefunCartesian( t, U \
    , setGhostNodes, Dx, Dy, HV          \
    , c )

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
        
        # plt.contourf( xx0, yy0, W.dot(U[2,:,:]), 20 )
        plt.contourf( xx0, yy0, W.dot(U[0,:,:]), np.arange(-.17,.17+.02,.02) )
        plt.axis('equal')
        plt.colorbar()
        fig.savefig( '{0:04d}'.format(np.int(np.round(t)+1e-12))+'.png', bbox_inches = 'tight' )
        plt.clf()
        
        # if np.max(np.abs(U)) > 10. :
            # sys.exit("\nUnstable in time.\n")
        
    if plotFromSaved == 1 :
        t = t + dt
    else :
        [ t, U ] = rk( t, U, odefun, dt )

###########################################################################