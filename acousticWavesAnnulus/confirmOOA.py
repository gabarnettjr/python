import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import phs1, phs2
from gab.pseudospectral import periodic
from gab.annulus import common, waveEquation

###########################################################################

c           = .01                                               #wave speed
innerRadius = 2.
outerRadius = 3.
tf          = 20.                                               #final time
saveDel     = 2                            #time interval to save snapshots
exp         = 100.                 #controls steepness of initial condition
amp         = .10        #relative amplitude of trigonometric topo function
frq         = 9                   #frequency of trigonometric topo function

ord = 2                                        #norm to use for error check

contourErrors = 1

dimSplitA = 2
phsA      = 7
polA      = 5
stcA      = 13
ptbA      = .00
rkStagesA = 3

dimSplitB = 2
phsB      = 7
polB      = 5
stcB      = 13
ptbB      = .00
rkStagesB = 4

dimSplit0 = 2
phs0      = 7
pol0      = 5
stc0      = 13
ptb0      = .00
rkStages0 = 4

t0 = tf

ns0 = 256+2                                           #reference resolution
ns1 = 16+2
ns2 = 16+2
ns3 = 32+2
ns4 = 64+2
ns5 = 128+2

dt0 = 1./64.
dt1 = 1./4.
dt2 = 1./4.
dt3 = 1./8.
dt4 = 1./16.
dt5 = 1./32.

rSurf, rSurfPrime, sFunc, dsdth, dsdr \
= common.getHeightCoordinate( innerRadius, outerRadius, amp, frq )

###########################################################################

def loadSingleResult( dimSplit, phs, pol, stc, ptb, rkStages, ns, dt, t0 ) :
    
    saveString = waveEquation.getSavestring( c, innerRadius, outerRadius, tf, saveDel, exp, amp, frq \
    , dimSplit, phs, pol, stc, ptb, rkStages, ns, dt )
    
    U  = np.load( saveString+'{0:04d}'.format(np.int(np.round(t0)))+'.npy' )
    s  = np.load( saveString+'s'+'.npy' )
    th = np.load( saveString+'th'+'.npy' )
    
    return U[0,:,:], s, th

###########################################################################

#Load reference solution and things to compare it to:

def loadManyResults( dimSplit, phs, pol, stc, ptb, rkStages ) :
    
    U0, s0, th0 = loadSingleResult( dimSplit0, phs0, pol0, stc0, ptb0, rkStages0, ns0, dt0, t0 )
    U1, s1, th1 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns1, dt1, t0 )
    U2, s2, th2 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns2, dt2, t0 )
    U3, s3, th3 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns3, dt3, t0 )
    U4, s4, th4 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns4, dt4, t0 )
    U5, s5, th5 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns5, dt5, t0 )
    
    return U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5 \
    , th0, th1, th2, th3, th4, th5

###########################################################################

#Interpolate radially to reference grid using PHS-FD:

def interpRadial( U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5 ) :

    # W0 = phs1.getDM( x=s0, X=s0[1:-1], m=0, phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W1 = phs1.getDM( x=s1, X=s0[1:-1], m=0, phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W2 = phs1.getDM( x=s2, X=s0[1:-1], m=0, phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W3 = phs1.getDM( x=s3, X=s0[1:-1], m=0, phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W4 = phs1.getDM( x=s4, X=s0[1:-1], m=0, phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W5 = phs1.getDM( x=s5, X=s0[1:-1], m=0, phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )

    # U0 = W0.dot( U0 )
    U0 = U0[1:-1,:]
    U1 = W1.dot( U1 )
    U2 = W2.dot( U2 )
    U3 = W3.dot( U3 )
    U4 = W4.dot( U4 )
    U5 = W5.dot( U5 )
    
    return U0, U1, U2, U3, U4, U5

###########################################################################

#Interpolate angularly to reference grid using PHS-FD:

def interpAngular( U0, U1, U2, U3, U4, U5, th0, th1, th2, th3, th4, th5 ) :

    # W0 = phs1.getPeriodicDM( period=2*np.pi, x=th0, X=th0, m=0 \
    # , phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W1 = phs1.getPeriodicDM( period=2*np.pi, x=th1, X=th0, m=0 \
    , phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W2 = phs1.getPeriodicDM( period=2*np.pi, x=th2, X=th0, m=0 \
    , phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W3 = phs1.getPeriodicDM( period=2*np.pi, x=th3, X=th0, m=0 \
    , phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W4 = phs1.getPeriodicDM( period=2*np.pi, x=th4, X=th0, m=0 \
    , phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )
    W5 = phs1.getPeriodicDM( period=2*np.pi, x=th5, X=th0, m=0 \
    , phsDegree=phs0, polyDegree=pol0, stencilSize=stc0 )

    # U0 = np.transpose( W0.dot(np.transpose(U0)) )
    U1 = np.transpose( W1.dot(np.transpose(U1)) )
    U2 = np.transpose( W2.dot(np.transpose(U2)) )
    U3 = np.transpose( W3.dot(np.transpose(U3)) )
    U4 = np.transpose( W4.dot(np.transpose(U4)) )
    U5 = np.transpose( W5.dot(np.transpose(U5)) )
    
    return U0, U1, U2, U3, U4, U5

###########################################################################

def gatherErrors( U0, U1, U2, U3, U4, U5 ) :

    err1 = np.linalg.norm( U1-U0, ord )
    err2 = np.linalg.norm( U2-U0, ord )
    err3 = np.linalg.norm( U3-U0, ord )
    err4 = np.linalg.norm( U4-U0, ord )
    err5 = np.linalg.norm( U5-U0, ord )
    
    err = np.hstack(( err1, err2, err3, err4, err5 ))
    
    err = err / np.linalg.norm(U0,ord)
    
    return err

###########################################################################

def getErrorVector( dimSplit, phs, pol, stc, ptb, rkStages ) :
    
    U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5, th0, th1, th2, th3, th4, th5  \
    = loadManyResults( dimSplit, phs, pol, stc, ptb, rkStages )
    
    U0, U1, U2, U3, U4, U5 = interpRadial( U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5 )
    
    U0, U1, U2, U3, U4, U5 = interpAngular( U0, U1, U2, U3, U4, U5 \
    , th0, th1, th2, th3, th4, th5 )
    
    err = gatherErrors( U0, U1, U2, U3, U4, U5 )
    
    return err, U0, U1, U2, U3, U4, U5, th0, s0

###########################################################################

#Plot the error to check convergence:

errA, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( dimSplitA, phsA, polA, stcA, ptbA, rkStagesA )

errB, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( dimSplitB, phsB, polB, stcB, ptbB, rkStagesB )

ns= np.hstack(( ns1, ns2, ns3, ns4, ns5 ))
ns = ns - 2

plt.plot( np.log(ns), np.log(errA), '-' )
plt.plot( np.log(ns), np.log(errB), '-' )
plt.plot( np.array([3.,5.]), np.array([-1.,-5.]), '--' )
plt.plot( np.array([3.,5.]), np.array([-1.,-7.]), '--' )
plt.plot( np.array([3.,5.]), np.array([-1.,-9.]),  '--' )
# plt.plot( np.array([3.,5.]), np.array([-1.,-11.]), '--' )
plt.legend(( 'A', 'B', '2nd order', '3rd order', '4th order' ))
plt.plot( np.log(ns), np.log(errA), 'k.' )
plt.plot( np.log(ns), np.log(errB), 'k.' )
plt.xlabel( 'log(ns)' )
plt.ylabel( 'log(relMaxNormErr)' )
plt.show()

###########################################################################

#Contour plot of differences at final time:

if contourErrors == 1 :

    rSurf, rSurfPrime = common.getTopoFunc( innerRadius, outerRadius, amp, frq )

    thth0, ss0 = np.meshgrid( th0, s0[1:-1] )
    rr0 = common.getRadii( thth0, ss0, innerRadius, outerRadius, rSurf )
    xx0 = rr0 * np.cos(thth0)
    yy0 = rr0 * np.sin(thth0)

    nContours = 20

    plt.figure(1)
    plt.contourf( xx0, yy0, U1-U0, nContours )
    plt.axis('equal')
    plt.colorbar()

    plt.figure(2)
    plt.contourf( xx0, yy0, U2-U0, nContours )
    plt.axis('equal')
    plt.colorbar()

    plt.figure(3)
    plt.contourf( xx0, yy0, U3-U0, nContours )
    plt.axis('equal')
    plt.colorbar()

    plt.figure(4)
    plt.contourf( xx0, yy0, U4-U0, nContours )
    plt.axis('equal')
    plt.colorbar()

    plt.figure(5)
    plt.contourf( xx0, yy0, U5-U0, nContours )
    plt.axis('equal')
    plt.colorbar()

    plt.show()

###########################################################################