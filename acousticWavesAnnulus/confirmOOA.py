#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import phs1
from gab.annulus import common, waveEquation

###########################################################################

c           = .03                                     #wave speed (c**2=RT)
innerRadius = 1.
outerRadius = 2.
tf          = 20.                                               #final time
saveDel     = 2                            #time interval to save snapshots
exp         = 200.                 #controls steepness of initial condition
amp         = .10                 #amplitude of trigonometric topo function
frq         = 5                   #frequency of trigonometric topo function

ord = 2                                        #norm to use for error check

contourErrors = 1

mlvA      = 1
phsA      = 7
polA      = 5
stcA      = 13
ptbA      = .00
rkStagesA = 4

mlvB      = 1
phsB      = 7
polB      = 5
stcB      = 13
ptbB      = .30
rkStagesB = 4

mlv0      = 1
phs0      = 7
pol0      = 5
stc0      = 13
ptb0      = .00
rkStages0 = 4

t0 = tf                                                    #time to look at

ns0 = 384                                             #reference resolution
ns1 = 12
ns2 = 24
ns3 = 48
ns4 = 96
ns5 = 192

dt0 = 1./32.
dt1 = 1./1.
dt2 = 1./2.
dt3 = 1./4.
dt4 = 1./8.
dt5 = 1./16.

###########################################################################

def loadSingleResult( mlv, phs, pol, stc, ptb, rkStages, ns, dt, t0 ) :
    
    if mlv == 1 :
        ns = ns + 2
    else :
        ns = ns + 3
    
    saveString = waveEquation.getSavestring( c, innerRadius, outerRadius \
    , tf, saveDel, exp, amp, frq \
    , mlv, phs, pol, stc, ptb, rkStages, ns, dt )
    
    U  = np.load( saveString + '{0:04d}'.format(np.int(np.round(t0))) + '.npy' )
    s  = np.load( saveString + 's' + '.npy' )
    th = np.load( saveString + 'th' + '.npy' )
    
    return U[0,:,:], s, th

###########################################################################

#Load reference solution and things to compare it to:

def loadManyResults( mlv, phs, pol, stc, ptb, rkStages ) :
    
    U0, s0, th0 = loadSingleResult( mlv0, phs0, pol0, stc0, ptb0, rkStages0, ns0, dt0, t0 )
    U1, s1, th1 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rkStages,  ns1, dt1, t0 )
    U2, s2, th2 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rkStages,  ns2, dt2, t0 )
    U3, s3, th3 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rkStages,  ns3, dt3, t0 )
    U4, s4, th4 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rkStages,  ns4, dt4, t0 )
    U5, s5, th5 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rkStages,  ns5, dt5, t0 )
    
    return U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5 \
    , th0, th1, th2, th3, th4, th5

###########################################################################

#Interpolate radially to reference grid using PHS-FD:

def interpRadial( U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5 ) :
    
    phsR = 7
    polR = 5
    stcR = 13
    
    W1 = phs1.getDM( x=s1, X=s0[1:-1], m=0 \
    , phsDegree=phsR, polyDegree=polR, stencilSize=stcR )
    W2 = phs1.getDM( x=s2, X=s0[1:-1], m=0 \
    , phsDegree=phsR, polyDegree=polR, stencilSize=stcR )
    W3 = phs1.getDM( x=s3, X=s0[1:-1], m=0 \
    , phsDegree=phsR, polyDegree=polR, stencilSize=stcR )
    W4 = phs1.getDM( x=s4, X=s0[1:-1], m=0 \
    , phsDegree=phsR, polyDegree=polR, stencilSize=stcR )
    W5 = phs1.getDM( x=s5, X=s0[1:-1], m=0 \
    , phsDegree=phsR, polyDegree=polR, stencilSize=stcR )
    
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
    
    phsA = 9
    polA = 7
    stcA = 17
    
    W1 = phs1.getPeriodicDM( period=2*np.pi, x=th1, X=th0, m=0 \
    , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
    W2 = phs1.getPeriodicDM( period=2*np.pi, x=th2, X=th0, m=0 \
    , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
    W3 = phs1.getPeriodicDM( period=2*np.pi, x=th3, X=th0, m=0 \
    , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
    W4 = phs1.getPeriodicDM( period=2*np.pi, x=th4, X=th0, m=0 \
    , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
    W5 = phs1.getPeriodicDM( period=2*np.pi, x=th5, X=th0, m=0 \
    , phsDegree=phsA, polyDegree=polA, stencilSize=stcA )
    
    U1 = np.transpose( W1.dot(np.transpose(U1)) )
    U2 = np.transpose( W2.dot(np.transpose(U2)) )
    U3 = np.transpose( W3.dot(np.transpose(U3)) )
    U4 = np.transpose( W4.dot(np.transpose(U4)) )
    U5 = np.transpose( W5.dot(np.transpose(U5)) )
    
    return U0, U1, U2, U3, U4, U5

###########################################################################

def gatherErrors( U0, U1, U2, U3, U4, U5 ) :
    
    # err1 = np.linalg.norm( U1-U2, ord )
    # err2 = np.linalg.norm( U2-U3, ord )
    # err3 = np.linalg.norm( U3-U4, ord )
    # err4 = np.linalg.norm( U4-U5, ord )
    # err5 = np.linalg.norm( U5-U0, ord )
    
    err1 = np.linalg.norm( U1-U0, ord )
    err2 = np.linalg.norm( U2-U0, ord )
    err3 = np.linalg.norm( U3-U0, ord )
    err4 = np.linalg.norm( U4-U0, ord )
    err5 = np.linalg.norm( U5-U0, ord )
    
    err = np.hstack(( err1, err2, err3, err4, err5 ))
    
    # err = err / np.linalg.norm(U0,ord)
    
    return err

###########################################################################

def getErrorVector( mlv, phs, pol, stc, ptb, rkStages ) :
    
    U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5, th0, th1, th2, th3, th4, th5  \
    = loadManyResults( mlv, phs, pol, stc, ptb, rkStages )
    
    U0, U1, U2, U3, U4, U5 = interpRadial( U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5 )
    
    U0, U1, U2, U3, U4, U5 = interpAngular( U0, U1, U2, U3, U4, U5 \
    , th0, th1, th2, th3, th4, th5 )
    
    err = gatherErrors( U0, U1, U2, U3, U4, U5 )
    
    return err, U0, U1, U2, U3, U4, U5, th0, s0

###########################################################################

#Plot the error to check convergence:

errA, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( mlvA, phsA, polA, stcA, ptbA, rkStagesA )

errB, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( mlvB, phsB, polB, stcB, ptbB, rkStagesB )

ns= np.hstack(( ns1, ns2, ns3, ns4, ns5 ))
ns = ns - 2

dom = np.array( [ 1.4, 2.4 ] )
width = dom[1] - dom[0]
shift = -0.

plt.plot( np.log10(ns), np.log10(errA), '-' )
plt.plot( np.log10(ns), np.log10(errB), '-' )
plt.plot( dom, np.array([shift,shift-1.*width]), '--' )
plt.plot( dom, np.array([shift,shift-2.*width]), '--' )
plt.plot( dom, np.array([shift,shift-3.*width]), '--' )
plt.plot( dom, np.array([shift,shift-4.*width]), '--' )
plt.plot( dom, np.array([shift,shift-5.*width]), '--' )
plt.legend(( 'A', 'B', '1st order', '2nd order' \
, '3rd order', '4th order', '5th order' ))
plt.plot( np.log10(ns), np.log10(errA), 'k.' )
plt.plot( np.log10(ns), np.log10(errB), 'k.' )
plt.xlabel( 'log10(ns)' )
plt.ylabel( 'log10(absErr)' )
plt.show()

###########################################################################

#Contour plot of differences at final time:

if contourErrors == 1 :
    
    rSurf, rSurfPrime = common.getTopoFunc( innerRadius, outerRadius, amp, frq )
    
    thth0, ss0 = np.meshgrid( th0, s0[1:-1] )
    rr0 = common.getRadii( thth0, ss0, innerRadius, outerRadius, rSurf )
    xx0 = rr0 * np.cos(thth0)
    yy0 = rr0 * np.sin(thth0)
    
    # e1 = np.abs( U1 - U2 )
    # e2 = np.abs( U2 - U3 )
    # e3 = np.abs( U3 - U4 )
    # e4 = np.abs( U4 - U5 )
    # e5 = np.abs( U5 - U0 )
    
    e1 = np.abs( U1 - U0 )
    e2 = np.abs( U2 - U0 )
    e3 = np.abs( U3 - U0 )
    e4 = np.abs( U4 - U0 )
    e5 = np.abs( U5 - U0 )
    
    e1 = e1.flatten()
    e2 = e2.flatten()
    e3 = e3.flatten()
    e4 = e4.flatten()
    e5 = e5.flatten()
    
    # ep = 10.**-20.
    
    # e1[e1<=ep] = ep
    # e2[e2<=ep] = ep
    # e3[e3<=ep] = ep
    # e4[e4<=ep] = ep
    # e5[e5<=ep] = ep
    
    # e1 = np.log10(e1)
    # e2 = np.log10(e2)
    # e3 = np.log10(e3)
    # e4 = np.log10(e4)
    # e5 = np.log10(e5)
    
    e1 = np.reshape( e1, np.shape(xx0) )
    e2 = np.reshape( e2, np.shape(xx0) )
    e3 = np.reshape( e3, np.shape(xx0) )
    e4 = np.reshape( e4, np.shape(xx0) )
    e5 = np.reshape( e5, np.shape(xx0) )
    
    # cvec = np.arange(-20,1,1)
    cvec = 20
    
    plt.figure(1)
    plt.contourf( xx0, yy0, e1, cvec )
    plt.axis('equal')
    plt.colorbar()
    plt.title( 'max={0:1.2e}'.format(np.max(np.abs(e1))) )
    
    plt.figure(2)
    plt.contourf( xx0, yy0, e2, cvec )
    plt.axis('equal')
    plt.colorbar()
    plt.title( 'max={0:1.2e}'.format(np.max(np.abs(e2))) )
    
    plt.figure(3)
    plt.contourf( xx0, yy0, e3, cvec )
    plt.axis('equal')
    plt.colorbar()
    plt.title( 'max={0:1.2e}'.format(np.max(np.abs(e3))) )
    
    plt.figure(4)
    plt.contourf( xx0, yy0, e4, cvec )
    plt.axis('equal')
    plt.colorbar()
    plt.title( 'max={0:1.2e}'.format(np.max(np.abs(e4))) )
    
    plt.figure(5)
    plt.contourf( xx0, yy0, e5, cvec )
    plt.axis('equal')
    plt.colorbar()
    plt.title( 'max={0:1.2e}'.format(np.max(np.abs(e5))) )
    
    plt.show()

###########################################################################
