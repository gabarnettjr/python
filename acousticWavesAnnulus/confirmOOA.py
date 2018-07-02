#!/usr/bin/python3
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import phs1
from gab.annulus import common, waveEquation

###########################################################################

args = waveEquation.parseInput()
#get rid of the args prefix on all the variable names:
d = vars(args)
for k in d.keys() :
    exec("{} = args.{}".format(k,k))

###########################################################################

errNorm = np.inf                               #norm to use for error check

contourErrors = 1

mlvA = 1
phsA = 5
polA = 3
stcA = 7
ptbA = 0
rksA = 3

mlvB = 1
phsB = 7
polB = 5
stcB = 13
ptbB = 0
rks  = 4

mlv0 = 1
phs0 = 7
pol0 = 5
stc0 = 13
ptb0 = 0
rks0 = 4

t0 = tf                                                    #time to look at

nlv0 = 384                                             #reference resolution
nlv1 = 12
nlv2 = 24
nlv3 = 48
nlv4 = 96
nlv5 = 192

dti0reg = 32
dti1reg = 1
dti2reg = 2
dti3reg = 4
dti4reg = 8
dti5reg = 16

dti1ptb = dti1reg * 1
dti2ptb = dti2reg * 1
dti3ptb = dti3reg * 1
dti4ptb = dti4reg * 1
dti5ptb = dti5reg * 1

###########################################################################

def loadSingleResult( mlv, phs, pol, stc, ptb, rks, nlv, dti, t0 ) :
    
    if mlv == 1 :
        nlv = nlv + 2
    else :
        nlv = nlv + 3
    
    saveString = waveEquation.getSavestring( c, innerRadius, outerRadius \
    , tf, saveDel, exp, amp, frq \
    , mlv, phs, pol, stc, ptb, rks, nlv, dti )
    
    U  = np.load( saveString + '{0:04d}'.format(np.int(np.round(t0))) + '.npy' )
    s  = np.load( saveString + 's' + '.npy' )
    th = np.load( saveString + 'th' + '.npy' )
    
    return U[0,:,:], s, th

###########################################################################

#Load reference solution and things to compare it to:

def loadManyResults( mlv, phs, pol, stc, ptb, rks ) :
    
    if ptb == .00 :
        
        U0, s0, th0 = loadSingleResult( mlv0, phs0, pol0, stc0, ptb0, rks0, nlv0, dt0reg, t0 )
        U1, s1, th1 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv1, dt1reg, t0 )
        U2, s2, th2 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv2, dt2reg, t0 )
        U3, s3, th3 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv3, dt3reg, t0 )
        U4, s4, th4 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv4, dt4reg, t0 )
        U5, s5, th5 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv5, dt5reg, t0 )
        
    else :
        
        U0, s0, th0 = loadSingleResult( mlv0, phs0, pol0, stc0, ptb0, rks0, nlv0, dti0reg, t0 )
        U1, s1, th1 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv1, dti1ptb, t0 )
        U2, s2, th2 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv2, dti2ptb, t0 )
        U3, s3, th3 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv3, dti3ptb, t0 )
        U4, s4, th4 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv4, dti4ptb, t0 )
        U5, s5, th5 = loadSingleResult( mlv,  phs,  pol,  stc,  ptb,  rks,  nlv5, dti5ptb, t0 )
        
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
    
    # err1 = np.linalg.norm( U1-U2, errNorm )
    # err2 = np.linalg.norm( U2-U3, errNorm )
    # err3 = np.linalg.norm( U3-U4, errNorm )
    # err4 = np.linalg.norm( U4-U5, errNorm )
    # err5 = np.linalg.norm( U5-U0, errNorm )
    
    err1 = np.linalg.norm( U1-U0, errNorm )
    err2 = np.linalg.norm( U2-U0, errNorm )
    err3 = np.linalg.norm( U3-U0, errNorm )
    err4 = np.linalg.norm( U4-U0, errNorm )
    err5 = np.linalg.norm( U5-U0, errNorm )
    
    err = np.hstack(( err1, err2, err3, err4, err5 ))
    
    # err = err / np.linalg.norm(U0,errNorm)
    
    return err

###########################################################################

def getErrorVector( mlv, phs, pol, stc, ptb, rkStages ) :
    
    U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5, th0, th1, th2, th3, th4, th5  \
    = loadManyResults( mlv, phs, pol, stc, ptb, rks )
    
    U0, U1, U2, U3, U4, U5 = interpRadial( U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5 )
    
    U0, U1, U2, U3, U4, U5 = interpAngular( U0, U1, U2, U3, U4, U5 \
    , th0, th1, th2, th3, th4, th5 )
    
    err = gatherErrors( U0, U1, U2, U3, U4, U5 )
    
    return err, U0, U1, U2, U3, U4, U5, th0, s0

###########################################################################

#Plot the error to check convergence:

errA, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( mlvA, phsA, polA, stcA, ptbA, rksA )

errB, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( mlvB, phsB, polB, stcB, ptbB, rksB )

nlv= np.hstack(( nlv1, nlv2, nlv3, nlv4, nlv5 ))
nlv = nlv - 2

dom = np.array( [ 1.4, 2.4 ] )
width = dom[1] - dom[0]
shift = -0.

plt.plot( np.log10(nlv), np.log10(errA), '-' )
plt.plot( np.log10(nlv), np.log10(errB), '-' )
plt.plot( dom, np.array([shift,shift-1.*width]), '--' )
plt.plot( dom, np.array([shift,shift-2.*width]), '--' )
plt.plot( dom, np.array([shift,shift-3.*width]), '--' )
plt.plot( dom, np.array([shift,shift-4.*width]), '--' )
plt.plot( dom, np.array([shift,shift-5.*width]), '--' )
plt.legend(( 'A', 'B', '1st order', '2nd order' \
, '3rd order', '4th order', '5th order' ))
plt.plot( np.log10(nlv), np.log10(errA), 'k.' )
plt.plot( np.log10(nlv), np.log10(errB), 'k.' )
plt.xlabel( 'log10(nlv)' )
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
