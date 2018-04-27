import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import phs1, phs2
from gab.pseudospectral import periodic
from gab.acousticWaveEquation import annulus

###########################################################################

c           = .1                                                #wave speed
innerRadius = 2.
outerRadius = 4.
tf          = 20.                                               #final time
saveDel     = 2                            #time interval to save snapshots
exp         = 25.                  #controls steepness of initial condition
amp         = .05        #relative amplitude of trigonometric topo function
frq         = 9                   #frequency of trigonometric topo function

ord = np.inf                                   #norm to use for error check

contourErrors = 1

dimSplitA = 2
phsA      = 5
polA      = 3
stcA      = 7
ptbA      = .00
rkStagesA = 3

dimSplitB = 2
phsB      = 5
polB      = 4
stcB      = 5
ptbB      = .00
rkStagesB = 3

dimSplit0 = 2
phs0      = 5
pol0      = 4
stc0      = 5
ptb0      = .00
rkStages0 = 3

t0 = tf

ns0 = 256+2                                           #reference resolution
ns1 = 8+2
ns2 = 16+2
ns3 = 32+2
ns4 = 64+2
ns5 = 128+2

dt0 = 1./32.
dt1 = 1./1.
dt2 = 1./2.
dt3 = 1./4.
dt4 = 1./8.
dt5 = 1./16.

rSurf, rSurfPrime, sFunc, dsdth, dsdr \
= annulus.getHeightCoordinate( innerRadius, outerRadius, amp, frq )

###########################################################################

def loadSingleResult( dimSplit, phs, pol, stc, ptb, rkStages, ns, dt, t0 ) :
    
    saveString = annulus.getSavestring( c, innerRadius, outerRadius, tf, saveDel, exp, amp, frq \
    , dimSplit, phs, pol, stc, ptb, rkStages, ns, dt )
    
    U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t0)))+'.npy' )
    s = np.load( saveString+'s'+'.npy' )
    
    return U[0,:,:], s

###########################################################################

#Load reference solution and things to compare it to:

def loadManyResults( dimSplit, phs, pol, stc, ptb, rkStages ) :
    
    U0, s0 = loadSingleResult( dimSplit0, phs0, pol0, stc0, ptb0, rkStages0, ns0, dt0, t0 )
    U1, s1 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns1, dt1, t0 )
    U2, s2 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns2, dt2, t0 )
    U3, s3 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns3, dt3, t0 )
    U4, s4 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns4, dt4, t0 )
    U5, s5 = loadSingleResult( dimSplit,  phs,  pol,  stc,  ptb,  rkStages,  ns5, dt5, t0 )
    
    return U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5

###########################################################################

#Get regularly spaced theta levels:

def getThetaLevels() :

    nth0 = annulus.getNth( innerRadius, outerRadius, ns0 )
    nth1 = annulus.getNth( innerRadius, outerRadius, ns1 )
    nth2 = annulus.getNth( innerRadius, outerRadius, ns2 )
    nth3 = annulus.getNth( innerRadius, outerRadius, ns3 )
    nth4 = annulus.getNth( innerRadius, outerRadius, ns4 )
    nth5 = annulus.getNth( innerRadius, outerRadius, ns5 )

    th0 = np.linspace( 0, 2.*np.pi, nth0+1 )
    th1 = np.linspace( 0, 2.*np.pi, nth1+1 )
    th2 = np.linspace( 0, 2.*np.pi, nth2+1 )
    th3 = np.linspace( 0, 2.*np.pi, nth3+1 )
    th4 = np.linspace( 0, 2.*np.pi, nth4+1 )
    th5 = np.linspace( 0, 2.*np.pi, nth5+1 )

    th0 = th0[0:-1]
    th1 = th1[0:-1]
    th2 = th2[0:-1]
    th3 = th3[0:-1]
    th4 = th4[0:-1]
    th5 = th5[0:-1]
    
    return th0, th1, th2, th3, th4, th5

###########################################################################

# def getXYpoints( th, s ) :
    
    # thth, ss = np.meshgrid( th, s )
    # rr = annulus.getRadii( thth, ss, innerRadius, outerRadius, rSurf )
    # x = ( rr * np.cos(thth) ) . flatten()
    # y = ( rr * np.sin(thth) ) . flatten()
    
    # return x, y

###########################################################################

# def interp2D( U0, U1, U2, U3, U4, U5 \
# ,  s0,  s1,  s2,  s3,  s4,  s5 \
# , th0, th1, th2, th3, th4, th5 ) :
    
    # x0, y0 = getXYpoints( th0, s0[1:-1] )
    # x1, y1 = getXYpoints( th1, s1 )
    # x2, y2 = getXYpoints( th2, s2 )
    # x3, y3 = getXYpoints( th3, s3 )
    # x4, y4 = getXYpoints( th4, s4 )
    # x5, y5 = getXYpoints( th5, s5 )
    
    # U0 = U0[1:-1,:]
    
    # rbfParam = 5
    # polyorder = 3
    # stencilSize = 25
    
    # stencils = phs2.getStencils( x1, y1, x0, y0, stencilSize )
    # A = phs2.getAmatrices( stencils, rbfParam, polyorder )
    # W = phs2.getWeights( stencils, A, "i", 0 )
    # U1 = np.sum( W*U1.flatten()[stencils.idx], axis=1 )
    # U1 = np.reshape( U1, (len(s0)-2,len(th0)) )
    
    # stencils = phs2.getStencils( x2, y2, x0, y0, stencilSize )
    # A = phs2.getAmatrices( stencils, rbfParam, polyorder )
    # W = phs2.getWeights( stencils, A, "i", 0 )
    # U2 = np.sum( W*U2.flatten()[stencils.idx], axis=1 )
    # U2 = np.reshape( U2, (len(s0)-2,len(th0)) )
    
    # stencils = phs2.getStencils( x3, y3, x0, y0, stencilSize )
    # A = phs2.getAmatrices( stencils, rbfParam, polyorder )
    # W = phs2.getWeights( stencils, A, "i", 0 )
    # U3 = np.sum( W*U3.flatten()[stencils.idx], axis=1 )
    # U3 = np.reshape( U3, (len(s0)-2,len(th0)) )
    
    # stencils = phs2.getStencils( x4, y4, x0, y0, stencilSize )
    # A = phs2.getAmatrices( stencils, rbfParam, polyorder )
    # W = phs2.getWeights( stencils, A, "i", 0 )
    # U4 = np.sum( W*U4.flatten()[stencils.idx], axis=1 )
    # U4 = np.reshape( U4, (len(s0)-2,len(th0)) )
    
    # stencils = phs2.getStencils( x5, y5, x0, y0, stencilSize )
    # A = phs2.getAmatrices( stencils, rbfParam, polyorder )
    # W = phs2.getWeights( stencils, A, "i", 0 )
    # U5 = np.sum( W*U5.flatten()[stencils.idx], axis=1 )
    # U5 = np.reshape( U5, (len(s0)-2,len(th0)) )
    
    # return U0, U1, U2, U3, U4, U5

###########################################################################

#Interpolate radially to reference grid using PHS-FD:

def interpRadial( U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5 \
, phs, pol, stc ) :

    W0 = phs1.getDM( x=s0, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
    W1 = phs1.getDM( x=s1, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
    W2 = phs1.getDM( x=s2, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
    W3 = phs1.getDM( x=s3, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
    W4 = phs1.getDM( x=s4, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
    W5 = phs1.getDM( x=s5, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )

    U0 = W0.dot( U0 )
    U1 = W1.dot( U1 )
    U2 = W2.dot( U2 )
    U3 = W3.dot( U3 )
    U4 = W4.dot( U4 )
    U5 = W5.dot( U5 )
    
    return U0, U1, U2, U3, U4, U5

###########################################################################

#Interpolate angularly to reference grid using periodic pseudospectral:

def interpAngular( U0, U1, U2, U3, U4, U5, th0, th1, th2, th3, th4, th5 ) :

    W0 = periodic.getDM( th=th0, TH=th0, m=0 )
    W1 = periodic.getDM( th=th1, TH=th0, m=0 )
    W2 = periodic.getDM( th=th2, TH=th0, m=0 )
    W3 = periodic.getDM( th=th3, TH=th0, m=0 )
    W4 = periodic.getDM( th=th4, TH=th0, m=0 )
    W5 = periodic.getDM( th=th5, TH=th0, m=0 )

    U0 = np.transpose( W0.dot(np.transpose(U0)) )
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
    
    U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5  \
    = loadManyResults( dimSplit, phs, pol, stc, ptb, rkStages )
    
    th0, th1, th2, th3, th4, th5 = getThetaLevels()
    
    U0, U1, U2, U3, U4, U5 = interpRadial( U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5, 7, 5, 13 )
    
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
plt.plot( np.array([3.,5.]), np.array([-0.,-4.]), '--' )
plt.plot( np.array([3.,5.]), np.array([-0.,-6.]), '--' )
# plt.plot( np.array([3.,5.]), np.array([-1.,-9.]),  '--' )
# plt.plot( np.array([3.,5.]), np.array([-1.,-11.]), '--' )
plt.legend(( 'A', 'B', '2nd order', '3rd order' ))
plt.plot( np.log(ns), np.log(errA), 'k.' )
plt.plot( np.log(ns), np.log(errB), 'k.' )
plt.xlabel( 'log(ns)' )
plt.ylabel( 'log(relMaxNormErr)' )
plt.show()

###########################################################################

#Contour plot of differences at final time:

if contourErrors == 1 :

    rSurf, rSurfPrime = annulus.getTopoFunc( innerRadius, outerRadius, amp, frq )

    thth0, ss0 = np.meshgrid( th0, s0[1:-1] )
    rr0 = annulus.getRadii( thth0, ss0, innerRadius, outerRadius, rSurf )
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