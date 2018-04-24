import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import phs1
from gab.pseudospectral import periodic
from gab.acousticWaveEquation import annulus

###########################################################################

c           = .1                                                #wave speed
innerRadius = 1.
outerRadius = 2.
tf          = 10.                                               #final time
k           = 200.                 #controls steepness of initial condition
amp         = .05                 #amplitude of trigonometric topo function

ord = np.inf                                   #norm to use for error check

contourErrors = 1

phsA = 5
polA = 3
stcA = 7
ptbA = .30

phsB = 7
polB = 5
stcB = 13
ptbB = .30

phs0 = 7
pol0 = 5
stc0 = 13
ptb0 = .00

ns0 = 256+2                                           #reference resolution
ns1 = 16+2
ns2 = 16+2
ns3 = 32+2
ns4 = 64+2
ns5 = 128+2

###########################################################################

def loadSingleResult( ns, phs, pol, stc, ptb ) :
    
    saveString = 'c' + '{0:1.2f}'.format(c)   \
    + '_ri'  + '{0:1.0f}'.format(innerRadius) \
    + '_ro'  + '{0:1.0f}'.format(outerRadius) \
    + '_tf'  + '{0:04.0f}'.format(tf)         \
    + '_k'   + '{0:03.0f}'.format(k)          \
    + '_amp' + '{0:1.2f}'.format(amp)

    saveString = saveString + '/'            \
    + 'dimSplit' + '{0:1d}'.format(dimSplit) \
    + '_phs'     + '{0:1d}'.format(phs)      \
    + '_pol'     + '{0:1d}'.format(pol)      \
    + '_stc'     + '{0:1d}'.format(stc)      \
    + '_ptb'     + '{0:1.2f}'.format(ptb)    \
    + '_ns'      + '{0:1d}'.format(ns-2)     \
    + '/'
    
    U = np.load( saveString+'{0:04d}'.format(np.int(np.round(tf)))+'.npy' )
    s = np.load( saveString+'s'+'.npy' )
    
    return U[0,:,:], s

###########################################################################

#Load reference solution and things to compare it to:

def loadManyResults( ns0, ns1, ns2, ns3, ns4, ns5 \
, phs, pol, stc, ptb ) :
    
    U0, s0 = loadSingleResult( ns0, phs0, pol0, stc0, ptb0 )
    U1, s1 = loadSingleResult( ns1, phs,  pol,  stc,  ptb  )
    U2, s2 = loadSingleResult( ns2, phs,  pol,  stc,  ptb  )
    U3, s3 = loadSingleResult( ns3, phs,  pol,  stc,  ptb  )
    U4, s4 = loadSingleResult( ns4, phs,  pol,  stc,  ptb  )
    U5, s5 = loadSingleResult( ns5, phs,  pol,  stc,  ptb  )
    
    return U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5

###########################################################################

#Get regularly spaced theta levels:

def getThetaLevels( ns0, ns1, ns2, ns3, ns4, ns5 ) :

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

def getErrorVector( ns0, ns1, ns2, ns3, ns4, ns5, phs, pol, stc, ptb ) :

    U0, U1, U2, U3, U4, U5, s0, s1, s2, s3, s4, s5  \
    = loadManyResults( ns0, ns1, ns2, ns3, ns4, ns5, phs, pol, stc, ptb )

    th0, th1, th2, th3, th4, th5 = getThetaLevels( ns0, ns1, ns2, ns3, ns4, ns5 )

    U0, U1, U2, U3, U4, U5 = interpRadial( U0, U1, U2, U3, U4, U5 \
    , s0, s1, s2, s3, s4, s5, phs, pol, stc )

    U0, U1, U2, U3, U4, U5 = interpAngular( U0, U1, U2, U3, U4, U5 \
    , th0, th1, th2, th3, th4, th5 )

    err = gatherErrors( U0, U1, U2, U3, U4, U5 )
    
    return err, U0, U1, U2, U3, U4, U5, th0, s0

###########################################################################

#Plot the error to check convergence:

errA, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( ns0, ns1, ns2, ns3, ns4, ns5, phsA, polA, stcA, ptbA )

errB, U0, U1, U2, U3, U4, U5, th0, s0 \
= getErrorVector( ns0, ns1, ns2, ns3, ns4, ns5, phsB, polB, stcB, ptbB )

ns= np.hstack(( ns1, ns2, ns3, ns4, ns5 ))
ns = ns - 2

plt.plot( np.log(ns), np.log(errA), '-' )
plt.plot( np.log(ns), np.log(errB), '-' )
plt.plot( np.array([3.,5.]), np.array([-1.,-7.]),  '--' )
plt.plot( np.array([3.,5.]), np.array([-1.,-9.]),  '--' )
plt.plot( np.array([3.,5.]), np.array([-1.,-11.]), '--' )
plt.legend(( 'A', 'B', '3rd order', '4th order', '5th order' ))
plt.plot( np.log(ns), np.log(errA), 'k.' )
plt.plot( np.log(ns), np.log(errB), 'k.' )
plt.xlabel( 'log(ns)' )
plt.ylabel( 'log(relMaxNormErr)' )
plt.show()

###########################################################################

#Contour plot of differences at final time:

if contourErrors == 1 :

    rSurf, rSurfPrime = annulus.getTopoFunc( innerRadius, outerRadius, amp )

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