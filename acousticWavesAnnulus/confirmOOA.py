import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( '../site-packages' )

from gab import phs1
from gab.pseudospectral import periodic
from gab.acousticWaveEquation import annulus

###########################################################################

ord = np.inf                                   #norm to use for error check

innerRadius = 1.
outerRadius = 2.
tf          = 100.

phs = 5
pol = 3
stc = 7
ptb = .30

ns0 = 256+2                                          #reference resolution

phs0 = 7
pol0 = 6
stc0 = 7
ptb0 = .00

ns1 = 16+2
ns2 = 2*(ns1-2)+2
ns3 = 2*(ns2-2)+2
ns4 = 2*(ns3-2)+2

###########################################################################

def loadResult( ns, phs, pol, stc, ptb, tf ) :
    saveString = './results/'     \
    + 'ns'   + '{0:1d}'.format(ns-2)   \
    + '_phs' + '{0:1d}'.format(phs)    \
    + '_pol' + '{0:1d}'.format(pol)    \
    + '_stc' + '{0:1d}'.format(stc)    \
    + '_ptb' + '{0:1.2f}'.format(ptb)  \
    + '/'
    U = np.load( saveString+'{0:04d}'.format(np.int(np.round(tf)))+'.npy' )
    s = np.load( saveString+'s'+'.npy' )
    return U[0,:,:], s

###########################################################################

#Load reference solution and things to compare it to:

U0, s0 = loadResult( ns0, phs0, pol0, stc0, ptb0, tf )
U1, s1 = loadResult( ns1, phs,  pol,  stc,  ptb,  tf )
U2, s2 = loadResult( ns2, phs,  pol,  stc,  ptb,  tf )
U3, s3 = loadResult( ns3, phs,  pol,  stc,  ptb,  tf )
U4, s4 = loadResult( ns4, phs,  pol,  stc,  ptb,  tf )

###########################################################################

#Get regularly spaced theta levels:

nth0 = annulus.getNth( innerRadius, outerRadius, ns0 )
nth1 = annulus.getNth( innerRadius, outerRadius, ns1 )
nth2 = annulus.getNth( innerRadius, outerRadius, ns2 )
nth3 = annulus.getNth( innerRadius, outerRadius, ns3 )
nth4 = annulus.getNth( innerRadius, outerRadius, ns4 )

th0 = np.linspace( 0, 2.*np.pi, nth0+1 )
th1 = np.linspace( 0, 2.*np.pi, nth1+1 )
th2 = np.linspace( 0, 2.*np.pi, nth2+1 )
th3 = np.linspace( 0, 2.*np.pi, nth3+1 )
th4 = np.linspace( 0, 2.*np.pi, nth4+1 )

th0 = th0[0:-1]
th1 = th1[0:-1]
th2 = th2[0:-1]
th3 = th3[0:-1]
th4 = th4[0:-1]

###########################################################################

#Interpolate radially to reference grid using PHS-FD:

W0 = phs1.getDM( x=s0, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
W1 = phs1.getDM( x=s1, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
W2 = phs1.getDM( x=s2, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
W3 = phs1.getDM( x=s3, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )
W4 = phs1.getDM( x=s4, X=s0[1:-1], m=0, phsDegree=phs, polyDegree=pol, stencilSize=stc )

U0 = W0.dot( U0 )
U1 = W1.dot( U1 )
U2 = W2.dot( U2 )
U3 = W3.dot( U3 )
U4 = W4.dot( U4 )

###########################################################################

#Interpolate angularly to reference grid using periodic pseudospectral:

W0 = periodic.getDM( th=th0, TH=th0, m=0 )
W1 = periodic.getDM( th=th1, TH=th0, m=0 )
W2 = periodic.getDM( th=th2, TH=th0, m=0 )
W3 = periodic.getDM( th=th3, TH=th0, m=0 )
W4 = periodic.getDM( th=th4, TH=th0, m=0 )

U0 = np.transpose( W0.dot(np.transpose(U0)) )
U1 = np.transpose( W1.dot(np.transpose(U1)) )
U2 = np.transpose( W2.dot(np.transpose(U2)) )
U3 = np.transpose( W3.dot(np.transpose(U3)) )
U4 = np.transpose( W4.dot(np.transpose(U4)) )

###########################################################################

#Plot the error:

err1 = np.linalg.norm( U1 - U0, ord )
err2 = np.linalg.norm( U2 - U0, ord )
err3 = np.linalg.norm( U3 - U0, ord )
err4 = np.linalg.norm( U4 - U0, ord )
err = np.hstack(( err1, err2, err3, err4 ))
err = err / np.linalg.norm(U0,ord)

ns= np.hstack(( ns1, ns2, ns3, ns4 ))
ns = ns - 2

plt.plot( np.log(ns), np.log(err), '-' )
plt.plot( np.log(ns), np.log(err), '.' )
plt.plot( np.array([3.,5.]), np.array([-1.,-9.]), '-' )
# plt.axis( 'equal' )
plt.xlabel( 'log(ns)' )
plt.ylabel( 'log(relMaxNormErr)' )
plt.show()

###########################################################################