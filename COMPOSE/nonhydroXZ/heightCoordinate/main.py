import numpy as np

#parameters for the test case:
xLeft = 0.
xRight = 10000.
n = 100
nLev = 100
dx = ( xRight - xLeft ) / n
def zSurf(xTilde) :
    return 500. * ( 1. + np.sin( 2.*np.pi*xTilde / 5000. ) )
def zSurfPrime(xTilde) :
    return np.pi/5. * np.cos( 2.*np.pi*xTilde / 5000. )
zTop = 10000.

#definition of the new coordinate s, and its derivatives:
def s( xTilde, zTilde ) :
    return ( zTop - zTilde ) / ( zTop - zSurf(xTilde) )
def s_xTilde( xTilde, zTilde ) :
    return ( zTop - zTilde ) * zSurfPrime(xTilde) / ( zTop - zSurf(xTilde) )**2
def s_zTilde( xTilde, zTilde ) :
    return -1 / ( zTop - zSurf(xTilde) )

#x and z as 2D arrays:
x = np.linspace( xLeft-3./2.*dx, xRight+3./2.*dx, n+4 )
zs = zSurf(x)
dz = ( zTop - zs ) / nLev
z = np.zeros(( nLev, n+4 ))
for i in range(nLev) :
    z[i,:] = zs + dz/2. + i*dz
x = np.tile( x, (nLev,1) )

