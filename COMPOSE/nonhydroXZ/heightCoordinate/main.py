import sys
import numpy as np

testCase = "bubble"

#atmospheric constants:
Cp = 1004.
Cv = 717.
Rd = Cp - Cv
g = 9.81
Po = 10.**5.

#domain parameters:
if testCase == "bubble" :
    xLeft = 0.
    xRight = 10000.
    nCol = 50
    nLev = 50
    def zSurf(xTilde) :
        return 500. * ( 1. + np.sin( 2.*np.pi*xTilde / 5000. ) )
    def zSurfPrime(xTilde) :
        return np.pi/5. * np.cos( 2.*np.pi*xTilde / 5000. )
    zTop = 10000.
    t = 0.
    dt = 1./2.
    nTimesteps = 3000
    rkStages = 4
else :
    sys.exit( "\nError: Invalid test case string.\n" )

#definition of the s coordinate and its derivatives:
def s( xTilde, zTilde ) :
    return ( zTop - zTilde ) / ( zTop - zSurf(xTilde) )
def dsdx( xTilde, zTilde ) :
    return ( zTop - zTilde ) * zSurfPrime(xTilde) / ( zTop - zSurf(xTilde) )**2
def dsdz( xTilde, zTilde ) :
    return -1 / ( zTop - zSurf(xTilde) )

#x and z as 2D arrays:
dx = ( xRight - xLeft ) / nCol
x = np.linspace( xLeft-3./2.*dx, xRight+3./2.*dx, nCol+4 )
zs = zSurf(x)
dz = ( zTop - zs ) / nLev
xInterfaces = np.zeros(( nLev+1, nCol+4 ))
zInterfaces = np.zeros(( nLev+1, nCol+4 ))
for i in range(nLev+1) :
    xInterfaces[i,:] = x
    zInterfaces[i,:] = zs + i*dz
x = np.tile( x, (nLev,1) )
z = ( zInterfaces[0:nLev,:] + zInterfaces[1:nLev+1,:] ) / 2.
ds = 1. / nLev

#initial condition parameters:
U = np.zeros(( 4, nLev, nCol+4 ))
if testCase == "bubble" :
    thetaBar = 300. * np.ones(( nLev, nCol+4 ))
    piBar = 1. - g / Cp / thetaBar * z
    R = 1500.
    xc = 5000.
    zc = 3000.
    r = np.sqrt( (x-xc)**2 + (z-zc)**2 )
    ind = r < R
    thetaPrime0 = np. zeros( np.shape(r) )
    thetaPrime0[ind] = 2. * ( 1. - r[ind]/R )
    piPrime0 = 0.
    U[0,:,:] = np.zeros(( nLev, nCol+4 ))
    U[1,:,:] = np.zeros(( nLev, nCol+4 ))
    U[2,:,:] = thetaBar + thetaPrime0
    # U[3,:,:] = piBar + piPrime0;
    U[3,:,:] = -g / Rd / U[2,:,:] / dsdz(x,z) * Po * (piBar+piPrime0)**(Cv/Rd)
else :
    sys.exit("\nError: Invalid test case string.\n")

#convert functions to values on nodes:
dsdx = dsdx( xInterfaces[1:nLev,2:nCol+2], zInterfaces[1:nLev,2:nCol+2] )
dsdz = dsdz( xInterfaces[1:nLev,2:nCol+2], zInterfaces[1:nLev,2:nCol+2] )

sDot = np.zeros(( 3, nLev+1, nCol ))

def setGhostNodes( U, sDot ) :
    #get sDot on all interfaces (set it to zero on both boundaries):
    uInterfaces = ( U[0,0:nLev-1,2:nCol+2] + U[0,1:nLev,2:nCol+2] ) / 2.
    wInterfaces = ( U[1,0:nLev-1,2:nCol+2] + U[1,1:nLev,2:nCol+2] ) / 2.
    sDot[:,1:nLev,:] = np.tile( uInterfaces * dsdx + wInterfaces * dsdz, (3,1,1) )
    #get Us on non-boundary interfaces:
    Us = ( U[0:3,1:nLev,2:nCol+2] - U[0:3,0:nLev-1,2:nCol+2] ) / ds
    #get sDot*Us on all interfaces, then average to get values at layer midpts:
    sDot[:,1:nLev,:] = sDot[:,1:nLev,:] * Us
    sDotUs = ( sDot[:,0:nLev,:] + sDot[:,1:nLev+1,:] ) / 2.
    #periodic laterally in every variable:
    U[:,:,0:2] = U[:,:,nCol:nCol+2]
    U[:,:,nCol+2:nCol+4] = U[:,:,2:4]
    return U

def odefun( t, U ) :
    U = setGhostNodes( U, sDot )
    V = np.zeros(( 4, nLev, nCol+4 ))
    Ux = 1/dx * ( 1./12.*U[0:3,:,0:nCol] - 2./3.*U[0:3,:,1:nCol+1] + 2./3.*U[0:3,:,3:nCol+3] - 1./12.*U[0:3,:,4:nCol+4] )
    return V

def rk( t, U ) :
    if rkStages == 4 :
        q1 = odefun( t, U )
        q2 = odefun( t+dt/2, U+dt/2*q1 )
        q3 = odefun( t+dt/2, U+dt/2*q2 )
        q4 = odefun( t+dt, U+dt*q3 )
        return U + dt/6 * ( q1 + 2*q2 + 2*q3 + q4 )
    else :
        sys.exit( "\nError: rkStages should be 4.\n" )

#stepping forward in time with explicit RK:
for i in range(nTimesteps) :
    U = rk( t, U )
    t = t + dt


































