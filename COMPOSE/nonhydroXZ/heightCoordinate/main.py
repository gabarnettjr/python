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
t = 0.
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
    tf = 1500.
    dt = 1./2.
    rkStages = 4
else :
    sys.exit( "\nError: Invalid test case string.  Only ''bubble'' for now\n" )
nTimesteps = round( (tf-t) / dt )

#definition of the s coordinate and its derivatives:
def s( xTilde, zTilde ) :
    return ( zTop - zTilde ) / ( zTop - zSurf(xTilde) )
def dsdx( xTilde, zTilde ) :
    return ( zTop - zTilde ) * zSurfPrime(xTilde) / ( zTop - zSurf(xTilde) )**2
def dsdz( xTilde, zTilde ) :
    return -1 / ( zTop - zSurf(xTilde) )

#x and z as 2D arrays:
dx = ( xRight - xLeft ) / nCol
x = np.linspace( xLeft+1./2.*dx, xRight-1./2.*dx, nCol )
zs = zSurf(x)
dz = ( zTop - zs ) / nLev
xInterfaces = np.zeros(( nLev+1, nCol ))
zInterfaces = np.zeros(( nLev+1, nCol ))
for i in range(nLev+1) :
    xInterfaces[i,:] = x
    zInterfaces[i,:] = zs + i*dz
x = np.tile( x, (nLev,1) )
z = ( zInterfaces[0:nLev,:] + zInterfaces[1:nLev+1,:] ) / 2.
ds = 1. / nLev

#initial condition parameters:
U = np.zeros(( 4, nLev, nCol ))
if testCase == "bubble" :
    thetaBar = 300. * np.ones(( nLev, nCol ))
    piBar = 1. - g / Cp / thetaBar * z
    R = 1500.
    xc = 5000.
    zc = 3000.
    r = np.sqrt( (x-xc)**2 + (z-zc)**2 )
    ind = r < R
    thetaPrime0 = np. zeros( np.shape(r) )
    thetaPrime0[ind] = 2. * ( 1. - r[ind]/R )
    piPrime0 = 0.
    U[0,:,:] = np.zeros(( nLev, nCol ))
    U[1,:,:] = np.zeros(( nLev, nCol ))
    U[2,:,:] = thetaBar + thetaPrime0
    U[3,:,:] = -g / Rd / U[2,:,:] / dsdz(x,z) * Po * (piBar+piPrime0)**(Cv/Rd)
else :
    sys.exit("\nError: Invalid test case string.\n")

#convert functions to values on nodes:
dsdxInterfaces = dsdx( xInterfaces[1:nLev,:], zInterfaces[1:nLev,:] )
dsdzInterfaces = dsdz( xInterfaces[1:nLev,:], zInterfaces[1:nLev,:] )
dsdx = dsdx( x, z )
dsdz = dsdz( x, z )

sDot = np.zeros(( 3, nLev+1, nCol ))

def Dx(U) :
    c = [ 1./12., -2./3., 0., 2./3., -1./12. ]
    V = np.zeros( np.shape(U) )
    if np.shape(np.shape(U))[0] == 3 :
        V[:,:,0]        = c[0]*U[:,:,nCol-2]   + c[1]*U[:,:,nCol-1]   + c[2]*U[:,:,0]        + c[3]*U[:,:,1]        + c[4]*U[:,:,2]
        V[:,:,1]        = c[0]*U[:,:,nCol-1]   + c[1]*U[:,:,0]        + c[2]*U[:,:,1]        + c[3]*U[:,:,2]        + c[4]*U[:,:,3]
        V[:,:,2:nCol-2] = c[0]*U[:,:,0:nCol-4] + c[1]*U[:,:,1:nCol-3] + c[2]*U[:,:,2:nCol-2] + c[3]*U[:,:,3:nCol-1] + c[4]*U[:,:,4:nCol]
        V[:,:,nCol-2]   = c[0]*U[:,:,nCol-4]   + c[1]*U[:,:,nCol-3]   + c[2]*U[:,:,nCol-2]   + c[3]*U[:,:,nCol-1]   + c[4]*U[:,:,0]
        V[:,:,nCol-1]   = c[0]*U[:,:,nCol-3]   + c[1]*U[:,:,nCol-2]   + c[2]*U[:,:,nCol-1]   + c[3]*U[:,:,0]        + c[4]*U[:,:,1]
    elif np.shape(np.shape(U))[0] == 2 :
        V[:,0]        = c[0]*U[:,nCol-2]   + c[1]*U[:,nCol-1]   + c[2]*U[:,0]        + c[3]*U[:,1]        + c[4]*U[:,2]
        V[:,1]        = c[0]*U[:,nCol-1]   + c[1]*U[:,0]        + c[2]*U[:,1]        + c[3]*U[:,2]        + c[4]*U[:,3]
        V[:,2:nCol-2] = c[0]*U[:,0:nCol-4] + c[1]*U[:,1:nCol-3] + c[2]*U[:,2:nCol-2] + c[3]*U[:,3:nCol-1] + c[4]*U[:,4:nCol]
        V[:,nCol-2]   = c[0]*U[:,nCol-4]   + c[1]*U[:,nCol-3]   + c[2]*U[:,nCol-2]   + c[3]*U[:,nCol-1]   + c[4]*U[:,0]
        V[:,nCol-1]   = c[0]*U[:,nCol-3]   + c[1]*U[:,nCol-2]   + c[2]*U[:,nCol-1]   + c[3]*U[:,0]        + c[4]*U[:,1]
    else :
        sys.exit( "\nError: Invalid array dimensions.  U must be a 2D or 3D array.\n" )
    return V/dx

def HV(U) :
    c = [ 1., -4., 6., -4., 1. ] 
    V = np.zeros( np.shape(U) )
    V[:,:,0]        = c[0]*U[:,:,nCol-2]   + c[1]*U[:,:,nCol-1]   + c[2]*U[:,:,0]        + c[3]*U[:,:,1]        + c[4]*U[:,:,2]
    V[:,:,1]        = c[0]*U[:,:,nCol-1]   + c[1]*U[:,:,0]        + c[2]*U[:,:,1]        + c[3]*U[:,:,2]        + c[4]*U[:,:,3]
    V[:,:,2:nCol-2] = c[0]*U[:,:,0:nCol-4] + c[1]*U[:,:,1:nCol-3] + c[2]*U[:,:,2:nCol-2] + c[3]*U[:,:,3:nCol-1] + c[4]*U[:,:,4:nCol]
    V[:,:,nCol-2]   = c[0]*U[:,:,nCol-4]   + c[1]*U[:,:,nCol-3]   + c[2]*U[:,:,nCol-2]   + c[3]*U[:,:,nCol-1]   + c[4]*U[:,:,0]
    V[:,:,nCol-1]   = c[0]*U[:,:,nCol-3]   + c[1]*U[:,:,nCol-2]   + c[2]*U[:,:,nCol-1]   + c[3]*U[:,:,0]        + c[4]*U[:,:,1]
    return -1/12/dx * V

def verticalOperations( U, sDot ) :
    #get sDot on all interfaces (set it to zero on both boundaries):
    uInterfaces = ( U[0,0:nLev-1,:] + U[0,1:nLev,:] ) / 2.
    wInterfaces = ( U[1,0:nLev-1,:] + U[1,1:nLev,:] ) / 2.
    sDot[:,1:nLev,:] = np.tile( uInterfaces * dsdxInterfaces + wInterfaces * dsdzInterfaces, (3,1,1) )
    #get Us on non-boundary interfaces:
    Us = ( U[0:3,1:nLev,:] - U[0:3,0:nLev-1,:] ) / ds
    #get sDot*Us on all interfaces, then average to get values at layer midpts:
    sDotUs = sDot
    sDotUs[:,1:nLev,:] = sDotUs[:,1:nLev,:] * Us
    sDotUs = ( sDot[:,0:nLev,:] + sDot[:,1:nLev+1,:] ) / 2.
    #average to get sDot*dpids on interfaces, then use FD to get (sDot*dpids)_s:
    sDotDpids = np.squeeze( sDot[0,:,:] )
    sDotDpids[1:nLev,:] = sDotDpids[1:nLev,:] * ( U[3,0:nLev-1,:] + U[3,1:nLev,:] ) / 2
    sDotDpids_s = ( sDotDpids[1:nLev+1,:] - sDotDpids[0:nLev,:] ) / ds
    #EOS to get P on mid-levels:
    P = Po**(-Rd/Cv) * ( -Rd*U[2,:,:]/g * U[3,:,:] * dsdz ) ** (Cp/Cv)
    #average/extrapolate to get P on interfaces:
    Pinterfaces = np.zeros(( nLev+1, nCol ))
    Pinterfaces[1:nLev,:] = ( P[0:nLev-1,:] + P[1:nLev] ) / 2.
    Pinterfaces[0,:] = 2*P[0,:] - Pinterfaces[1,:]
    Pinterfaces[nLev,:] = 2*P[nLev-1,:] - Pinterfaces[nLev-1,:]
    dpds = ( Pinterfaces[1:nLev+1,:] - Pinterfaces[0:nLev,:] ) / ds
    return sDotUs, sDotDpids_s, P, dpds

def odefun( t, U ) :
    sDotUs, sDotDpids_s, P, dpds = verticalOperations( U, sDot )
    #initialize RHS of system of ODEs:
    V = np.zeros(( 4, nLev, nCol ))
    #approximate d/dx of u,w,th:
    V[0:3,:,:] = -np.tile(U[0,:,:],(3,1,1)) * Dx(U[0:3,:,:]) -sDotUs
    V[0,:,:] = V[0,:,:] + g/U[3,:,:]/dsdz * ( Dx(P) + dpds*dsdx )
    V[1,:,:] = V[1,:,:] - g * ( 1. - dpds/U[3,:,:] )
    V[3,:,:] = -Dx(U[3,:,:]*U[0,:,:]) - sDotDpids_s
    V = V + HV(U)
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


































