import sys
import numpy as np
import time
import matplotlib.pyplot as plt

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
        # return np.zeros( np.shape(xTilde) )
    def zSurfPrime(xTilde) :
        return np.pi/5. * np.cos( 2.*np.pi*xTilde / 5000. )
        # return np.zeros( np.shape(xTilde) )
    zTop = 10000.
    tf = 1500.
    dt = 1./8.
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
x = np.linspace( xLeft+dx/2., xRight-dx/2., nCol )
zs = zSurf(x)
dz = ( zTop - zs ) / nLev
xInterfaces = np.zeros(( nLev+1, nCol ))
zInterfaces = np.zeros(( nLev+1, nCol ))
for i in range(nLev+1) :
    xInterfaces[i,:] = x
    zInterfaces[i,:] = zs + i*dz
x = np.tile( x, (nLev,1) )
z = ( zInterfaces[0:nLev,:] + zInterfaces[1:nLev+1,:] ) / 2.
ds = - 1. / nLev

#initial condition parameters:
U = np.zeros(( 4, nLev, nCol ))
if testCase == "bubble" :
    thetaBar = 300. * np.ones(( nLev, nCol ))
    piBar = 1. - g / Cp / thetaBar * z
    dpidsBar = -g / Rd / thetaBar / dsdz(x,z) * Po * piBar**(Cv/Rd)
    Pbar = Po**(-Rd/Cv) * ( -Rd*thetaBar/g * dpidsBar * dsdz(x,z) ) ** (Cp/Cv)
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
dsdxBottom = dsdx( xInterfaces[0,:], zInterfaces[0,:] )
dsdzBottom = dsdz( xInterfaces[0,:], zInterfaces[0,:] )
dsdx = dsdx( x, z )
dsdz = dsdz( x, z )

sDot = np.zeros(( 3, nLev+1, nCol ))

wx = [ 1./12., -2./3., 0., 2./3., -1./12. ]
whv = [ 1., -4., 6., -4., 1. ]

def Dx(U) :
    V = np.zeros( np.shape(U) )
    if np.shape(np.shape(U))[0] == 3 :
        V[:,:,0]        = wx[0]*U[:,:,nCol-2]   + wx[1]*U[:,:,nCol-1]   + wx[2]*U[:,:,0]        + wx[3]*U[:,:,1]        + wx[4]*U[:,:,2]
        V[:,:,1]        = wx[0]*U[:,:,nCol-1]   + wx[1]*U[:,:,0]        + wx[2]*U[:,:,1]        + wx[3]*U[:,:,2]        + wx[4]*U[:,:,3]
        V[:,:,2:nCol-2] = wx[0]*U[:,:,0:nCol-4] + wx[1]*U[:,:,1:nCol-3] + wx[2]*U[:,:,2:nCol-2] + wx[3]*U[:,:,3:nCol-1] + wx[4]*U[:,:,4:nCol]
        V[:,:,nCol-2]   = wx[0]*U[:,:,nCol-4]   + wx[1]*U[:,:,nCol-3]   + wx[2]*U[:,:,nCol-2]   + wx[3]*U[:,:,nCol-1]   + wx[4]*U[:,:,0]
        V[:,:,nCol-1]   = wx[0]*U[:,:,nCol-3]   + wx[1]*U[:,:,nCol-2]   + wx[2]*U[:,:,nCol-1]   + wx[3]*U[:,:,0]        + wx[4]*U[:,:,1]
    elif np.shape(np.shape(U))[0] == 2 :
        V[:,0]        = wx[0]*U[:,nCol-2]   + wx[1]*U[:,nCol-1]   + wx[2]*U[:,0]        + wx[3]*U[:,1]        + wx[4]*U[:,2]
        V[:,1]        = wx[0]*U[:,nCol-1]   + wx[1]*U[:,0]        + wx[2]*U[:,1]        + wx[3]*U[:,2]        + wx[4]*U[:,3]
        V[:,2:nCol-2] = wx[0]*U[:,0:nCol-4] + wx[1]*U[:,1:nCol-3] + wx[2]*U[:,2:nCol-2] + wx[3]*U[:,3:nCol-1] + wx[4]*U[:,4:nCol]
        V[:,nCol-2]   = wx[0]*U[:,nCol-4]   + wx[1]*U[:,nCol-3]   + wx[2]*U[:,nCol-2]   + wx[3]*U[:,nCol-1]   + wx[4]*U[:,0]
        V[:,nCol-1]   = wx[0]*U[:,nCol-3]   + wx[1]*U[:,nCol-2]   + wx[2]*U[:,nCol-1]   + wx[3]*U[:,0]        + wx[4]*U[:,1]
    elif np.shape(np.shape(U))[0] == 1 :
        V[0]        = wx[0]*U[nCol-2]   + wx[1]*U[nCol-1]   + wx[2]*U[0]        + wx[3]*U[1]        + wx[4]*U[2]
        V[1]        = wx[0]*U[nCol-1]   + wx[1]*U[0]        + wx[2]*U[1]        + wx[3]*U[2]        + wx[4]*U[3]
        V[2:nCol-2] = wx[0]*U[0:nCol-4] + wx[1]*U[1:nCol-3] + wx[2]*U[2:nCol-2] + wx[3]*U[3:nCol-1] + wx[4]*U[4:nCol]
        V[nCol-2]   = wx[0]*U[nCol-4]   + wx[1]*U[nCol-3]   + wx[2]*U[nCol-2]   + wx[3]*U[nCol-1]   + wx[4]*U[0]
        V[nCol-1]   = wx[0]*U[nCol-3]   + wx[1]*U[nCol-2]   + wx[2]*U[nCol-1]   + wx[3]*U[0]        + wx[4]*U[1]
    else :
        sys.exit( "\nError: Invalid array dimensions.  U must be a 1D, 2D, or 3D array.\n" )
    return V/dx

def HV(U) :
    V = np.zeros( np.shape(U) )
    V[:,:,0]        = whv[0]*U[:,:,nCol-2]   + whv[1]*U[:,:,nCol-1]   + whv[2]*U[:,:,0]        + whv[3]*U[:,:,1]        + whv[4]*U[:,:,2]
    V[:,:,1]        = whv[0]*U[:,:,nCol-1]   + whv[1]*U[:,:,0]        + whv[2]*U[:,:,1]        + whv[3]*U[:,:,2]        + whv[4]*U[:,:,3]
    V[:,:,2:nCol-2] = whv[0]*U[:,:,0:nCol-4] + whv[1]*U[:,:,1:nCol-3] + whv[2]*U[:,:,2:nCol-2] + whv[3]*U[:,:,3:nCol-1] + whv[4]*U[:,:,4:nCol]
    V[:,:,nCol-2]   = whv[0]*U[:,:,nCol-4]   + whv[1]*U[:,:,nCol-3]   + whv[2]*U[:,:,nCol-2]   + whv[3]*U[:,:,nCol-1]   + whv[4]*U[:,:,0]
    V[:,:,nCol-1]   = whv[0]*U[:,:,nCol-3]   + whv[1]*U[:,:,nCol-2]   + whv[2]*U[:,:,nCol-1]   + whv[3]*U[:,:,0]        + whv[4]*U[:,:,1]
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
    sDotUs = ( sDotUs[:,0:nLev,:] + sDotUs[:,1:nLev+1,:] ) / 2.
    #average to get sDot*dpids on interfaces, then use FD to get (sDot*dpids)_s:
    sDotDpids = np.squeeze( sDot[0,:,:] )
    sDotDpids[1:nLev,:] = sDotDpids[1:nLev,:] * ( U[3,0:nLev-1,:] + U[3,1:nLev,:] ) / 2
    sDotDpids_s = ( sDotDpids[1:nLev+1,:] - sDotDpids[0:nLev,:] ) / ds
    #EOS to get P on mid-levels:
    P = Po**(-Rd/Cv) * ( -Rd*U[2,:,:]/g * U[3,:,:] * dsdz ) ** (Cp/Cv)
    #get dpids and dpdx on bottom interface:
    dpids = 3./2.*U[3,0,:] - 1./2.*U[3,1,:]
    dpdx = Dx( P[0:2,:] )
    dpdx = 3./2.*dpdx[0,:] - 1./2.*dpdx[1,:]
    #initialize dpds array:
    dpds = np.zeros(( nLev+1, nCol ))
    #get dpds on non-boundary interfaces:
    dpds[1:nLev,:] = ( P[1:nLev,:] - P[0:nLev-1,:] ) / ds
    #get dpds on bottom interface:
    normGradS = np.sqrt( dsdxBottom**2 + dsdzBottom**2 )
    dpds[0,:] = 1/normGradS**2 * ( dpids*dsdzBottom**2 - dpdx*dsdxBottom )
    #extrapolate to get dpds(=dpids) on top interface (flat):
    dpds[nLev,:] = 3./2.*U[3,nLev-1,:] - 1./2.*U[3,nLev-2,:]
    #average to get dpds on mid-levels:
    dpds = ( dpds[0:nLev,:] + dpds[1:nLev+1,:] ) / 2.
    return sDotUs, sDotDpids_s, P, dpds

def odefun( t, U ) :
    sDotUs, sDotDpids_s, P, dpds = verticalOperations( U, sDot )
    #initialize RHS of system of ODEs:
    V = np.zeros(( 4, nLev, nCol ))
    #advective part that the first three equations all have in common:
    V[0:3,:,:] = -np.tile(U[0,:,:],(3,1,1)) * Dx(U[0:3,:,:]) - sDotUs
    #extra stuff for u:
    V[0,:,:] = V[0,:,:] + g/U[3,:,:]/dsdz * ( Dx(P) + dpds*dsdx )
    #extra stuff for w:
    V[1,:,:] = V[1,:,:] - g * ( 1. - dpds/U[3,:,:] )
    #entire RHS for dpids:
    V[3,:,:] = -Dx(U[3,:,:]*U[0,:,:]) - sDotDpids_s
    #add hyperviscosity:
    V = V + HV(U)
    return V

def rk( t, U ) :
    if rkStages == 4 :
        q1 = odefun( t, U )
        q2 = odefun( t+dt/2, U+dt/2*q1 )
        q3 = odefun( t+dt/2, U+dt/2*q2 )
        q4 = odefun( t+dt, U+dt*q3 )
        return U + dt/6 * ( q1 + 2*q2 + 2*q3 + q4 )
    elif rkStages == 3 :
        q1 = odefun( t, U );
        q2 = odefun( t+dt/3, U+dt/3*q1 );
        q2 = odefun( t+2*dt/3, U+2*dt/3*q2 );
        return U + dt/4 * ( q1 + 3*q2 );
    else :
        sys.exit( "\nError: rkStages should be 3 or 4.\n" )

print()
print("t =",t)       
print( [ np.min(U[0,:,:]), np.max(U[0,:,:]) ] )
print( [ np.min(U[1,:,:]), np.max(U[1,:,:]) ] )
print( [ np.min(U[2,:,:]), np.max(U[2,:,:]) ] )
print( [ np.min(U[3,:,:]), np.max(U[3,:,:]) ] )
rho = -1/g*U[3,:,:]*dsdz
print( [ np.min(rho), np.max(rho) ] )
P = Po**(-Rd/Cv) * ( -Rd*U[2,:,:]/g * U[3,:,:] * dsdz ) ** (Cp/Cv)
print( [ np.min(P), np.max(P) ] )
print()

#stepping forward in time with explicit RK:
plt.ion()
for i in range(nTimesteps) :
    
    # plt.contourf( x, z, np.squeeze(U[3,:,:])-dpidsBar )
    plt.contourf( x, z, np.squeeze(Po**(-Rd/Cv)*(-Rd*U[2,:,:]/g*U[3,:,:]*dsdz)**(Cp/Cv)) - Pbar )
    plt.colorbar()
    plt.axis( 'equal' )
    plt.waitforbuttonpress()
    plt.clf()
    
    U = rk( t, U )
    t = t + dt
    
    print( "t =", t )
    print( [ np.min(U[0,:,:]), np.max(U[0,:,:]) ] )
    print( [ np.min(U[1,:,:]), np.max(U[1,:,:]) ] )
    print( [ np.min(U[2,:,:]), np.max(U[2,:,:]) ] )
    print( [ np.min(U[3,:,:]), np.max(U[3,:,:]) ] )
    # rho = -1/g*U[3,:,:]*dsdz
    # print( [ np.min(rho), np.max(rho) ] )
    # P = Po**(-Rd/Cv) * ( -Rd*U[2,:,:]/g * U[3,:,:] * dsdz ) ** (Cp/Cv)
    # print( [ np.min(P), np.max(P) ] )
    print()


































