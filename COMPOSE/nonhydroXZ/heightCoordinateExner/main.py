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
    kap = 10.
    def zSurf(xTilde) :
        # return 1000. * np.exp( -(kap*(xTilde-6000.)/(xRight-xLeft))**2 )
        return 500. * ( 1. + np.sin( 2.*np.pi*xTilde / 5000. ) )
        # return np.zeros( np.shape(xTilde) )
    def zSurfPrime(xTilde) :
        # return -2. * ( kap*(xTilde-6000.)/(xRight-xLeft) ) * kap/(xRight-xLeft) * zSurf(xTilde)
        return np.pi/5. * np.cos( 2.*np.pi*xTilde / 5000. )
        # return np.zeros( np.shape(xTilde) )
    zTop = 10000.
    tf = 1500.
    dt = 1./3.
    rkStages = 3
else :
    sys.exit( "\nError: Invalid test case string.  Only ''bubble'' for now\n" )
nTimesteps = round( (tf-t) / dt )

#definition of the scale-preserving s coordinate and its derivatives:
def s( xTilde, zTilde ) :
    return ( zTilde - zSurf(xTilde) ) / ( zTop - zSurf(xTilde) ) * zTop
def dsdx( xTilde, zTilde ) :
    return ( zTilde - zTop ) * zSurfPrime(xTilde) / ( zTop - zSurf(xTilde) )**2 * zTop
def dsdz( xTilde, zTilde ) :
    return zTop / ( zTop - zSurf(xTilde) )

#x and z as 2D arrays:
dx = ( xRight - xLeft ) / nCol
x = np.linspace( xLeft+dx/2., xRight-dx/2., nCol )
zs = zSurf(x)
dz = ( zTop - zs ) / nLev
z = np.zeros(( nLev+2, nCol ))
z[0,:] = zs - dz/2
for i in np.arange(1,nLev+1) :
    z[i,:] = zs + dz/2 + (i-1)*dz
z[nLev+1,:] = zTop + dz/2
x = np.tile( x, (nLev+2,1) )
ds = zTop * dz / ( zTop - zSurf(x[0,:]) )
ds = ds[0]

#get tangent and normal vectors at surface:
Tx = np.ones(( 1, nCol ))
Tz = zSurfPrime( x[0,:] )
normT = np.sqrt( Tx**2 + Tz**2 )
Tx = Tx / normT
Tz = Tz / normT
Nx = -Tz
Nz = Tx

#initial condition parameters:
U = np.zeros(( 4, nLev+2, nCol ))
if testCase == "bubble" :
    thetaBar = 300. * np.ones(( nLev+2, nCol ))
    piBar = 1. - g / Cp / thetaBar * z
    R = 1500.
    xc = 5000.
    zc = 3000.
    r = np.sqrt( (x-xc)**2 + (z-zc)**2 )
    ind = r < R
    thetaPrime0 = np. zeros( np.shape(r) )
    thetaPrime0[ind] = 2. * ( 1. - r[ind]/R )
    piPrime0 = 0.
    U[0,:,:] = np.zeros(( nLev+2, nCol ))
    U[1,:,:] = np.zeros(( nLev+2, nCol ))
    U[2,:,:] = thetaBar + thetaPrime0
    U[3,:,:] = piBar + piPrime0
else :
    sys.exit("\nError: Invalid test case string.\n")

#convert functions to values on nodes:
dsdxBottom = dsdx( x[0,:], zSurf(x[0,:]) )
dsdzBottom = dsdz( x[0,:], zSurf(x[0,:]) )
normGradS = np.sqrt( dsdxBottom**2 + dsdzBottom**2 )
dsdx = dsdx( x[1:nLev+1,:], z[1:nLev+1,:] )
dsdz = dsdz( x[1:nLev+1,:], z[1:nLev+1,:] )

wx = [ 1./12., -2./3., 0., 2./3., -1./12. ]
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

ws = [ -1./2., 0, 1./2. ]
def Ds(U) :
    return ( ws[0]*U[:,0:nLev,:] + ws[1]*U[:,1:nLev+1,:] + ws[2]*U[:,2:nLev+2,:] ) / ds

wxhv = [ 1., -4., 6., -4., 1. ]
def HVx(U) :
    V = np.zeros( np.shape(U) )
    V[:,:,0]        = wxhv[0]*U[:,:,nCol-2]   + wxhv[1]*U[:,:,nCol-1]   + wxhv[2]*U[:,:,0]        + wxhv[3]*U[:,:,1]        + wxhv[4]*U[:,:,2]
    V[:,:,1]        = wxhv[0]*U[:,:,nCol-1]   + wxhv[1]*U[:,:,0]        + wxhv[2]*U[:,:,1]        + wxhv[3]*U[:,:,2]        + wxhv[4]*U[:,:,3]
    V[:,:,2:nCol-2] = wxhv[0]*U[:,:,0:nCol-4] + wxhv[1]*U[:,:,1:nCol-3] + wxhv[2]*U[:,:,2:nCol-2] + wxhv[3]*U[:,:,3:nCol-1] + wxhv[4]*U[:,:,4:nCol]
    V[:,:,nCol-2]   = wxhv[0]*U[:,:,nCol-4]   + wxhv[1]*U[:,:,nCol-3]   + wxhv[2]*U[:,:,nCol-2]   + wxhv[3]*U[:,:,nCol-1]   + wxhv[4]*U[:,:,0]
    V[:,:,nCol-1]   = wxhv[0]*U[:,:,nCol-3]   + wxhv[1]*U[:,:,nCol-2]   + wxhv[2]*U[:,:,nCol-1]   + wxhv[3]*U[:,:,0]        + wxhv[4]*U[:,:,1]
    return -1./12. * V / dx

wshv = [ 1., -2., 1 ]
def HVs(U) :
    return 1./2. * ( wshv[0]*U[:,0:nLev,:] + wshv[1]*U[:,1:nLev+1,:] + wshv[2]*U[:,2:nLev+2,:] ) / ds

def setGhostNodes( U ) :
    #extrapolate uT to bottom ghost nodes:
    uT = U[0,1:3,:]*np.vstack((Tx,Tx)) + U[1,1:3,:]*np.vstack((Tz,Tz))
    uT = 2*uT[0,:] - uT[1,:]
    #get uN on bottom ghost nodes:
    uN = U[0,1,:]*Nx + U[1,1,:]*Nz
    uN = -uN
    #use uT and uN to get u and w on bottom ghost nodes:
    U[0,0,:] = uT*Tx + uN*Nx
    U[1,0,:] = uT*Tz + uN*Nz
    #extrapolate theta to bottom ghost nodes:
    U[2,0,:] = 2*U[2,1,:] - U[2,2,:]
    #get pi on bottom ghost nodes using derived BC:
    dpidx = Dx( U[3,1:3,:] )
    dpidx = 2*dpidx[0,:] - dpidx[1,:]
    th = ( U[2,0,:] + U[2,1,:] ) / 2.
    U[3,0,:] = U[3,1,:] + ds/normGradS**2 * ( g/Cp/th*dsdzBottom + dpidx*dsdxBottom )
    #extrapolate u to top ghost nodes:
    U[0,nLev+1,:] = 2*U[0,nLev,:] - U[0,nLev-1,:]
    #get w on top ghost nodes:
    U[1,nLev+1,:] = -U[1,nLev,:]
    #extrapolate theta to top ghost nodes:
    U[2,nLev+1,:] = 2*U[2,nLev,:] - U[2,nLev-1,:]
    #get pi on top ghost nodes:
    th = ( U[2,nLev,:] + U[2,nLev+1,:] ) / 2.
    U[3,nLev+1,:] = U[3,nLev,:] - ds/dsdzBottom*g/Cp/th
    return U

def odefun( t, U ) :
    V = np.zeros( np.shape(U) )
    #set ghost node values for all variables:
    U = setGhostNodes( U )
    #get Us and vertical dissipation, then remove ghost nodes from U (no longer needed):
    Us = Ds(U)
    tmp = HVs(U)
    U = U[:,1:nLev+1,:]
    #get RHS of ode function:
    Ux = Dx(U)
    sDot = U[0,:,:]*dsdx + U[1,:,:]*dsdz
    tmp = tmp - np.tile(U[0,:,:],(4,1,1)) * Ux - np.tile(sDot,(4,1,1)) * Us
    tmp[0,:,:] = tmp[0,:,:] - Cp*U[2,:,:] * ( Ux[3,:,:] + Us[3,:,:]*dsdx )
    tmp[1,:,:] = tmp[1,:,:] - Cp*U[2,:,:] * Us[3,:,:]*dsdz - g
    tmp[3,:,:] = tmp[3,:,:] - Rd/Cv*U[3,:,:] * ( Ux[0,:,:]+Us[0,:,:]*dsdx + Us[1,:,:]*dsdz )
    #apply horizontal dissipation:
    tmp = tmp + HVx(U)
    #set output:
    V[:,1:nLev+1,:] = tmp
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

#stepping forward in time with explicit RK:
plt.ion()
print()
for i in range(nTimesteps+1) :
    if np.mod( i, np.round(100./dt) ) == 0 :
        U = setGhostNodes(U)
        print( "t =", t )
        print( [ np.min(U[0,:,:]), np.max(U[0,:,:]) ] )
        print( [ np.min(U[1,:,:]), np.max(U[1,:,:]) ] )
        print( [ np.min(U[2,:,:]-thetaBar), np.max(U[2,:,:]-thetaBar) ] )
        print( [ np.min(U[3,:,:]-piBar), np.max(U[3,:,:]-piBar) ] )
        print()
        plt.contourf( x, z, np.squeeze(U[3,:,:]) - piBar )
        plt.colorbar()
        plt.axis( 'equal' )
        if i == nTimesteps :
            plt.waitforbuttonpress()
            sys.exit( "\nDone.\n" )
        else :
            plt.pause(.1)
        plt.clf()
    U = rk( t, U )
    t = t + dt
































