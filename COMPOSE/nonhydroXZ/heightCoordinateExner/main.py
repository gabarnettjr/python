import sys
import numpy as np
import time
import matplotlib.pyplot as plt

#"exner", "hydrostaticPressure":
formulation = "exner"

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent", "movingDensityCurrent":
testCase = "movingDensityCurrent"

plotFromSaved = 1

highOrderZ = 0          #NOTE:  highOrderZ=1 is not working yet.

###########################################################################

#atmospheric constants:
Cp = 1004.
Cv = 717.
Rd = Cp - Cv
g = 9.81
Po = 10.**5.

#domain parameters:
t = 0.
rkStages = 3
if testCase == "bubble" :
    xLeft = 0.
    xRight = 10000.
    nLev = 100
    nCol = 100
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
    dt = 1./5.
elif testCase == "igw" :
    xLeft = 0.
    xRight = 300000.
    nLev = 40
    nCol = 20*30
    def zSurf(xTilde) :
        return np.zeros( np.shape(xTilde) )
    def zSurfPrime(xTilde) :
        return np.zeros( np.shape(xTilde) )
    zTop = 10000.
    tf = 3000.
    dt = 1./2.
elif testCase == "densityCurrent" :
    xLeft = -25600.
    xRight = 25600.
    nLev = 32
    nCol = 32*8
    def zSurf(xTilde) :
        return np.zeros( np.shape(xTilde) )
    def zSurfPrime(xTilde) :
        return np.zeros( np.shape(xTilde) )
    zTop = 6400.
    tf = 900.
    dt = 1./3.
elif testCase == "doubleDensityCurrent" :
    xLeft = -6400.
    xRight = 6400.
    nLev = 64
    nCol = 64*2
    def zSurf(xTilde) :
        return 1000. * np.exp( -(16.*(xTilde-1000.)/(xRight-xLeft))**2. )
        # return np.zeros( np.shape(xTilde) )
    def zSurfPrime(xTilde) :
        return -2. * 16.*(xTilde-1000.)/(xRight-xLeft) *  16./(xRight-xLeft) * zSurf(xTilde)
        # return np.zeros( np.shape(xTilde) )
    zTop = 6400.
    tf = 900.
    dt = 1./6.
elif testCase == "movingDensityCurrent" :
    xLeft = -18000.
    xRight = 18000.
    nLev = 32
    nCol = 180
    def zSurf(xTilde) :
        return np.zeros( np.shape(xTilde) )
    def zSurfPrime(xTilde) :
        return np.zeros( np.shape(xTilde) )
    zTop = 6400.
    tf = 900.
    dt = 1./3.
else :
    sys.exit( "\nError: Invalid test case string.\n" )
nTimesteps = round( (tf-t) / dt )

#path to saved results:
saveString = './results/' + formulation + '/' + testCase + '/nLev' + '{0:1d}'.format(nLev) + '_nCol' + '{0:1d}'.format(nCol) + '/'

#definition of the scale-preserving s-coordinate and its derivatives:
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
z[nLev+1,:] = zTop + dz/2.
x = np.tile( x, (nLev+2,1) )
ds = zTop * dz / ( zTop - zs )
ds = ds[0]

#tangent and normal vectors at surface:
Tx = np.ones(( 1, nCol ))
Tz = zSurfPrime( x[0,:] )
normT = np.sqrt( Tx**2 + Tz**2 )
Tx = Tx / normT
Tz = Tz / normT
bigTx = np.vstack((Tx,Tx))
bigTz = np.vstack((Tz,Tz))
Nx = -Tz
Nz = Tx

#definition of background states and initial conditions:
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
elif testCase == "igw" :
    N = .01
    theta0 = 300.
    thetaBar = theta0 * np.exp( (N**2/g) * z )
    piBar = 1. + g**2. / Cp / theta0 / N**2. * ( np.exp(-N**2./g*z) - 1. )
    thetaC = .01
    hC = 10000.
    aC = 5000.
    xC = 100000.
    thetaPrime0 = thetaC * np.sin( np.pi*z/hC ) / ( 1. + ((x-xC)/aC)**2 )
    piPrime0 = 0.
    U[0,:,:] = 20. * np.ones( np.shape(thetaPrime0) )
elif testCase == "densityCurrent" :
    thetaBar = 300. * np.ones(( nLev+2, nCol ))
    piBar = 1. - g / Cp / thetaBar * z
    xc = 0.
    zc = 3000.
    xr = 4000.
    zr = 2000.
    rTilde = np.sqrt( ((x-xc)/xr)**2 + ((z-zc)/zr)**2 )
    Tprime0 = np.zeros( np.shape(thetaBar) )
    ind = rTilde <= 1
    Tprime0[ind] = -15./2. * ( 1. + np.cos(np.pi*rTilde[ind]) )
    thetaPrime0 = Tprime0 / piBar
    piPrime0 = 0.
    U[0,:,:] = np.zeros( np.shape(thetaBar) )
elif testCase == "doubleDensityCurrent" :
    thetaBar = 300. * np.ones(( nLev+2, nCol ))
    piBar = 1. - g / Cp / thetaBar * z
    xc1 = -6400.
    xc2 = 6400.
    zc = 3000.
    xr = 4000.
    zr = 2000.
    rTilde1 = np.sqrt( ((x-xc1)/xr)**2 + ((z-zc)/zr)**2 )
    rTilde2 = np.sqrt( ((x-xc2)/xr)**2 + ((z-zc)/zr)**2 )
    Tprime0 = np.zeros( np.shape(thetaBar) )
    ind1 = rTilde1 <= 1
    ind2 = rTilde2 <= 1
    Tprime0[ind1] = -15./2. * ( 1. + np.cos(np.pi*rTilde1[ind1]) )
    Tprime0[ind2] = -15./2. * ( 1. + np.cos(np.pi*rTilde2[ind2]) )
    thetaPrime0 = Tprime0 / piBar
    piPrime0 = 0.
    U[0,:,:] = np.zeros( np.shape(thetaBar) )
elif testCase == "movingDensityCurrent" :
    thetaBar = 300. * np.ones(( nLev+2, nCol ))
    piBar = 1. - g / Cp / thetaBar * z
    xc = 0.
    zc = 3000.
    xr = 4000.
    zr = 2000.
    rTilde = np.sqrt( ((x-xc)/xr)**2 + ((z-zc)/zr)**2 )
    Tprime0 = np.zeros( np.shape(thetaBar) )
    ind = rTilde <= 1
    Tprime0[ind] = -15./2. * ( 1. + np.cos(np.pi*rTilde[ind]) )
    thetaPrime0 = Tprime0 / piBar
    piPrime0 = 0.
    U[0,:,:] = 20. * np.ones( np.shape(thetaBar) )
else :
    sys.exit("\nError: Invalid test case string.\n")
U[1,:,:] = np.zeros( np.shape(thetaBar) )
U[2,:,:] = thetaBar + thetaPrime0
if formulation == "exner" :
    U[3,:,:] = piBar + piPrime0
elif formulation == "hydrostaticPressure" :
    U[3,:,:] = -g / Rd / U[2,:,:] / dsdz(x,z) * Po * (piBar+piPrime0)**(Cv/Rd)
    dpidsBar = -g / Rd / thetaBar / dsdz(x,z) * Po * piBar**(Cv/Rd)
else :
    sys.exit( "\nError: Invalid formulation string.\n" )

#convert functions to values on nodes:
dsdxBottom = dsdx( x[0,:], zs )
dsdzBottom = dsdz( x[0,:], zs )
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
    if np.shape(np.shape(U))[0] == 3 :
        return ( ws[0]*U[:,0:nLev,:] + ws[1]*U[:,1:nLev+1,:] + ws[2]*U[:,2:nLev+2,:] ) / ds
    elif np.shape(np.shape(U))[0] == 2 :
        return ( ws[0]*U[0:nLev,:] + ws[1]*U[1:nLev+1,:] + ws[2]*U[2:nLev+2,:] ) / ds
    else :
        sys.exit( "\nError: U should be a 2D or 3D array.\n" )

wxhv = [ 1., -4., 6., -4., 1. ]
def HVx( U, u ) :
    V = np.zeros( np.shape(U) )
    if np.shape(np.shape(U))[0] == 3 :
        V[:,:,0]        = wxhv[0]*U[:,:,nCol-2]   + wxhv[1]*U[:,:,nCol-1]   + wxhv[2]*U[:,:,0]        + wxhv[3]*U[:,:,1]        + wxhv[4]*U[:,:,2]
        V[:,:,1]        = wxhv[0]*U[:,:,nCol-1]   + wxhv[1]*U[:,:,0]        + wxhv[2]*U[:,:,1]        + wxhv[3]*U[:,:,2]        + wxhv[4]*U[:,:,3]
        V[:,:,2:nCol-2] = wxhv[0]*U[:,:,0:nCol-4] + wxhv[1]*U[:,:,1:nCol-3] + wxhv[2]*U[:,:,2:nCol-2] + wxhv[3]*U[:,:,3:nCol-1] + wxhv[4]*U[:,:,4:nCol]
        V[:,:,nCol-2]   = wxhv[0]*U[:,:,nCol-4]   + wxhv[1]*U[:,:,nCol-3]   + wxhv[2]*U[:,:,nCol-2]   + wxhv[3]*U[:,:,nCol-1]   + wxhv[4]*U[:,:,0]
        V[:,:,nCol-1]   = wxhv[0]*U[:,:,nCol-3]   + wxhv[1]*U[:,:,nCol-2]   + wxhv[2]*U[:,:,nCol-1]   + wxhv[3]*U[:,:,0]        + wxhv[4]*U[:,:,1]
    elif np.shape(np.shape(U))[0] == 2 :
        V[:,0]        = wxhv[0]*U[:,nCol-2]   + wxhv[1]*U[:,nCol-1]   + wxhv[2]*U[:,0]        + wxhv[3]*U[:,1]        + wxhv[4]*U[:,2]
        V[:,1]        = wxhv[0]*U[:,nCol-1]   + wxhv[1]*U[:,0]        + wxhv[2]*U[:,1]        + wxhv[3]*U[:,2]        + wxhv[4]*U[:,3]
        V[:,2:nCol-2] = wxhv[0]*U[:,0:nCol-4] + wxhv[1]*U[:,1:nCol-3] + wxhv[2]*U[:,2:nCol-2] + wxhv[3]*U[:,3:nCol-1] + wxhv[4]*U[:,4:nCol]
        V[:,nCol-2]   = wxhv[0]*U[:,nCol-4]   + wxhv[1]*U[:,nCol-3]   + wxhv[2]*U[:,nCol-2]   + wxhv[3]*U[:,nCol-1]   + wxhv[4]*U[:,0]
        V[:,nCol-1]   = wxhv[0]*U[:,nCol-3]   + wxhv[1]*U[:,nCol-2]   + wxhv[2]*U[:,nCol-1]   + wxhv[3]*U[:,0]        + wxhv[4]*U[:,1]
    else :
        sys.exit( "\nError: U should be a 2D or 3D array.\n" )
    return -1./12. * np.abs(u) * V / dx

wshv = [ 1., -2., 1 ]
def HVs( U, sDot ) :
    if np.shape(np.shape(U))[0] == 3 :
        return 1./2. * np.abs(sDot) * ( wshv[0]*U[:,0:nLev,:] + wshv[1]*U[:,1:nLev+1,:] + wshv[2]*U[:,2:nLev+2,:] ) / ds
    elif np.shape(np.shape(U))[0] == 2 :
        return 1./2. * np.abs(sDot) * ( wshv[0]*U[0:nLev,:] + wshv[1]*U[1:nLev+1,:] + wshv[2]*U[2:nLev+2,:] ) / ds
    else :
        sys.exit( "\nError: U should be a 2D or 3D array.\n" )

###########################################################################

#Setting ghost node values and getting the RHS of the ODE system:

if formulation == "exner" :
    if highOrderZ == 1 :
        def setGhostNodes( U ) :
            #extrapolate uT to bottom ghost nodes:
            uT = U[0,1:12,:]*bigTx + U[1,1:12,:]*bigTz
            uT = np.sum( we*uT, axis=0 )
            #get uN on bottom ghost nodes:
            uN = U[0,1:11,:]*bigNx + U[1,1:11,:]*bigNz
            uN = np.sum( wi*uN, axis=0 )
            #use uT and uN to get (u,w) on bottom ghost nodes, then get (u,w) on top ghost nodes:
            U[0,0,:] = uT*Tx + uN*Nx
            U[1,0,:] = uT*Tz + uN*Nz
            U[0,nLev+1,:] = 2*U[0,nLev,:] - U[0,nLev-1,:]
            U[1,nLev+1,:] = -U[1,nLev,:]
            #extrapolate theta to bottom ghost nodes, then top ghost nodes:
            U[2,0,:] = thetaBar[0,:] + 2*(U[2,1,:]-thetaBar[1,:]) - (U[2,2,:]-thetaBar[2,:])
            U[2,nLev+1,:] = thetaBar[nLev+1,:] + 2*(U[2,nLev,:]-thetaBar[nLev,:]) - (U[2,nLev-1,:]-thetaBar[nLev-1,:])
            #get pi on bottom ghost nodes using derived BC:
            dpidx = Dx( U[3,1:3,:] )
            dpidx = 3./2.*dpidx[0,:] - 1./2.*dpidx[1,:]
            th = ( U[2,0,:] + U[2,1,:] ) / 2.
            U[3,0,:] = U[3,1,:] + ds/normGradS**2 * ( g/Cp/th*dsdzBottom + dpidx*dsdxBottom )
            #get pi on top ghost nodes:
            th = ( U[2,nLev,:] + U[2,nLev+1,:] ) / 2.
            U[3,nLev+1,:] = U[3,nLev,:] - ds/dsdzBottom*g/Cp/th
            return U
    elif highOrderZ == 0 :
        def setGhostNodes( U ) :
            #extrapolate uT to bottom ghost nodes:
            uT = U[0,1:3,:]*bigTx + U[1,1:3,:]*bigTz
            uT = 2*uT[0,:] - uT[1,:]
            #get uN on bottom ghost nodes:
            uN = U[0,1,:]*Nx + U[1,1,:]*Nz
            uN = -uN
            #use uT and uN to get (u,w) on bottom ghost nodes, then get (u,w) on top ghost nodes:
            U[0,0,:] = uT*Tx + uN*Nx
            U[1,0,:] = uT*Tz + uN*Nz
            U[0,nLev+1,:] = 2*U[0,nLev,:] - U[0,nLev-1,:]
            U[1,nLev+1,:] = -U[1,nLev,:]
            #extrapolate theta to bottom ghost nodes, then top ghost nodes:
            U[2,0,:] = thetaBar[0,:] + 2*(U[2,1,:]-thetaBar[1,:]) - (U[2,2,:]-thetaBar[2,:])
            U[2,nLev+1,:] = thetaBar[nLev+1,:] + 2*(U[2,nLev,:]-thetaBar[nLev,:]) - (U[2,nLev-1,:]-thetaBar[nLev-1,:])
            #get pi on bottom ghost nodes using derived BC:
            dpidx = Dx( U[3,1:3,:] )
            dpidx = 3./2.*dpidx[0,:] - 1./2.*dpidx[1,:]
            th = ( U[2,0,:] + U[2,1,:] ) / 2.
            U[3,0,:] = U[3,1,:] + ds/normGradS**2 * ( g/Cp/th*dsdzBottom + dpidx*dsdxBottom )
            #get pi on top ghost nodes:
            th = ( U[2,nLev,:] + U[2,nLev+1,:] ) / 2.
            U[3,nLev+1,:] = U[3,nLev,:] - ds/dsdzBottom*g/Cp/th
            return U
    else :
        sys.exit( "\nError: highOrderZ should be zero or one.\n" )
    def odefun( t, U ) :
        V = np.zeros( np.shape(U) )
        #set ghost node values for all variables:
        U = setGhostNodes( U )
        #get Us and vertical dissipation, then remove ghost nodes from U (no longer needed):
        Us = Ds(U)
        sDot = U[0,1:nLev+1,:]*dsdx + U[1,1:nLev+1,:]*dsdz
        sDot = np.tile( sDot, (4,1,1) )
        V[:,1:nLev+1,:] = HVs( U, sDot )
        U = U[:,1:nLev+1,:]
        #get RHS of ode function:
        Ux = Dx(U)
        u = np.tile( U[0,:,:], (4,1,1) )
        V[:,1:nLev+1,:] = V[:,1:nLev+1,:] - u * Ux - sDot * Us + HVx( U, u )
        V[0,1:nLev+1,:] = V[0,1:nLev+1,:] - Cp*U[2,:,:] * ( Ux[3,:,:] + Us[3,:,:]*dsdx )
        V[1,1:nLev+1,:] = V[1,1:nLev+1,:] - Cp*U[2,:,:] * ( Us[3,:,:]*dsdz ) - g
        V[3,1:nLev+1,:] = V[3,1:nLev+1,:] - Rd/Cv*U[3,:,:] * ( Ux[0,:,:]+Us[0,:,:]*dsdx + Us[1,:,:]*dsdz )
        return V
elif formulation == "hydrostaticPressure" :
    def setGhostNodes( U, P, sDot ) :
        #extrapolate uT to bottom ghost nodes:
        uT = U[0,1:3,:]*np.vstack((Tx,Tx)) + U[1,1:3,:]*np.vstack((Tz,Tz))
        uT = 2*uT[0,:] - uT[1,:]
        #get uN on bottom ghost nodes:
        uN = U[0,1,:]*Nx + U[1,1,:]*Nz
        uN = -uN
        #use uT and uN to get (u,w) on bottom ghost nodes, then get (u,w) on top ghost nodes:
        U[0,0,:] = uT*Tx + uN*Nx
        U[1,0,:] = uT*Tz + uN*Nz
        U[0,nLev+1,:] = 2*U[0,nLev,:] - U[0,nLev-1,:]
        U[1,nLev+1,:] = -U[1,nLev,:]
        #extrapolate theta to bottom ghost nodes, then to top ghost nodes:
        U[2,0,:] = thetaBar[0,:] + 2*(U[2,1,:]-thetaBar[1,:]) - (U[2,2,:]-thetaBar[2,:])
        U[2,nLev+1,:] = thetaBar[nLev+1,:] + 2*(U[2,nLev,:]-thetaBar[nLev,:]) - (U[2,nLev-1,:]-thetaBar[nLev-1,:])
        #get P on bottom ghost nodes using derived BC:
        dpids = (dpidsBar[0,:]+dpidsBar[1,:])/2. + 3./2.*(U[3,1,:]-dpidsBar[1,:]) - 1./2.*(U[3,2,:]-dpidsBar[2,:])
        dpdx = Dx(P[1:3,:])
        dpdx = 3./2.*dpdx[0,:] - 1./2.*dpdx[1,:]
        P[0,:] = P[1,:] - ds/normGradS**2. * ( dpids*dsdzBottom**2. - dpdx*dsdxBottom )
        #get P on top ghost nodes:
        dpids = (dpidsBar[nLev,:]+dpidsBar[nLev+1,:])/2. + 3./2.*(U[3,nLev,:]-dpidsBar[nLev,:]) - 1./2.*(U[3,nLev-1,:]-dpidsBar[nLev-1,:])
        P[nLev+1,:] = P[nLev,:] + ds*dpids
        #get sDot on bottom, then on top ghost nodes:
        sDot[0,:] = -sDot[1,:]
        sDot[nLev+1,:] = -sDot[nLev,:]
        #get dpids on bottom, then top ghost nodes:
        U[3,0,:] = -g / Rd / U[2,0,:] / dsdzBottom * ( Po**(Rd/Cv) * P[0,:] ) ** (Cv/Cp)
        U[3,nLev+1,:] = -g / Rd / U[2,nLev+1,:] / dsdzBottom * ( Po**(Rd/Cv) * P[nLev+1,:] ) ** (Cv/Cp)
        return U, P, sDot
    def odefun( t, U ) :
        #get diagnostic pressure on interior nodes:
        P = np.zeros(( nLev+2, nCol ))
        P[1:nLev+1,:] = Po**(-Rd/Cv) * ( -Rd * U[2,1:nLev+1,:] / g * U[3,1:nLev+1,:] * dsdz ) ** (Cp/Cv)
        #get sDot on interior nodes:
        sDot = np.zeros(( nLev+2, nCol ))
        sDot[1:nLev+1,:] = U[0,1:nLev+1,:]*dsdx + U[1,1:nLev+1,:]*dsdz
        #set ghost node values using boundary conditions:
        U, P, sDot = setGhostNodes( U, P, sDot )
        #get velocity on interior nodes:
        u = U[0,1:nLev+1,:]
        sdotDpids = sDot * U[3,:,:]
        sDot = sDot[1:nLev+1,:]
        #get RHS of ode function:
        V = np.zeros(( 4, nLev+2, nCol ))
        V[0,1:nLev+1,:] = -u * Dx(U[0,1:nLev+1,:]) - sDot * Ds(U[0,:,:]) \
            + g / U[3,1:nLev+1,:] / dsdz * ( Dx(P[1:nLev+1,:]) + Ds(P)*dsdx ) \
            + HVx( U[0,1:nLev+1,:], u ) + HVs( U[0,:,:], sDot )
        V[1,1:nLev+1,:] = -u * Dx(U[1,1:nLev+1,:]) - sDot * Ds(U[1,:,:]) \
            - g * ( 1. - Ds(P) / U[3,1:nLev+1,:] ) \
            + HVx( U[1,1:nLev+1,:], u ) + HVs( U[1,:,:], sDot )
        V[2,1:nLev+1,:] = -u * Dx(U[2,1:nLev+1,:]) - sDot * Ds(U[2,:,:]) \
            + HVx( U[2,1:nLev+1,:], u ) + HVs( U[2,:,:], sDot )
        V[3,1:nLev+1,:] = -Dx(U[3,1:nLev+1,:]*u) - Ds(sdotDpids) \
            + HVx( U[3,1:nLev+1,:], u ) + HVs( U[3,:,:], sDot )
        return V
else :
    sys.exit( "\nError: Invalid formulation string.\n" )

###########################################################################

def rk( t, U ) :
    if rkStages == 4 :
        q1 = odefun( t,      U         )
        q2 = odefun( t+dt/2, U+dt/2*q1 )
        q3 = odefun( t+dt/2, U+dt/2*q2 )
        q4 = odefun( t+dt,   U+dt*q3   )
        return U + dt/6 * ( q1 + 2*q2 + 2*q3 + q4 )
    elif rkStages == 3 :
        q1 = odefun( t,        U           )
        q2 = odefun( t+dt/3,   U+dt/3*q1   )
        q2 = odefun( t+2*dt/3, U+2*dt/3*q2 )
        return U + dt/4 * ( q1 + 3*q2 )
    else :
        sys.exit( "\nError: rkStages should be 3 or 4.\n" )

###########################################################################

#set how often to save the results, and set contour levels:

if testCase == "bubble" :
    saveDel = 100.
    # saveDel = 1500.
    CL = np.arange( -.05, 2.15, .1 )
elif testCase == "igw" :
    saveDel = 200.
    # saveDel = 600.
    CL = np.arange( -.0021, .0035, .0002 )
    # CL = np.arange( -.0015, .0035, .0005 )
elif ( testCase == "densityCurrent" ) | ( testCase == "doubleDensityCurrent" ) | ( testCase == "movingDensityCurrent" ) :
    saveDel = 50.
    # saveDel = 900.
    CL = np.arange( -16.5, 1.5, 1. )
else :
    sys.exit( "\nError: Invalid test case string.\n" )

###########################################################################

#stepping forward in time with explicit RK:

et = 0.             #elapsed time
plt.ion()           #interactive plotting on
print()
if plotFromSaved == 1 :
    if testCase == "densityCurrent" :
        fig = plt.figure( figsize = (30,10) )
    elif testCase == "doubleDensityCurrent" :
        fig = plt.figure( figsize = (30,15) )
    elif testCase == "movingDensityCurrent" :
        fig = plt.figure( figsize = (30,10) )
    elif testCase == "bubble" :
        fig = plt.figure( figsize = (12,10) )
    elif testCase == "igw" :
        fig = plt.figure( figsize = (30,3) )
    else :
        sys.exit( "\nError: Invalid test case string.\n" )
    ax = fig.add_subplot(111)

for i in range(nTimesteps+1) :
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        if plotFromSaved == 0 :
            if formulation == "hydrostaticPressure" :
                P = np.zeros(( nLev+2, nCol ))
                P[1:nLev+1,:] = Po**(-Rd/Cv) * ( -Rd * U[2,1:nLev+1,:] / g * U[3,1:nLev+1,:] * dsdz ) ** (Cp/Cv)
                sDot = np.zeros(( nLev+2, nCol ))
                sDot[1:nLev+1,:] = U[0,1:nLev+1,:]*dsdx + U[1,1:nLev+1,:]*dsdz
                U, P, sDot = setGhostNodes( U, P, sDot )
            elif formulation == "exner" :
                U = setGhostNodes(U)
            else :
                sys.exit( "\nError: Invalid formulation string.\n" )
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        else :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
            cp = plt.contourf( x, z, np.squeeze(U[2,:,:])-thetaBar, CL )
            if testCase == "densityCurrent" :
                cb = plt.colorbar( cp, orientation = 'horizontal', fraction = .25, pad = -.05, aspect = 100 )
                cb.ax.tick_params( labelsize = 30 )
            elif testCase == "doubleDensityCurrent" :
                cb = plt.colorbar( cp )
                cb.ax.tick_params( labelsize = 20 )
            elif testCase == "movingDensityCurrent" :
                cb = plt.colorbar( cp, orientation = 'horizontal', fraction = .25, pad = -.05, aspect = 100 )
                cb.ax.tick_params( labelsize = 30 )
            elif testCase == "bubble" :
                cb = plt.colorbar( cp )
                cb.ax.tick_params( labelsize = 20 )
            elif testCase == "igw" :
                cb = plt.colorbar( cp, orientation = 'horizontal', fraction = .25, pad = .05, aspect = 100 )
                cb.ax.tick_params( labelsize = 20 )
            else :
                sys.exit( "\nError: Invalid test case string.\n" )
            plt.title( '{0}, t = {1:04.0f}, ' . format( testCase, t ) )
            plt.axis( 'equal' )
            # plt.axis( [ xLeft-dx, xRight+dx, -dx, zTop+dx ] )
            plt.axis( 'off' )
            # fig.savefig( 'foo' + '{0:1d}'.format(np.int(np.round(t))) + '.png', bbox_inches = 'tight' )
            plt.waitforbuttonpress()
            plt.clf()
        print( "t =", np.int(np.round(t)) )
        print( "et =", time.clock()-et )
        et = time.clock()
        print( [ np.min(U[0,:,:]), np.max(U[0,:,:]) ] )
        print( [ np.min(U[1,:,:]), np.max(U[1,:,:]) ] )
        print( [ np.min(U[2,:,:]-thetaBar), np.max(U[2,:,:]-thetaBar) ] )
        if formulation == "exner" :
            pi = U[3,:,:]
        elif formulation == "hydrostaticPressure" :
            pi = ( -Rd * U[2,:,:] * np.tile(dsdzBottom,(nLev+2,1)) / g / Po * U[3,:,:] ) ** (Rd/Cv)
        else :
            sys.exit( "\nError: Invalid formulation string.\n" )
        print( [ np.min(pi-piBar), np.max(pi-piBar) ] )
        print()
    if plotFromSaved == 0 :
        U = rk( t, U )
    t = t + dt

###########################################################################



























