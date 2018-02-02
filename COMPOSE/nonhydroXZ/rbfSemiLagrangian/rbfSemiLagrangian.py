import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gab import nonhydro, phs2, rk

###########################################################################

testCase = "doubleDensityCurrent"
formulation = "exner"
semiLagrangian = 0
dx = 100.
ds = 100.
FD = 4                          #Order of lateral FD (positive even number)
rbforder = 3
polyorder = 1
stencilSize = 9
saveDel = 50
var = 3
plotFromSaved = 1

###########################################################################

t = 0.

saveString = './results/' + testCase + '/dx' \
+ '{0:1d}'.format(np.int(dx)) + 'ds' + '{0:1d}'.format(np.int(ds)) + '/'

###########################################################################

Cp, Cv, Rd, g, Po = nonhydro.getConstants()

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= nonhydro.getSpaceDomain( testCase, dx, ds, FD )

tf, dt, nTimesteps = nonhydro.getTimeDomain( testCase, dx, ds )

s, dsdx, dsdz = nonhydro.getHeightCoordinate( zTop, zSurf, zSurfPrime )

ii = np.arange( 1, nLev+1 )
FDo2 = np.int( FD/2 )
jj = np.arange( FDo2, nCol+FDo2 )

i0 = ii[0]
i1 = ii[-1] + 1
j0 = jj[0]
j1 = jj[-1] + 1

Tx, Tz, Nx, Nz = nonhydro.getTanNorm( zSurfPrime, x[0,jj] )

U, thetaBar, piBar = nonhydro.getInitialConditions( testCase, formulation \
, nLev, nCol, FD, x, z \
, Cp, Cv, Rd, g, Po \
, dsdx, dsdz )

###########################################################################

dsdxBottom = dsdx( x[0,jj], zSurf(x[0,jj]) )
dsdzBottom = dsdz( x[0,jj], zSurf(x[0,jj]) )
dsdx = dsdx( x[ii,:][:,jj], z[ii,:][:,jj] )
dsdz = dsdz( x[ii,:][:,jj], z[ii,:][:,jj] )

ind = nonhydro.getIndexes( x, z, xLeft, xRight, zSurf, zTop, FD, nLev, nCol )

###########################################################################

if FD == 2 :
    wx = np.array( [ -1./2., 0., 1./2. ] )
    wxhv = np.array( [ 1., -2., 1. ] )
elif FD == 4 :
    wx = np.array( [ 1./12., -2./3., 0., 2./3., -1./12. ] )
    wxhv = np.array( [ 1., -4., 6., -4., 1. ] )
else :
    sys.exit( "\nError: FD should be 2 or 4.\n" )

ws = np.array( [ -1./2., 0., 1./2. ] )
wshv = np.array( [ 1., -2., 1. ] )

###########################################################################

bigTx = np.tile( Tx, (2,1) )
bigTz = np.tile( Tz, (2,1) )

normGradS = np.sqrt( dsdxBottom**2. + dsdzBottom**2. )

def setGhostNodes( U ) :
    
    #extrapolate uT to bottom ghost nodes:
    uT = U[0,1:3,:][:,jj] * bigTx + U[1,1:3,:][:,jj] * bigTz
    uT = 2*uT[0,:] - uT[1,:]
    
    #get uN on bottom ghost nodes:
    uN = U[0,1,jj]*Nx + U[1,1,jj]*Nz
    uN = -uN
    
    #use uT and uN to get (u,w) on bottom ghost nodes, then get (u,w) on top ghost nodes:
    U[0,0,jj] = uT*Tx + uN*Nx
    U[1,0,jj] = uT*Tz + uN*Nz
    U[0,nLev+1,jj] = 2*U[0,nLev,jj] - U[0,nLev-1,jj]
    U[1,nLev+1,jj] = -U[1,nLev,jj]
    
    #extrapolate theta to bottom ghost nodes, then top ghost nodes:
    U[2,0,jj] = thetaBar[0,jj] + 2*(U[2,1,jj]-thetaBar[1,jj]) - (U[2,2,jj]-thetaBar[2,jj])
    U[2,nLev+1,jj] = thetaBar[nLev+1,jj] + 2*(U[2,nLev,jj]-thetaBar[nLev,jj]) \
    - (U[2,nLev-1,jj]-thetaBar[nLev-1,jj])
    
    #get pi on bottom ghost nodes using derived BC:
    dpidx = nonhydro.Lx( U[3,1:3,:], wx, jj, dx, FD, FDo2 )
    dpidx = 3./2.*dpidx[0,:] - 1./2.*dpidx[1,:]
    th = ( U[2,0,jj] + U[2,1,jj] ) / 2.
    U[3,0,jj] = U[3,1,jj] + ds/normGradS**2 * ( g/Cp/th*dsdzBottom + dpidx*dsdxBottom )
    
    #get pi on top ghost nodes:
    th = ( U[2,nLev,jj] + U[2,nLev+1,jj] ) / 2.
    U[3,nLev+1,jj] = U[3,nLev,jj] - ds/dsdzBottom*g/Cp/th
    
    #enforce periodic lateral boundary condition:
    U[:,:,0:FDo2] = U[:,:,nCol:nCol+FDo2]
    U[:,:,nCol+FDo2:nCol+FD] = U[:,:,FDo2:FD]
    
    return U

###########################################################################

def odefun( t, U ) :
    
    #Initialize output array:
    V = np.zeros( np.shape(U) )
    
    #Set ghost node values for all variables:
    U = setGhostNodes( U )
    
    #Us and Ux:
    Us = nonhydro.Ls( U[:,:,jj], ws, ii, ds )
    Ux = nonhydro.Lx( U[:,ii,:], wx, jj, dx, FD, FDo2 )
    
    #Get u and sDot:
    u = np.tile( U[0,ii,:][:,jj], (4,1,1) )
    sDot = U[0,ii,:][:,jj] * dsdx + U[1,ii,:][:,jj] * dsdz
    sDot = np.tile( sDot, (4,1,1) )
    
    #RHS of ODE function:
    V[ :, i0:i1, j0:j1 ] = - u * Ux - sDot * Us
    V[ 0, i0:i1, j0:j1 ] = V[0,ii,:][:,jj] - Cp*U[2,ii,:][:,jj] * ( Ux[3,:,:] + Us[3,:,:]*dsdx )
    V[ 1, i0:i1, j0:j1 ] = V[1,ii,:][:,jj] - Cp*U[2,ii,:][:,jj] * ( Us[3,:,:]*dsdz ) - g
    V[ 3, i0:i1, j0:j1 ] = V[3,ii,:][:,jj] - Rd/Cv*U[3,ii,:][:,jj] * ( Ux[0,:,:]+Us[0,:,:]*dsdx + Us[1,:,:]*dsdz )
    
    #Add dissipation:
    if FD == 2 :
        V[ :, i0:i1, j0:j1 ] = V[:,ii,:][:,:,jj] \
        + (1./2.) * np.abs(u) * nonhydro.Lx( U[:,ii,:], wxhv, jj, dx, FD, FDo2 )
    elif FD == 4 :
        V[ :, i0:i1, j0:j1 ] = V[:,ii,:][:,:,jj] \
        - (1./12.) * np.abs(u) * nonhydro.Lx( U[:,ii,:], wxhv, jj, dx, FD, FDo2 )
    else :
        sys.exit( "\nError: FD should be 2 or 4.\n" )
    V[ :, i0:i1, j0:j1 ] = V[:,ii,:][:,:,jj] \
    + (1./2.) * np.abs(sDot) * nonhydro.Ls( U[:,:,jj], wshv, ii, ds )
    
    return V

###########################################################################

#Time-stepping:

if testCase == "densityCurrent" :
    fig = plt.figure( figsize = (30,10) )
    CL = np.arange( -16.5, 1.5, 1. )
elif testCase == "doubleDensityCurrent" :
    fig = plt.figure( figsize = (30,12) )
    CL = np.arange( -16.5, 1.5, 1. )
elif testCase == "movingDensityCurrent" :
    fig = plt.figure( figsize = (30,10) )
    CL = np.arange( -16.5, 1.5, 1. )
elif testCase == "bubble" :
    fig = plt.figure( figsize = (18,14) )
    CL = np.arange( -.05, 2.15, .1 )
elif testCase == "igw" :
    fig = plt.figure( figsize = (40,4) )
    CL = np.arange( -.0021, .0035, .0002 )
else :
    sys.exit( "\nError: Invalid test case string.\n" )

for i in range(nTimesteps+1) :
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        if plotFromSaved == 0 :
            U = setGhostNodes( U )
            np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        elif plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        if var == 0 :
            plt.contourf( x, z, U[0,:,:], 20 )
        elif var == 1 :
            plt.contourf( x, z, U[1,:,:], 20 )
        elif var == 2 :
            plt.contourf( x, z, U[2,:,:]-thetaBar, CL )
        elif var == 3 :
            plt.contourf( x, z, U[3,:,:]-piBar, 20 )
        else :
            sys.exit( "\nError: var should be 0, 1, 2, or 3.\n" )
        if testCase != "igw" :
            plt.axis( 'equal' )
        plt.colorbar()
        plt.title( 'testCase = {0}, t = {1:04d}'.format(testCase,np.int(t)) )
        fig.savefig( 'foo' + '{0:1d}'.format(np.int(np.round(t))) + '.png', bbox_inches = 'tight' )
        plt.clf()
    if plotFromSaved == 0 :
        U = rk.rk3( t, U, odefun, dt )
    t = t + dt

###########################################################################

sys.exit("\nStop here for now.\n")

###########################################################################

#Go from 3D array to 2D array:
U = np.transpose( np.reshape( U, ( 4, (nLev+2)*(nCol+FD) ) ) )
#Go from 2D array back to 3D array:
U = np.reshape( np.transpose(U), ( 4, nLev+2, nCol+FD ) )

###########################################################################

#Print some things:

print()
print("dx =",dx)
print()
print("ds =",ds)
print()
print("dt =",dt)
print()
print("xLeft =",xLeft)
print()
print("xRight =",xRight)
print()
print("nLev =",nLev)
print()
print("nCol =",nCol)
print()
print("zTop =",zTop)
print()
print("zSurf =",zSurf)
print()
print("zSurfPrime =",zSurfPrime)
print()
print("tf =",tf)
print()
print("nTimesteps =",nTimesteps)
print()
print(s)
print()
print("sizeU =",np.shape(U))
print()

###########################################################################

#Contour plot of something:

plt.contourf( x[ii,:], z[ii,:], nonhydro.Ls( U[2,:,:]-thetaBar, ws, ii, ds ) )
if testCase != "igw" :
    plt.axis( 'equal' )
plt.colorbar()
fig.savefig( 'foo' + '.png', bbox_inches = 'tight' )
plt.clf()

###########################################################################

#Plot some nodes:

x = x.flatten()
z = z.flatten()

ms = 12

plt.plot( x, z, ".", color="black" )
plt.plot( x[ind.m],  z[ind.m],  "o", color="black",  fillstyle="none", markersize=10 )
plt.plot( x[ind.gl], z[ind.gl], "o", color="red",    fillstyle="none", markersize=ms )
plt.plot( x[ind.r],  z[ind.r],  "s", color="red",    fillstyle="none", markersize=ms )
plt.plot( x[ind.gr], z[ind.gr], "o", color="blue",   fillstyle="none", markersize=ms )
plt.plot( x[ind.l],  z[ind.l],  "s", color="blue",   fillstyle="none", markersize=ms )
plt.plot( x[ind.gb], z[ind.gb], "^", color="orange", fillstyle="none", markersize=ms )
plt.plot( x[ind.b],  z[ind.b],  "v", color="orange", fillstyle="none", markersize=ms )
plt.plot( x[ind.gt], z[ind.gt], "v", color="green",  fillstyle="none", markersize=ms )
plt.plot( x[ind.t],  z[ind.t],  "^", color="green",  fillstyle="none", markersize=ms )
if testCase != "igw" :
    plt.axis( 'equal' )
plt.show()

############################################################################