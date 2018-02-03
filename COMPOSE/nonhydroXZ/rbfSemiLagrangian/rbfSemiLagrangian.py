import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gab import nonhydro, phs2, rk

###########################################################################

testCase = "bubble"
formulation = "exner"
semiLagrangian = 0
rbfs = 0
dx = 100.
ds = 100.
FD = 2                                    #Order of lateral FD (2, 4, or 6)
rbforder = 5
polyorder = 3
stencilSize = 45
saveDel = 100
var = 2
plotFromSaved = 0

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

FDo2 = np.int( FD/2 )
ii = np.arange( 1, nLev+1 )
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

###########################################################################

#Define operators using functions from nonhydro.m:

if rbfs == 0 :
    
    if FD == 2 :
        wx = np.array( [ -1./2., 0., 1./2. ] )
        wxhv = np.array( [ 1., -2., 1. ] )
        gamma = 1./2.
    elif FD == 4 :
        wx = np.array( [ 1./12., -2./3., 0., 2./3., -1./12. ] )
        wxhv = np.array( [ 1., -4., 6., -4., 1. ] )
        gamma = -1./12.
    elif FD == 6 :
        wx = np.array( [ -1./60., 3./20., -3./4., 0., 3./4, -3./20., 1./60. ] )
        wxhv = np.array( [ 1., -6., 15., -20., 15., -6., 1. ] )
        gamma = 1./60.
    else :
        sys.exit( "\nError: FD should be 2, 4, or 6.\n" )
    
    ws = np.array( [ -1./2., 0., 1./2. ] )
    wshv = np.array( [ 1., -2., 1. ] )
    
    def Lx( U ) :
        return nonhydro.LxFD( U, wx, jj, dx, FD, FDo2 )
    def Lxhv( U ) :
        return nonhydro.LxFD( U, wxhv, jj, dx, FD, FDo2 )
    def Ls( U ) :
        return nonhydro.LsFD( U, ws, ii, ds )
    def Lshv( U ) :
        return nonhydro.LsFD( U, wshv, ii, ds )
    
elif rbfs == 1 :
    
    ind = nonhydro.getIndexes( x, z, xLeft, xRight, zSurf, zTop, FD, nLev, nCol )
    
    sys.exit( "\nError: RBFs aren't working yet.\n" )
    
else :
    
    sys.exit( "\nError: rbfs should be 0 or 1.\n" )

###########################################################################

bigTx = np.tile( Tx, (2,1) )
bigTz = np.tile( Tz, (2,1) )

normGradS = np.sqrt( dsdxBottom**2. + dsdzBottom**2. )

###########################################################################

def setGhostNodes( U ) :
    return nonhydro.setGhostNodesFD( U \
    , Tx, Tz, Nx, Nz, bigTx, bigTz \
    , nLev, nCol, thetaBar, g, Cp \
    , normGradS, ds, dsdxBottom, dsdzBottom \
    , wx, jj, dx, FD, FDo2 )

def odefun( t, U ) :
    return nonhydro.odefunFD( t, U, setGhostNodes \
    , dx, ds, wx, ws, wxhv, wshv \
    , ii, jj, i0, i1, j0, j1 \
    , dsdx, dsdz, FD, FDo2 \
    , Cp, Cv, Rd, g, gamma )

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

plt.contourf( x[ii,:], z[ii,:], nonhydro.LsFD( U[2,:,:]-thetaBar, ws, ii, ds ) )
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