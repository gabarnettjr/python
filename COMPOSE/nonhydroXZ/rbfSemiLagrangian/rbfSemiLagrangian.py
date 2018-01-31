import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gab import nonhydro, phs2

###########################################################################

testCase = "bubble"
formulation = "exner"
semiLagrangian = 0
dx = 100.
ds = 100.
FD = 2
plotFromSaved = 0

###########################################################################

t = 0.

saveString = './results/' + testCase + '/dx' + \
'{0:1d}'.format(np.int(dx)) + 'ds' + '{0:1d}'.format(np.int(ds)) + '/'

###########################################################################

Cp, Cv, Rd, g, Po = nonhydro.getConstants()

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z = \
nonhydro.getSpaceDomain( testCase, dx, ds, FD )

tf, dt, nTimesteps = nonhydro.getTimeDomain( testCase, dx, ds )

s, dsdx, dsdz = nonhydro.getHeightCoordinate( zTop, zSurf, zSurfPrime )

Tx, Tz, Nx, Nz = nonhydro.getTanNorm( nCol, zSurfPrime, x )

U = nonhydro.getInitialConditions( testCase, formulation, \
nLev, nCol, FD, x, z, \
Cp, Cv, Rd, g, Po, \
dsdx, dsdz )

###########################################################################



###########################################################################

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
print("dt =",dt)
print()
print("nTimesteps =",nTimesteps)
print()
print(s)
print()
print(dsdx)
print()
print(dsdz)
print()
print("sizeU =",np.shape(U))
print()
print("Tx =",Tx)