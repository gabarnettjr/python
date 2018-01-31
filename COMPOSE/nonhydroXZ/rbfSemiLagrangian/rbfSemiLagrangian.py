import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from gab import nonhydro, phs2

testCase = "bubble"
dx = 100.
ds = 100.
plotFromSaved = 0

Cp, Cv, Rd, g, Po = nonhydro.getConstants()

t = 0.

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime = nonhydro.getSpaceDomain( testCase, dx, ds )

tf, dt, nTimesteps = nonhydro.getTimeDomain( testCase, dx, ds )

saveString = './results/' + testCase + '/dx' + '{0:1d}'.format(np.int(dx)) + 'ds' + '{0:1d}'.format(np.int(ds)) + '/'

s, dsdx, dsdz = nonhydro.getHeightCoordinate( zTop, zSurf, zSurfPrime )



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