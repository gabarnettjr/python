import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../site-packages')
from gab import semiImplicit, nonhydro

###########################################################################

testCase = "doubleDensityCurrent"
dx = 400.
ds = 400.
FD = 4

###########################################################################

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= nonhydro.getSpaceDomain( testCase, dx, ds, 2 )
x = x[1:nLev+1,1:nCol+1]
z = z[1:nLev+1,1:nCol+1]

Lx, Lz = semiImplicit.getDerivativeOperators( nCol, nLev, FD, dx, ds )

plt.spy(Lx)
plt.show()