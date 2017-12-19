import numpy as np
import scipy as sp
from scipy import spatial
from gab import phs2
import domain

testCase = "doubleStrakaTopo"
useMassFixer = 0
nx = 128+1
nz = 64+1
rbforder = 5
m = 10
n = 45
K = 2
gamma = -2^-3
tPlot = np.linspace( 0, 900, 90 )

rkStages = 3
dt = 1/24

seeDomain      = 0
seeStencils    = 0
computeWeights = 1
timeStep       = 1
seeContours    = 0
saveResults    = 1

xx, zz, xxc, zzc, dx, dz = domain.getNodes( testCase, a, b, c, d, nx, nz )