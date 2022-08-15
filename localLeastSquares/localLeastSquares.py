
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sys

sys.path.append("../site-packages")
import halton
from gab import rectangleSurf

################################################################################

seeNodes = 0
noise    = 1

# Main approximation parameters
useRbfs     = 0
interpolate = 0
rbfParam    = 7
pd          = 5

# Number of subregions going across the domain in each direction
n = 8
m = n

################################################################################

# Boundaries of the rectangular domain
a = 0
b = 1
c = 0
d = 1

# Define the scattered locations where the function values are known
N = 500
sc = -1/3/np.sqrt(N)
pts = halton.halton(2, N)
# pts = np.random.rand(N, 2)
pts[:,0] = a - sc*(b-a) + (1+2*sc)*(b-a) * pts[:,0]
pts[:,1] = c - sc*(d-c) + (1+2*sc)*(d-c) * pts[:,1]

# Add some boundary points
sN = int(np.sqrt(N))
tmpx = np.linspace(a, b, sN)
tmpy = np.linspace(c, d, sN)
bottom = np.vstack((tmpx, c*np.ones(np.shape(tmpx))))[:,:-1]
right = np.vstack((b*np.ones(np.shape(tmpy)), tmpy))[:,:-1]
top = np.vstack((tmpx, d*np.ones(np.shape(tmpx))))[:,1:]
left = np.vstack((a*np.ones(np.shape(tmpy)), tmpy))[:,1:]
pts = np.vstack((pts, bottom.T, right.T, top.T, left.T))

#Separate into x and y arrays for convenience
x = pts[:,0]
y = pts[:,1]

# Define the regular mesh where you WANT to know the function values
X = np.linspace(a, b, 128)
Y = np.linspace(c, d, 128)
X, Y = np.meshgrid(X, Y)
X = X.flatten()
Y = Y.flatten()

################################################################################

# The true function to use for this test

alp = .5
aRandom = -alp + 2 * alp * np.random.rand(21**2)

def func(x, y):
    # return x**2*y + y**3
    # return x*y - y**2
    # return x - y
    # return 2 * np.ones(len(x))
    z = np.exp(-100. * ((x - .5)**2 + (y - .5)**2))
    z = z + np.exp(-100. * ((x - .8)**2 + (y - .8)**2))
    z = z + np.exp(-100. * ((x - .2)**2 + (y - .8)**2))
    z = z + np.exp(-100. * ((x - .2)**2 + (y - .2)**2))
    z = z + np.exp(-100. * ((x - .8)**2 + (y - .2)**2))
    if noise:
        count = 0
        for xi in np.linspace(a, b, 21):
            for yi in np.linspace(c, d, 21):
                z = z + aRandom[count] * np.exp(-1600. * ((x - xi)**2 + (y - yi)**2))
                count = count + 1
    return z

################################################################################

# Subroutine that plots a quadrilateral mesh.  Do this before other plotting.

def plotQuadMesh(xm, ym):
    lw = 1
    a = np.min(xm)
    b = np.max(xm)
    c = np.min(ym)
    d = np.max(ym)
    m = len(ym)
    n = len(xm)
    for i in range(n):
        plt.plot([xm[i], xm[i]], [c, d], 'k', linewidth=lw)
    for j in range(m):
        plt.plot([a, b], [ym[j], ym[j]], 'k', linewidth=lw)

################################################################################

# The vector of all known function values
f = func(x, y)

# Get the length and width of the two rectangles of interest, small and large.
w = (b - a) / n / 2
W = 3 * w
ell = (d - c) / m / 2
ELL = 3 * ell

# Get the x and y vectors that define the quadrilateral mesh
xm = np.linspace(a, b, n+1)
ym = np.linspace(c, d, m+1)

# Get the x and y coordinates of the centers of each square in the mesh
xmc = (xm[:-1] + xm[1:]) / 2
ymc = (ym[:-1] + ym[1:]) / 2
xmc, ymc = np.meshgrid(xmc, ymc)
xmc = xmc.flatten()
ymc = ymc.flatten()

################################################################################

# Plot the nodes

if seeNodes:
    for i in range(m*n):
        IND = inSquare(x, y, xmc[i], ymc[i], ELL, W)
        plt.figure(1)
        plt.clf()
        plotQuadMesh(xm, ym)
        plt.plot(x, y, 'k.')
        plt.plot(xmc, ymc, 'r.')
        s = 50
        plt.scatter(x[IND], y[IND], facecolors='none', edgecolors='green', s=s)
        plt.scatter(xmc[i], ymc[i], facecolors='none', edgecolors='red', s=s)
        plt.axis('image')
        if useRbfs:
            ind = inSquare(x[IND], y[IND], xmc[i], ymc[i], ell, w)
            if interpolate:
                nRbfs = len(IND)
            else:
                nRbfs = len(ind)
        else:
            nRbfs = 0
        plt.title("n = " + str(len(IND)) + ", nRbfs = " + str(nRbfs) \
        + ", nPoly = " + str(int((pd+1)*(pd+2)/2)))
        plt.draw()
        plt.waitforbuttonpress()

################################################################################

# Loop through the subdomains, solve for the RBF/poly coefficients and evaluate.

if useRbfs:
    if interpolate:
        approx = rectangleSurf.RBFinterp(rbfParam, pd, x, y, f, X, Y \
        , xmc = xmc, ymc = ymc, ell = ell, w = w)
    else:
        approx = rectangleSurf.RBFLS(rbfParam, pd, x, y, f, X, Y \
        , xmc = xmc, ymc = ymc, ell = ell, w = w)
else:
    approx = rectangleSurf.polyLS(pd, x, y, f, X, Y \
    , xmc = xmc, ymc = ymc, ell = ell, w = w)

################################################################################

# Plot the approximation and compare to the true function

clevels = np.arange(-.5, 1.7, .2)
ms = 3

triang = mtri.Triangulation(X, Y)
fig = plt.figure()

ax = fig.add_subplot(121)
cs = ax.tricontourf(triang, approx, clevels)
plotQuadMesh(xm, ym)
plt.plot(x, y, 'k.', markersize=ms)
plt.title('Approximation')
fig.colorbar(cs)
plt.axis('image')

ax = fig.add_subplot(122)
cs = ax.tricontourf(triang, func(X,Y), clevels)
plotQuadMesh(xm, ym)
plt.plot(x, y, 'k.', markersize=ms)
plt.title('Exact Function')
fig.colorbar(cs)
plt.axis('image')

plt.show()
