
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy import spatial
import sys
import time

sys.path.append("../site-packages")
import halton

################################################################################

seeNodes = 1

# Main approximation parameters
useRbfs = 0
interpolate = 0
rbfParam = 7
pd = 5

# Number of subregions going across the domain in each direction
n = 8
m = 8

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

alp = .5
aRandom = -alp + 2 * alp * np.random.rand(21**2)

# The true function to use for this test
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
    count = 0
    for xi in np.linspace(a, b, 21):
        for yi in np.linspace(c, d, 21):
            z = z + aRandom[count] * np.exp(-1600. * ((x - xi)**2 + (y - yi)**2))
            count = count + 1
    return z

################################################################################

# Subroutine that plots a quadrilateral mesh.  Do this before other plotting.

def plotQuadMesh(xm, ym):
    a = np.min(xm)
    b = np.max(xm)
    c = np.min(ym)
    d = np.max(ym)
    m = len(ym)
    n = len(xm)
    for i in range(n):
        plt.plot([xm[i], xm[i]], [c, d], 'b')
    for j in range(m):
        plt.plot([a, b], [ym[j], ym[j]], 'b')

################################################################################

# Determine the index of the nodes in the square centered at [xmci, ymci]

def inSquare(x, y, xmci, ymci, ell, w):
    ind = np.array([], int)
    for j in range(len(x)):
        if (np.abs(x[j] - xmci) <= w) and (np.abs(y[j] - ymci) <= ell):
            ind = np.hstack((ind, j))
    return ind

################################################################################

# Construct a matrix with polynomial columns.  pd is the polynomial degree.

def poly(x, y, pd):
    p = np.zeros((len(x), int((pd+1)*(pd+2)/2)), float)
    if pd >= 0:
        p[:,0] = 1.
    if pd >= 1:
        p[:,1] = x
        p[:,2] = y
    if pd >= 2:
        p[:,3] = x**2
        p[:,4] = x * y
        p[:,5] = y**2
    if pd >= 3:
        p[:,6] = x**3
        p[:,7] = x**2 * y
        p[:,8] = x * y**2
        p[:,9] = y**3
    if pd >= 4:
        p[:,10] = x**4
        p[:,11] = x**3 * y
        p[:,12] = x**2 * y**2
        p[:,13] = x * y**3
        p[:,14] = y**4
    if pd >= 5:
        p[:,15] = x**5
        p[:,16] = x**4 * y
        p[:,17] = x**3 * y**2
        p[:,18] = x**2 * y**3
        p[:,19] = x * y**4
        p[:,20] = y**5
    if pd >= 6:
        p[:,21] = x**6
        p[:,22] = x**5 * y
        p[:,23] = x**4 * y**2
        p[:,24] = x**3 * y**3
        p[:,25] = x**2 * y**4
        p[:,26] = x * y**5
        p[:,27] = y**6
    if (pd < 0) or (pd > 6):
        sys.exit("Please choose a better polynomial degree (0 <= pd <= 6).")
    return p

################################################################################

# Polyharmonic spline radial basis function

def phs(x, y, rbfParam):
    return (x**2 + y**2) ** (rbfParam/2)

################################################################################

# Construct a matrix with RBF columns

def rbf(x, y, xc, yc, rbfParam):
    A = np.zeros((len(x), len(xc)), float)
    for i in range(len(x)):
        for j in range(len(xc)):
            A[i,j] = phs(x[i] - xc[j], y[i] - yc[j], rbfParam)
    return A

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
            nRbfs = len(ind)
        else:
            nRbfs = 0
        plt.title("n = " + str(len(IND)) + ", nRbfs = " + str(nRbfs) \
        + ", nPoly = " + str(int((pd+1)*(pd+2)/2)))
        plt.draw()
        plt.waitforbuttonpress()

# Loop through the subdomains, solve for the RBF/poly coefficients and evaluate.
approx = np.zeros(len(X), float)
for i in range(m*n):
    IND = inSquare(x, y, xmc[i], ymc[i], ELL, W)
    p = poly(x[IND], y[IND], pd)
    if useRbfs:
        if interpolate:
            xc = x[IND]
            yc = y[IND]
            A = rbf(x[IND], y[IND], xc, yc, rbfParam)
            A = np.hstack((A, p))
            numP = int((pd + 1) * (pd + 2) / 2)
            tmp = np.hstack(( p.T, np.zeros((numP, numP))))
            A = np.vstack((A, tmp))
            tmp = np.zeros((len(IND), 1))
            tmp[:,0] = f[IND]
            lam = np.linalg.solve(A, np.vstack((tmp, np.zeros((numP, 1)))))
        else:
            ind = inSquare(x[IND], y[IND], xmc[i], ymc[i], ell, w)
            xc = x[IND][ind]
            yc = y[IND][ind]
            A = rbf(x[IND], y[IND], xc, yc, rbfParam)
            A = np.hstack((A, p))
            lam = np.linalg.lstsq(A, f[IND], rcond=None)[0]
        IND = inSquare(X, Y, xmc[i], ymc[i], ell, w)
        A = rbf(X[IND], Y[IND], xc, yc, rbfParam)
        B = poly(X[IND], Y[IND], pd)
        B = np.hstack((A, B))
    else:
        lam = np.linalg.lstsq(p, f[IND], rcond=None)[0]
        IND = inSquare(X, Y, xmc[i], ymc[i], ell, w)
        B = poly(X[IND], Y[IND], pd)
    approx[IND] = B.dot(lam).flatten()

################################################################################

# Plot the approximation and compare to the true function

clevels = np.arange(-.5, 1.7, .2)

triang = mtri.Triangulation(X, Y)
fig = plt.figure()

ax = fig.add_subplot(121)
cs = ax.tricontourf(triang, approx, clevels)
plt.title('Approximation')
fig.colorbar(cs)
plotQuadMesh(xm, ym)
plt.axis('image')

ax = fig.add_subplot(122)
if interpolate:
    cs = ax.tricontourf(triang, approx - func(X,Y))
    plt.title('Error')
else:
    cs = ax.tricontourf(triang, func(X,Y), clevels)
    plt.title('Exact Function')
fig.colorbar(cs)
plotQuadMesh(xm, ym)
plt.axis('image')

plt.show()
