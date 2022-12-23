"""
This script interpolates the data (x, y, errX) to (xe, ye, F).
Similarly, it interpolates the data (x, y, errY) to (xe, ye, G).

The user must supply arrays x, y, errX, errY, xe, ye, errXe, errYe as input:
x.txt, y.txt, errX.txt, errY.txt, xe.txt, ye.txt, errXe.txt, errYe.txt

The output is the interpolated values, contained in files F.txt and G.txt

Greg Barnett
December 2022
"""
################################################################################

from sys import path, argv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri as mtri

path.append("C:\\Users\\gabarne\\OneDrive\\python\\surf2d")
import rectangleSurf

################################################################################

# PARAMETERS

showPlots = True

# The exponent in the polyharmonic spline (PHS) radial basis function (RBF)
rbfExp = 3

# The highest degree polynomial to include in the basis
deg = 1

# Set this to True to use (only) polynomial least squares, instead of RBFs.
polyLS = False

# Parameters for performing local approximations, rather than one global
local = False                                   # if False, then no subdivisions
mSubd = 8                                      # number of vertical subdivisions
nSubd = mSubd                                # number of horizontal subdivisions

# OPTIONAL COMMAND-LINE PARAMETER (working directory where text files are saved)
# x.txt, y.txt, errX.txt, errY.txt, xe.txt, ye.txt, errXe.txt, errYe.txt
# NOTE: Type the directory using "windows style", like this:
#       C:\Users\gabarne\cdInterpData\smoothData
if len(argv) == 2 :
    wdir = argv[1]
    tmp = wdir.split("\\")
    wdir = tmp.pop() + "\\"
    while tmp :
        wdir = tmp.pop() + "\\" + wdir
else :
    wdir = "C:\\Users\\gabarne\\cdInterpData\\smoothData\\"

################################################################################

# Nodes (x,y).  Coordinates where you KNOW the function values.

x = np.array([])
with open(wdir + "x.txt") as f :
    for line in f :
        x = np.hstack((x, np.float64(line.strip())))

i = 0
y = np.zeros(len(x))
with open(wdir + "y.txt") as f :
    for line in f :
        y[i] = np.float64(line.strip())
        i += 1

################################################################################

# Values of the two functions (errX and errY) at the nodes.

i = 0
errX = np.zeros(len(x))
with open(wdir + "errX.txt") as f :
    for line in f :
        errX[i] = np.float64(line.strip())
        i += 1

i = 0
errY = np.zeros(len(x))
with open(wdir + "errY.txt") as f :
    for line in f :
        errY[i] = np.float64(line.strip())
        i += 1

################################################################################

# # Try using built-in linear Delaunay interpolation.  This doesn't actually
# # work, because some of the evaluation points fall outside the triangulation
# # induced by the nodes.

# triang = mtri.Triangulation(x, y)
# H = mtri.triinterpolate.LinearTriInterpolator(triang, errX)
# I = mtri.triinterpolate.LinearTriInterpolator(triang, errY)

################################################################################

# Evaluation points (xe,ye).  Where you WANT to know the function.

xe = np.array([])
with open(wdir + "xe.txt") as f :
    for line in f :
        xe = np.hstack((xe, np.float64(line.strip())))

i = 0
ye = np.zeros(len(xe))
with open(wdir + "ye.txt") as f :
    for line in f :
        ye[i] = np.float64(line.strip())
        i += 1

################################################################################

# Known values at evaluation points, to check performance of the interpolation.

i = 0
errXe = np.zeros(len(xe))
with open(wdir + "errXe.txt") as f :
    for line in f :
        errXe[i] = np.float64(line.strip())
        i += 1

i = 0
errYe = np.zeros(len(xe))
with open(wdir + "errYe.txt") as f :
    for line in f :
        errYe[i] = np.float64(line.strip())
        i += 1

################################################################################

# Shift/scale the coordinates for a well-conditioned problem.

# Shift so that (0,0) is the center of the computational domain.
xavg = np.mean(x)
x = x - xavg
xe = xe - xavg
yavg = np.mean(y)
y = y - yavg
ye = ye - yavg

# Constant re-scaling of the x and y coordinates (needed for high order poly).
alp = (np.max(np.abs(x)) + np.max(np.abs(y))) / 2
x = x / alp
xe = xe / alp
y = y / alp
ye = ye / alp

# The rectangular computational domain, [a,b] x [c,d].
a = np.min(np.hstack((x, xe)))
b = np.max(np.hstack((x, xe)))
c = np.min(np.hstack((y, ye)))
d = np.max(np.hstack((y, ye)))

# Find the center (xmc,ymc) and dimensions of each rectangular subdomain.
if local :
    eps = 0.001
    xmc = np.linspace(a - eps, b + eps, nSubd + 1)
    ymc = np.linspace(c - eps, d + eps, mSubd + 1)
    xmc = (xmc[:-1] + xmc[1:]) / 2
    ymc = (ymc[:-1] + ymc[1:]) / 2
    xmc, ymc = np.meshgrid(xmc, ymc)
    xmc = xmc.flatten()
    ymc = ymc.flatten()
    wSubd = (b - a + 2*eps) / nSubd / 2
    ellSubd = (d - c + 2*eps) / mSubd / 2
else :
    xmc, ymc, ellSubd, wSubd, tmp, tmp = rectangleSurf.assignDefaults(x, y \
    , np.array([]), np.array([]), [], [], [], [])

################################################################################

# Use rectangleSurf to get approximations, F and G, for errX and errY.

if polyLS :

    (F, LAM) = rectangleSurf.polyLS(deg, x, y, errX, xe, ye \
    , xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)
    
    # # TEST showing how "coeff" works.
    # print(np.shape(LAM))
    # (tmp, LAM) = rectangleSurf.polyLS(deg, x, y, errX, xe[0:5], ye[0:5] \
    # , coeff = LAM, xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)
    # print(np.shape(LAM))
    # # print(tmp)
    # tmp = rectangleSurf.polyLS(deg, x, y, errX, xe[0:10], ye[0:10] \
    # , coeff = LAM, xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)[0]
    # print(np.shape(LAM))
    # # print(tmp)

    G = rectangleSurf.polyLS(deg, x, y, errY, xe, ye \
    , xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)[0]
    
else :

    (F, LAM) = rectangleSurf.RBFinterp(rbfExp, deg, x, y, errX, xe, ye \
    , xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)

    # # TEST showing how "coeff" works.
    # print(LAM == [])
    # (tmp, LAM) = rectangleSurf.RBFinterp(rbfExp, deg, x, y, errX, xe[0:5], ye[0:5] \
    # , coeff = LAM, xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)
    # print(LAM == [])
    # # print(tmp)
    # tmp = rectangleSurf.RBFinterp(rbfExp, deg, x, y, errX, xe[0:10], ye[0:10] \
    # , coeff = LAM, xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)[0]
    # print(LAM == [])
    # # print(tmp)

    G = rectangleSurf.RBFinterp(rbfExp, deg, x, y, errY, xe, ye \
    , xmc = xmc, ymc = ymc, ell = ellSubd, w = wSubd)[0]

################################################################################

# Print estimated values at evaluation points (xe,ye) to output files.

with open(wdir + "F.txt", "w") as f :
    for i in range(len(F)) :
        f.write("{0:17.14f}\n".format(F[i]))

with open(wdir + "G.txt", "w") as f :
    for i in range(len(G)) :
        f.write("{0:17.14f}\n".format(G[i]))

################################################################################

if showPlots :

    # Re-scale the x and y coordinates to their original values, for viewing.
    
    x = alp * x + xavg
    xe = alp * xe + xavg
    y = alp * y + yavg
    ye = alp * ye + yavg
    
    a = alp * a + xavg
    b = alp * b + xavg
    c = alp * c + yavg
    d = alp * d + yavg
    
    ############################################################################

    # Stuff needed for the plots below.

    triang = mtri.Triangulation(x, y)                                    # nodes
    TRIANG = mtri.Triangulation(xe, ye)                      # evaluation points
    ms = 1                                                         # marker size
    nc = 32                                                   # number of colors
    box = [a, b, c, d]                                         # plotting window
    lw = 1                                                           # linewidth
    
    ############################################################################
    
    def plotThings(figNum, errXY, FG, errXYe, titleString) :
        """
        Creates one 2x2 array of subplots, each displaying some useful
        information about how well FG approximates errXY.  For comparison,
        the known values at the evaluation points must be given in errXYe.
        """
        fig = plt.figure(figNum, figsize = (13, 9.5))
        # plt.set_cmap('jet')

        plt.subplots_adjust(top=0.963, bottom=0.041, left=0.012, right=0.988 \
        , hspace=0.145, wspace=0.0)
        
        clevels = rectangleSurf.getContourLevels(np.hstack((errXY, FG, errXYe))  \
        , useMeanOf = np.array([0]), minDiff = 0, nColors = nc)

        # The known values on the nodes.
        ax = fig.add_subplot(221)
        cs = ax.tricontourf(triang, errXY, levels = clevels)
        ax.plot(x, y, 'ko', markersize = ms)
        ax.axis('image')
        ax.axis(box)
        fig.colorbar(cs)
        plt.title(titleString + " on Scattered Nodes")

        # The interpolant evaluated on the grid.
        ax = fig.add_subplot(222)
        cs = ax.tricontourf(TRIANG, FG, levels = clevels)
        ax.plot(x, y, 'ko', markersize = ms)
        ax.plot(xe, ye, 'r.', markersize = ms/2)
        ax.axis('image')
        ax.axis(box)
        fig.colorbar(cs)
        plt.title("Interpolant on Grid")
        
        # The known values on the grid.
        ax = fig.add_subplot(223)
        cs = ax.tricontourf(TRIANG, errXYe, levels = clevels)
        ax.plot(x, y, 'ko', markersize = ms)
        ax.plot(xe, ye, 'r.', markersize = ms/2)
        ax.axis('image')
        ax.axis(box)
        fig.colorbar(cs)
        plt.title("Known Values on Grid")
        
        # The error relative to the known values on the grid.
        tmp = (FG - errXYe) / np.max(np.abs(errXYe))
        clevels = rectangleSurf.getContourLevels(tmp \
        , useMeanOf = np.array([0]), minDiff = 0, nColors = nc)
        
        ax = fig.add_subplot(224)
        cs = ax.tricontourf(TRIANG, tmp, levels = clevels)
        ax.plot(x, y, 'ko', markersize = ms)
        ax.plot(xe, ye, 'r.', markersize = ms/2)
        ax.axis('image')
        ax.axis(box)
        fig.colorbar(cs)
        plt.title("Relative Error on Grid")
    
    ############################################################################

    # Plot errX stuff in Figure 1, errY stuff in Figure 2.
    
    plotThings(2, errY, G, errYe, "errY")
    plotThings(1, errX, F, errXe, "errX")

    ############################################################################

    plt.show()

