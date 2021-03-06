import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../../../site-packages')
from gab import rk
from gab.nonhydro import common

###########################################################################

#Switches that rarely need to be flipped:

heightCoord = False

saveArrays          = True
saveContours        = True
plotNodesAndExit    = False
plotBackgroundState = False

###########################################################################

#Explain the required and optional command-line arguments:

def printHelp():
    sys.exit("\n\
----------------------------------------------------------------------\n\n\
2D (XZ) DRY ATMOSPHERE PROTOTYPE USING HOMME-LIKE FORMULATION\n\n\
----------------------------------------------------------------------\n\n\
INFORMATION\n\n\
Governing equations are expressed in terms of velocities u and w, \n\
potential temperature theta, pseudo-density dpi/ds, geopotential phi, \n\
and diagnostic pressure perturbation P' (P = pi + P'), where pi \n\
is the hydrostatic pressure.\n\n\
You can switch between hydrostatic and nonhydrostatic governing \n\
equations, and in the hydrostatic case P' is zero, so w is zero for \n\
all time, and phi is diagnostic  instead of prognostic.\n\n\
To account for topography, a hybrid pressure coordinate s is used in \n\
the vertical direction, so that constant s-surfaces follow the \n\
topography at the bottom, but at the top they become nearly flat \n\
hydrostatic pressure surfaces.\n\n\
Variables are located vertically on either mid-levels or interfaces \n\
using Lorenz staggering.  w and phi are on interfaces, while the other \n\
prognostic variables and the diagnostic pressure are on mid-levels.\n\n\
----------------------------------------------------------------------\n\n\
REQUIRED ARGUMENTS\n\n\
argument 1 (atmospheric formulation)\n\
    hydrostatic\n\
    nonhydrostatic\n\n\
argument 2 (name of test case)\n\
    steadyState\n\
    risingBubble\n\
    densityCurrent\n\
    inertiaGravityWaves\n\
    tortureTest\n\
    scharMountainWaves\n\n\
argument 3 (vertical frame of reference, Eulerian or Lagrangian)\n\
    vEul\n\
    vLag\n\n\
argument 4 (refinement level)\n\
    0\n\
    1\n\
    2\n\
    3\n\
    4\n\n\
----------------------------------------------------------------------\n\n\
OPTIONAL ARGUMENTS\n\n\
argument 5 (what to plot)\n\
    u\n\
    w\n\
    theta\n\
    dpids\n\
    phi\n\
    P\n\
    pi\n\
    T\n\n\
argument 6 (contour levels)\n\
    number of contours (integer)\n\
    range of contours (using np.arange or np.linspace for example)\n\n\
final argument\n\
    If the final command line argument is fromSaved, then results will\n\
    be loaded and plotted from a previous run, otherwise the \n\
    simulation will start from scratch.\n\n\
----------------------------------------------------------------------\n\n\
EXAMPLES (run first example first)\n\n\
python main.py nonhydrostatic risingBubble vEul 1 P np.arange(-65,85,10)\n\n\
python main.py nonhydrostatic risingBubble vEul 1 w fromSaved\n\n\
python main.py nonhydrostatic risingBubble vEul 1 fromSaved\n\n\
----------------------------------------------------------------------\
")

###########################################################################

#Quick way to switch between [new simulation] and [plot from saved data]:

sysArgv = sys.argv.copy()

if sysArgv[-1] == "fromSaved":
    contourFromSaved = True
    sysArgv = sysArgv[:-1]
else:
    contourFromSaved = False

###########################################################################

#Parse required command-line inputs:

try:
    if sys.argv[1] == "hydrostatic":
        hydrostatic = True
    elif sys.argv[1] == "nonhydrostatic":
        hydrostatic = False
    else:
        raise ValueError("The first argument should be either " \
        + "hydrostatic or nonhydrostatic.")
    testCase = sysArgv[2]
    if sysArgv[3] == "vLag":
        verticallyLagrangian = True
    elif sysArgv[3] == "vEul":
        verticallyLagrangian = False
    else:
        raise ValueError("The third argument should be either vLag " \
        + "or vEul.")
    refinementLevel = np.int64(sysArgv[4])
except:
    printHelp()

###########################################################################

#Parse optional command-line inputs:

try:
    whatToPlot = sysArgv[5]
except:
    if testCase == "steadyState":
        whatToPlot = "u"
        contours = np.arange(-205, 215, 10) * 1e-2
    elif testCase == "risingBubble":
        whatToPlot = "theta"
        contours = np.arange(-.15, 2.25, .1)
    elif testCase == "densityCurrent":
        whatToPlot = "theta"
        contours = np.arange(-17.5, 2.5, 1)
    elif testCase == "inertiaGravityWaves":
        whatToPlot = "theta"
        contours = np.arange(-15, 37, 2) * 1e-4
    elif testCase == "tortureTest":
        whatToPlot = "u"
        contours = np.arange(-105, 115, 10) * 1e-2
    elif testCase == "scharMountainWaves":
        whatToPlot = "u"
        contours = np.arange(625, 1575, 50) * 1e-2
    else:
        raise ValueError("Invalid testCase string.")
else:
    try:
        contours = eval(sysArgv[6])
    except:
        contours = 20

###########################################################################

#Get string for saving results:

if hydrostatic:
    saveString = "./results/" + "hydrostatic"    + "_" + testCase + "_"
else:
    saveString = "./results/" + "nonhydrostatic" + "_" + testCase + "_"

if verticallyLagrangian:
    saveString = saveString + "vLag" + "_"
else:
    saveString = saveString + "vEul" + "_"

saveString = saveString + str(refinementLevel) + "/"

###########################################################################

#Remove old files, and make new directories if necessary:

if saveArrays and not contourFromSaved:
    if os.path.exists(saveString + '*.npy'):
        os.remove(saveString + '*.npy')                   #remove old files
    if not os.path.exists(saveString):
        os.makedirs(saveString)                       #make new directories

if saveContours:
    tmp = os.listdir(os.getcwd())
    for item in tmp:
        if item.endswith(".png"):
            os.remove(os.path.join(os.getcwd(), item))

if contourFromSaved:
    saveArrays = False
    saveContours = True

###########################################################################

#Definitions of atmospheric constants for dry air:
Cp, Cv, Rd, g, Po, th0, N = common.constants(testCase)

#Test-specific parameters describing domain and initial perturbation:
left, right, bottom, top, dx, nLev, dt, tf, saveDel, zSurf, thetaPtb \
= common.domainParameters(testCase, hydrostatic, refinementLevel \
, g, Cp, th0)

#Some other important parameters:
t = 0.                                                        #initial time
nTimesteps = np.int(np.round(tf / dt))                #number of time-steps
nCol = np.int(np.round((right - left) / dx))             #number of columns
x = np.linspace(left+dx/2, right-dx/2, nCol)        #array of x-coordinates

#Initial hydrostatic background state functions of z (given in test case):
potentialTemperature, potentialTemperatureDerivative \
, exnerPressure, inverseExnerPressure \
= common.hydrostaticProfiles(testCase, th0, Po, g, Cp, Rd, N)

###########################################################################

#Print some info about the test case that the user might want to know:

if verticallyLagrangian:
    vertically = "Lagrangian"
else:
    vertically = "Eulerian"
if hydrostatic:
    equations = "hydrostatic"
else:
    equations = "nonhydrostatic"
print("\n\
Equations  : {0:s}\n\
Test Case  : {1:s}\n\
Vertically : {2:s}\n\
Domain Box : [{3:g},{4:g},{5:g},{6:g}]\n\
Delta x    : {7:g}\n\
Levels     : {8:g}\n\
Delta t    : {9:g}\n\
Final time : {10:d}\n" \
. format(equations, testCase, vertically, left, right, bottom, top \
, dx, nLev, dt, np.int(tf)))

###########################################################################

#Functions for vertical derivatives (3-pt finite-differences):

def Ds(U):
    V = np.zeros(np.shape(U))
    V[0,:] = (-3./2.*U[0,:] + 2.*U[1,:] - 1./2.*U[2,:]) / ds
    V[1:-1,:] = (U[2:,:] - U[0:-2,:]) / (2.*ds)
    V[-1,:] = (3./2.*U[-1,:] - 2.*U[-2,:] + 1./2.*U[-3,:]) / ds
    return V

if (testCase == "inertiaGravityWaves") \
or (testCase == "tortureTest") \
or (testCase == "steadyState"):
    def HVs(U):
        return np.zeros((np.shape(U)[0]-2, np.shape(U)[1]))
else:
    def HVs(U):
        return 2.**-14./ds * (U[0:-2,:] - 2.*U[1:-1,:] + U[2:,:])

###########################################################################

#Weights and functions for lateral derivatives (7-pt finite-differences):

wd1 = np.array([-1./60., .15, -.75,   0., .75, -.15, 1./60.]) / dx
wd6 = np.array([     1., -6.,  15., -20., 15.,  -6.,     1.]) / dx**6.
whv = 2.**-1.*dx**5. * wd6               #multiply by dissipation parameter

def Da(U):
    d = len(np.shape(U))
    if d == 1:
        U = np.hstack((U[[-3,-2,-1]], U, U[[0,1,2]]))
        U = wd1[0]*U[0:-6] + wd1[1]*U[1:-5] + wd1[2]*U[2:-4] \
          + wd1[4]*U[4:-2] + wd1[5]*U[5:-1] + wd1[6]*U[6:]
    elif d == 2:
        U = np.hstack((U[:,[-3,-2,-1]], U, U[:,[0,1,2]]))
        U = wd1[0]*U[:,0:-6] + wd1[1]*U[:,1:-5] + wd1[2]*U[:,2:-4] \
          + wd1[4]*U[:,4:-2] + wd1[5]*U[:,5:-1] + wd1[6]*U[:,6:]
    else:
        raise ValueError("U should have either one or two dimensions.")
    return U

if (testCase == "inertiaGravityWaves") \
or (testCase == "tortureTest"):
    def HVa(U):
        return np.zeros(np.shape(U))
else:
    def HVa(U):
        d = len(np.shape(U))
        if d == 1:
            U = np.hstack((U[[-3,-2,-1]], U, U[[0,1,2]]))
            U = whv[0]*U[0:-6] + whv[1]*U[1:-5] + whv[2]*U[2:-4] \
              + whv[3]*U[3:-3] \
              + whv[4]*U[4:-2] + whv[5]*U[5:-1] + whv[6]*U[6:]
        elif d == 2:
            U = np.hstack((U[:,[-3,-2,-1]], U, U[:,[0,1,2]]))
            U = whv[0]*U[:,0:-6] + whv[1]*U[:,1:-5] + whv[2]*U[:,2:-4] \
              + whv[3]*U[:,3:-3] \
              + whv[4]*U[:,4:-2] + whv[5]*U[:,5:-1] + whv[6]*U[:,6:]
        else:
            raise ValueError("U should have either one or two dimensions.")
        return U

###########################################################################

#Equally spaced array of vertical coordinate values (s) on mid-levels:

piTop = Po * exnerPressure(top)  ** (Cp/Rd)             #HS pressure at top
piSurf = Po * exnerPressure(zSurf(x)) ** (Cp/Rd)    #HS pressure at surface
sTop = piTop / Po                             #value of s on upper boundary
ds = (1. - sTop) / nLev                                            #delta s
s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)    #equally-spaced on mid-levels

###########################################################################

#Prescribed functions A(s) and B(s) for setting hybrid coordinate levels:

def A(s):
    return (1. - s) / (1. - sTop) * sTop

def B(s):
    return (s - sTop) / (1. - sTop)

def Aprime(s):
    return -sTop / (1. - sTop)

def Bprime(s):
    return 1. / (1. - sTop)

ssMid = np.tile(s, (nCol, 1)).T                       #s-mesh on mid-levels
ssInt = (ssMid[0:-1,:] + ssMid[1:,:]) / 2.            #s-mesh on interfaces

###########################################################################

#Iterate to get initial vertical height levels on interfaces (zzInt):

#dpi/ds defined using vertical coordinate on interior mid-levels:
dpids = Aprime(ssMid[1:-1,:]) * Po \
+ Bprime(ssMid[1:-1,:]) * np.tile(piSurf, (nLev,1))

#integrate dpi/ds to get HS pressure, as will be done during time-stepping:
tmp = piTop * np.ones((nCol))
pi = np.zeros((nLev+1, nCol))
pi[0,:] = tmp.copy()
for i in range(nLev):
    tmp = tmp + dpids[i,:] * ds
    pi[i+1,:] = tmp.copy()
pi0 = pi.copy()

zzInt = inverseExnerPressure((pi/Po) ** (Rd/Cp))             #initial guess
zzInt0 = zzInt.copy()
xxInt = np.tile(x, (nLev+1,1))
xxMid = np.tile(x, (nLev, 1))

pi = (pi[:-1,:] + pi[1:,:]) / 2.             #avg pi to interior mid-levels

for j in range(10):
    zzMid = (zzInt[:-1,:] + zzInt[1:,:]) / 2.
    theta = potentialTemperature(zzMid) + thetaPtb(xxMid, zzMid)
    integrand = dpids * Rd * theta / Po**(Rd/Cp) / pi**(Cv/Cp) / g
    tmp = zzInt[-1,:].copy()
    for i in range(nLev):
        tmp = tmp + integrand[-(i+1),:] * ds
        zzInt[-(i+2),:] = tmp.copy()
    # plt.figure()
    # plt.contourf(xxInt, zzInt0, zzInt-zzInt0, 20)
    # plt.colorbar()
    # plt.show()

###########################################################################

if plotNodesAndExit:
    plt.figure()
    plt.plot(xxMid.flatten(), zzMid.flatten(), marker=".", color="black" \
    , linestyle="none")
    for i in range(nLev+1):
        plt.plot(x, zzInt[i,:], color="red", linestyle="-")
    plt.plot([left,left], [zSurf(left),top], color="red" \
    , linestyle="-")
    plt.plot([right,right], [zSurf(right),top], color="red" \
    , linestyle="-")
    plt.axis("image")
    plt.show()
    sys.exit("Finished plotting.")

###########################################################################

#Assignment of initial conditions:

theta0 = potentialTemperature(zzMid) + thetaPtb(xxMid, zzMid)
dpids0 = Aprime(ssMid) * Po + Bprime(ssMid) * np.tile(piSurf, (nLev+2,1))
mass0 = np.sum(dpids0[1:-1,:] / g * ds * dx)                  #initial mass

phi0 = g * zzInt
phi0mid = g * zzMid
# phi0mid = (phi0[0:-1,:] + phi0[1:,:]) / 2.
phi0mid = np.vstack((2.*phi0[0,:] - phi0mid[0,:] \
, phi0mid \
, 2.*phi0[-1,:] - phi0mid[-1,:]))

U = np.zeros((6, nLev+2, nCol))                  #Main 3D array of unknowns

if testCase == "inertiaGravityWaves":
    U[0,:,:] = 20. * np.ones((nLev+2, nCol))
elif testCase == "scharMountainWaves":
    U[0,:,:] = 10. * np.ones((nLev+2, nCol))     
else:
    U[0,:,:] = np.zeros((nLev+2, nCol)) #horizontal velocity u (mid-levels)

U[1,0:-1,:] = np.zeros((nLev+1, nCol))    #vertical velocity w (interfaces)

U[2,1:-1,:] = theta0.copy()       #potential temperature theta (mid-levels)

U[3,:,:] = dpids0.copy()                 #pseudo-density dpids (mid-levels)

U[4,0:-1,:] = phi0.copy()                    #geopotential phi (interfaces)

U[5,:,:] = np.zeros((nLev+2, nCol))   #pressure perturbation P (mid-levels)

###########################################################################

#Initialize a figure of the appropriate size:

if saveContours:
    if testCase == "inertiaGravityWaves":
        fig = plt.figure(figsize = (18,3))
    else:
        fig = plt.figure(figsize = (18,14))

###########################################################################

#Create and save a contour plot of the field specified by whatToPlot:

def contourSomething(U, t, pi, thetaBar, dpidsBar, phiBar \
, whatToPlot, figName, contours):

    if plotBackgroundState:
        if whatToPlot == "theta":
            tmp = thetaBar[1:-1,:].copy()
        elif whatToPlot == "dpids":
            tmp = dpidsBar[1:-1,:].copy()
        elif whatToPlot == "phi":
            tmp = phiBar.copy()
        elif whatToPlot == "P":
            tmp = (pi[:-1,:] + pi[1:,:]) / 2.
        elif whatToPlot == "T":
            pi = (pi[:-1,:] + pi[1:,:]) / 2.
            tmp = (pi / Po) ** (Rd/Cp) * thetaBar[1:-1,:]
        elif whatToPlot == "rho":
            pi = (pi[:-1,:] + pi[1:,:]) / 2.
            tmp = pi / Rd / (pi/Po)**(Rd/Cp) / thetaBar[1:-1,:]
        else:
            raise ValueError("Invalid whatToPlot string.  Choose theta" \
            + ", dpids, phi, or P.")
    else:
        if whatToPlot == "u":
            tmp = U[0,1:-1,:]
        elif whatToPlot == "w":
            tmp = U[1,0:-1,:]
        elif whatToPlot == "theta":
            tmp = U[2,1:-1,:] - thetaBar[1:-1,:]
        elif whatToPlot == "dpids":
            tmp = U[3,1:-1,:] - dpidsBar[1:-1,:]
        elif whatToPlot == "phi":
            tmp = U[4,0:-1,:] - phiBar
        elif whatToPlot == "P":
            tmp = U[5,1:-1,:]
        elif whatToPlot == "pi":
            tmp = pi - pi0
        elif whatToPlot == "T":
            pi = (pi[:-1,:] + pi[1:,:]) / 2.
            Tbar = (pi/Po)**(Rd/Cp) * thetaBar[1:-1,:]
            tmp = ((pi+U[5,1:-1,:])/Po)**(Rd/Cp) * U[2,1:-1,:] - Tbar
        elif whatToPlot == "rho":
            pi = (pi[:-1,:] + pi[1:,:]) / 2.
            rhoBar = pi / Rd / (pi/Po)**(Rd/Cp) / thetaBar[1:-1,:]
            tmp = -U[3,1:-1,:]/((U[4,1:-1,:]-U[4,0:-2,:])/ds) - rhoBar
        else:
            raise ValueError("Invalid whatToPlot string.  Choose u, " \
            + "w, theta, dpids, phi, P, or pi.")

    if (whatToPlot == "phi") | (whatToPlot == "w") | (whatToPlot == "pi"):
        xxTmp = xxInt
        zzTmp = U[4,0:-1,:] / g                          #changing z-levels
    else:
        xxTmp = xxMid
        zzTmp = (U[4,0:-2,:]+U[4,1:-1,:])/2. / g
    
    plt.clf()
    plt.contourf(xxTmp, zzTmp, tmp, contours)
    plt.plot(x, U[4,0,:]/g, linestyle="-", color="red")
    plt.plot(x, U[4,-2,:]/g, linestyle="-", color="red")
    # plt.plot(xxTmp.flatten(), zzTmp.flatten(), color="red", marker="." \
    # , linestyle="none", markersize=1)

    if (testCase == "inertiaGravityWaves") \
    or (testCase == "tortureTest") \
    or (testCase == "scharMountainWaves"):
        plt.colorbar(orientation="horizontal")
    elif (testCase == "densityCurrent"):
        plt.axis("image")
        plt.colorbar(orientation="horizontal")
    else:
        plt.axis("image")
        plt.colorbar(orientation="vertical")

    if (testCase == "scharMountainWaves") \
    and ((whatToPlot == "u") or (whatToPlot == "w")):
        plt.axis([-20000., 30000., np.min(zzInt[-1,:])-250., 20000.])
    else:
        plt.axis([left-250., right+250. \
        , np.min(zzInt[-1,:])-250., np.max(zzInt[0,:])+250.])

    if plotBackgroundState:
        plt.title("min={0:g}, max={1:g}".format(np.min(tmp),np.max(tmp)))
        plt.show()
        sys.exit("\nDone plotting the requested background state.")
    else:
        fig.savefig(figName, bbox_inches="tight")   #save figure as a png

###########################################################################

#Initialize the output array for the vertical re-map function:

if verticallyLagrangian:
    Vmidlevels  = np.zeros((2, nLev+2, nCol))
    Vinterfaces = np.zeros((2, nLev+1, nCol))

###########################################################################

#This will be used inside the setGhostNodes() function to evaluate
#background states on the moving vertical levels:

def fastBackgroundStates(phi, pi):
    
    #Get thetaBar on mid-levels from the given background state function:
    tmp = (phi[:-1,:] + phi[1:,:]) / 2.
    tmp = np.vstack((2.*phi[0,:] - tmp[0,:] \
    , tmp \
    , 2.*phi[-1,:] - tmp[-1,:]))
    thetaBar = potentialTemperature(tmp/g)

    if heightCoord:
        #Background state for phi is initial value of phi:
        phiBar = phi0
        #Solve for dpidsBar on mid-levels:
        dpidsBar = -((pi[:-1,:]+pi[1:,:])/2)**(Cv/Cp) \
        * ((phiBar[1:,:]-phiBar[:-1,:])/ds) * Po**(Rd/Cp) / Rd \
        / thetaBar[1:-1,:]
        # dpidsBar = np.vstack((dpidsBar[0,:] \
        # , dpidsBar \
        # , dpidsBar[-1,:]))
        dpidsBar = np.vstack((2.*dpidsBar[0,:] - dpidsBar[1,:] \
        , dpidsBar \
        , 2.*dpidsBar[-1,:] - dpidsBar[-2,:]))
    else:
        #Get background state for pseudo-density dpi/ds on mid-levels:
        dpidsBar = Aprime(ssMid) * Po \
        + Bprime(ssMid) * np.tile(pi[-1,:], (nLev+2, 1))
        #Integrate to get background state for geopotential on interfaces:
        tmp = phi[-1,:].copy()
        phiBar = np.zeros((nLev+1, nCol))
        phiBar[-1,:] = tmp.copy()
        integrand = dpidsBar[1:-1,:] * Rd * thetaBar[1:-1,:] / Po**(Rd/Cp) \
        / ((pi[:-1,:]+pi[1:,:])/2.)**(Cv/Cp)
        for i in range(nLev):
            tmp = tmp + integrand[-(i+1),:] * ds
            phiBar[-(i+2),:] = tmp.copy()

    return thetaBar, dpidsBar, phiBar

###########################################################################

def setGhostNodes(U):

    if heightCoord:
        dpids = 3./2.*U[3,1,:] - 1./2.*U[3,2,:]
        theta = 3./2.*U[2,1,:] - 1./2.*U[2,2,:]
        dphids = (-3./2.*U[4,0,:] + 2.*U[4,1,:] - 1./2.*U[4,2,:]) / ds
        pTop = (-dpids * Rd * theta / dphids / Po**(Rd/Cp)) ** (Cp/Cv)
    else:
        pTop = piTop * np.ones((nCol))

    #Integrate dpi/ds to get hydrostatic pressure pi on interfaces:
    tmp = pTop.copy()
    pi = np.zeros((nLev+1, nCol))
    pi[0,:] = tmp.copy()
    for i in range(nLev):
        tmp = tmp + U[3,i+1,:] * ds
        pi[i+1,:] = tmp.copy()

    if hydrostatic:
        #Integrate EOS to get diagnostic geopotential on interfaces:
        tmp = U[4,-2,:].copy()
        phi = np.zeros((nLev+1, nCol))
        phi[-1,:] = tmp.copy()
        integrand = U[3,1:-1,:] * Rd * U[2,1:-1,:] / Po**(Rd/Cp) \
        / ((pi[:-1,:]+pi[1:,:])/2.)**(Cv/Cp)
        for i in range(nLev):
            tmp = tmp + integrand[-(i+1),:] * ds
            phi[-(i+2),:] = tmp.copy()
        U[4,:-1,:] = phi.copy()

    #Get background states:
    thetaBar, dpidsBar, phiBar = fastBackgroundStates(U[4,:-1,:], pi)

    #Extrapolate dpids to bottom ghost nodes:
    # U[3,-1,:] = U[3,-2,:]
    U[3,-1,:] = 2. * (U[3,-2,:]-dpidsBar[-2,:]) - (U[3,-3,:]-dpidsBar[-3,:]) \
    + dpidsBar[-1,:]

    #Extrapolate dpids to top ghost nodes:
    # U[3,0,:] = U[3,1,:]
    U[3,0,:] = 2. * (U[3,1,:]-dpidsBar[1,:]) - (U[3,2,:]-dpidsBar[2,:]) \
    + dpidsBar[0,:]

    #Extrapolate theta to bottom ghost nodes:
    U[2,-1,:] = 2. * (U[2,-2,:]-thetaBar[-2,:]) - (U[2,-3,:]-thetaBar[-3,:]) \
    + thetaBar[-1,:]

    #Extrapolate theta to top ghost nodes:
    U[2,0,:] = 2. * (U[2,1,:]-thetaBar[1,:]) - (U[2,2,:]-thetaBar[2,:]) \
    + thetaBar[0,:]
    
    #extrapolate u to bottom ghost nodes:
    U[0,-1,:] = 2.*U[0,-2,:] - U[0,-3,:]

    #extrapolate u to top ghost nodes:
    U[0,0,:] = 2.*U[0,1,:] - U[0,2,:]

    if not hydrostatic:
        #Use EOS to get pressure perturbation on mid-levels:
        U[5,1:-1,:] = (-U[3,1:-1,:] / ((U[4,1:-1,:]-U[4,0:-2,:])/ds) * Rd \
        * U[2,1:-1,:] / Po**(Rd/Cp)) ** (Cp/Cv) \
        - (pi[:-1,:]+pi[1:,:])/2.
        if heightCoord:
            #Set pressure perturbation on top ghost nodes (Neumann BC):
            dphida = Da(U[4,0,:])
            dPda = Da(pi[0,:] + 3./2.*U[5,1,:] - 1./2.*U[5,2,:])
            dpids = 3./2.*U[3,1,:] - 1./2.*U[3,2,:]
            dphids = (-3./2.*U[4,0,:] + 2.*U[4,1,:] - 1./2.*U[4,2,:]) / ds
            RHS = (dPda * dphids - dphida * dpids) * dphida
            RHS = RHS / (g**2. + dphida**2.)
            U[5,0,:] = U[5,1,:] - ds * RHS
        else:
            #Set pressure perturbation on top ghost nodes (Dirichlet BC):
            U[5,0,:] = -U[5,1,:]
        #Set pressure perturbation on bottom ghost nodes using Neumann BC:
        dphida = Da(U[4,-2,:])
        dPda = Da(pi[-1,:] + 3./2.*U[5,-2,:] - 1./2.*U[5,-3,:])
        dpids = 3./2.*U[3,-2,:] - 1./2.*U[3,-3,:]
        dphids = (3./2.*U[4,-2,:] - 2.*U[4,-3,:] + 1./2.*U[4,-4,:]) / ds
        RHS = (dPda * dphids - dphida * dpids) * dphida
        RHS = RHS / (g**2. + dphida**2.)
        U[5,-1,:] = U[5,-2,:] + ds * RHS

    #get w on bottom boundary nodes:
    U[1,-2,:] = (U[0,-1,:]+U[0,-2,:])/2. / g * Da(U[4,-2,:])

    if heightCoord:
        #get w on top boundary nodes:
        U[1,0,:] = (U[0,0,:]+U[0,1,:])/2. / g * Da(U[4,0,:])

    return U, pi, thetaBar, dpidsBar, phiBar

###########################################################################

#Initial arrays of zeros for storing Runge-Kutta sub-steps.  If we are
#using RK3, then we need only two arrays, but for RK4 we need 4.  Note
#that dUdt and q1 are two different names for the same array.

rks = 3                           #hard-coded: number of Runge-Kutta stages

dUdt = np.zeros((6, nLev+2, nCol))
q1   = dUdt
q2   = np.zeros((6, nLev+2, nCol))

if rks == 4:
    q3 = np.zeros((6, nLev+2, nCol))
    q4 = np.zeros((6, nLev+2, nCol))

###########################################################################

#This describes the RHS of the system of ODEs in time that will be solved:

def odefun(t, U, dUdt):
    
    #Get some ghost node values and background states:
    U, pi, thetaBar, dpidsBar, phiBar = setGhostNodes(U)

    uInt = (U[0,:-1,:] + U[0,1:,:]) / 2.                   #u on interfaces
    dpidsInt = (U[3,:-1,:] + U[3,1:,:]) / 2.          #dpi/ds on interfaces

    #Get sDot:

    if verticallyLagrangian:
        sDot = np.zeros((nLev+1, nCol))
    else:
        if heightCoord:
            sDot = (-uInt*Da(U[4,:-1,:]) + g*U[1,:-1,:]) / Ds(U[4,:-1,:])
        else:
            #please see this overleaf document for a complete derivation,
            #starting from the governing equation for pseudo-density dpids:
            #https://www.overleaf.com/read/gcfkprynxvkw
            integrand = Da(U[3,1:-1,:] * U[0,1:-1,:])
            sDot = B(ssInt) * np.tile(np.sum(integrand*ds,0), (nLev+1,1))
            tmp = np.zeros((nCol))
            for j in range(nLev):
                tmp = tmp + integrand[j,:] * ds
                sDot[j+1,:] = sDot[j+1,:] - tmp
            sDot = sDot / dpidsInt
            # maxTop = np.max(np.abs(sDot[0,:]))
            # maxBot = np.max(np.abs(sDot[-1,:]))
            # if max(maxTop,maxBot) > 1e-15:
            #     raise ValueError("nonzero sDot on boundary.")
    
    #Main part:

    tmp = (U[0,1:,:] - U[0,0:-1,:]) / ds               #du/ds on interfaces
    tmp = sDot * tmp                              #sDot*du/ds on interfaces
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.          #sDot*du/ds on mid-levels
    dUdt[0,1:-1,:] = -U[0,1:-1,:] * Da(U[0,1:-1,:]) - tmp \
    + 1./U[3,1:-1,:] * (((U[4,1:-1,:]-U[4,0:-2,:])/ds) \
    * Da((pi[:-1,:]+pi[1:,:])/2. + U[5,1:-1,:]) \
    - (U[3,:,:] + Ds(U[5,:,:]))[1:-1,:] \
    * Da((U[4,0:-2,:]+U[4,1:-1,:])/2.)) \
    + HVa(U[0,1:-1,:]) \
    + HVs(U[0,:,:])                                     #du/dt (mid-levels)
    
    tmp = (U[1,1:-1,:] - U[1,0:-2,:]) / ds             #dw/ds on mid-levels
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.      #dw/ds on interior interfaces
    tmp = sDot[1:-1,:] * tmp             #sDot*dw/ds on interior interfaces
    dUdt[1,1:-2,:] = -uInt[1:-1,:] * Da(U[1,1:-2,:]) - tmp \
    + g * ((U[5,2:-1,:]-U[5,1:-2,:])/ds) / dpidsInt[1:-1,:] \
    + HVa(U[1,1:-2,:]) \
    + HVs(U[1,0:-1,:])                         #dw/dt (interior interfaces)
    if not heightCoord:
        dUdt[1,0,:] = -uInt[0,:] * Da(U[1,0,:]) \
        + g * ((U[5,1,:]-U[5,0,:])/ds) / dpidsInt[0,:] \
        + HVa(U[1,0,:])                                        #dw/dt (top)

    tmp = (U[2,1:,:] - U[2,0:-1,:]) / ds              #dth/ds on interfaces
    tmp = sDot * tmp                             #sDot*dth/ds on interfaces
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.         #sDot*dth/ds on mid-levels
    dUdt[2,1:-1,:] = -U[0,1:-1,:] * Da(U[2,1:-1,:]) - tmp \
    + HVa(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    + HVs(U[2,:,:] - thetaBar)                         #dth/dt (mid-levels)

    tmp = sDot * dpidsInt                        #sDot*dpi/ds on interfaces
    tmp = (tmp[1:,:] - tmp[0:-1,:]) / ds   #d(sDot*dpi/ds)/ds on mid-levels
    dUdt[3,1:-1,:] = -Da(U[3,1:-1,:] * U[0,1:-1,:]) - tmp \
    + HVa(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    + HVs(U[3,:,:] - dpidsBar)                   #d(dpi/ds)/dt (mid-levels)

    if not hydrostatic:
        tmp = (U[4,1:-1,:] - U[4,0:-2,:]) / ds       #dphi/ds on mid-levels
        tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.#dphi/ds on interior interfaces
        tmp = sDot[1:-1,:] * tmp       #sDot*dphi/ds on interior interfaces
        dUdt[4,1:-2,:] = -uInt[1:-1,:] * Da(U[4,1:-2,:]) - tmp \
        + g * U[1,1:-2,:] \
        + HVa(U[4,1:-2,:] - phiBar[1:-1,:]) \
        + HVs(U[4,0:-1,:] - phiBar)          #dphi/dt (interior interfaces)
        if not heightCoord:
            dUdt[4,0,:] = -uInt[0,:] * Da(U[4,0,:]) \
            + g * U[1,0,:] \
            + HVa(U[4,0,:] - phiBar[0,:])                    #dphi/dt (top)

    return dUdt

###########################################################################

def printMinAndMax(t, et, U, pi, thetaBar, dpidsBar, phiBar):
    
    print("t = {0:6d},  et = {1:6.2f},  |massDiff| = {8:+.1e},  \
MIN:  u = {2:+.2e},  \
w = {3:+.2e},  th = {4:+.2e},  pi = {5:+.2e},  \
phi = {6:+.2e},  P = {7:+.2e}" \
    . format(np.int(np.round(t)) \
    , et \
    , np.min(U[0,1:-1,:]) \
    , np.min(U[1,0:-1,:]) \
    , np.min(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.min(pi - pi0) \
    , np.min(U[4,0:-1,:] - phiBar) \
    , np.min(U[5,1:-1,:]) \
    , np.abs(np.sum(U[3,1:-1,:] / g * ds * dx) - mass0) / mass0))

    print("                                                   \
MAX:  u = {0:+.2e},  \
w = {1:+.2e},  th = {2:+.2e},  pi = {3:+.2e},  \
phi = {4:+.2e},  P = {5:+.2e}\n" \
    . format(np.max(U[0,1:-1,:]) \
    , np.max(U[1,0:-1,:]) \
    , np.max(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.max(pi - pi0) \
    , np.max(U[4,0:-1,:] - phiBar) \
    , np.max(U[5,1:-1,:])))

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,3) == 0):

        U, pi, thetaBar, dpidsBar, phiBar = setGhostNodes(U)
        piSurf = pi[-1,:].copy()
        
        if heightCoord:
            #Re-map w and phi:
            tmp = U[4,0:-1,:].copy()
            U[[1,4],0:-1,:] = common.verticalRemap(U[[1,4],0:-1,:] \
            , tmp, phi0, Vinterfaces)
            #Re-set geopotential:
            U[4,0:-1,:] = phi0.copy()
            #Avg phi to mid-levels:
            phi = (tmp[0:-1,:] + tmp[1:,:])/2.
            phi = np.vstack((2.*tmp[0,:] - phi[0,:] \
            , phi \
            , 2.*tmp[-1,:] - phi[-1,:]))
            #Re-map u and theta and dpi/ds:
            U[[0,2,3],:,:] = common.verticalRemap(U[[0,2,3],:,:] \
            , phi, phi0mid, np.zeros((3,nLev+2,nCol)))
        else:
            #New coordinate on interfaces:
            piNew = A(ssInt) * Po + B(ssInt) * np.tile(piSurf, (nLev+1, 1))
            #Re-map w and phi:
            U[[1,4],0:-1,:] = common.verticalRemap(U[[1,4],0:-1,:] \
            , pi, piNew, Vinterfaces)
            #Avg pi to mid-levels:
            pi = (pi[0:-1,:] + pi[1:,:])/2.
            pi = np.vstack((2.*piTop - pi[0,:] \
            , pi \
            , 2.*piSurf - pi[-1,:]))
            #New coordinate on mid-levels:
            piNew = A(ssMid) * Po + B(ssMid) * np.tile(piSurf, (nLev+2, 1))
            #Re-map u and theta:
            U[[0,2],:,:] = common.verticalRemap(U[[0,2],:,:] \
            , pi, piNew, Vmidlevels)
            #Re-set the pseudo-density:
            U[3,:,:] = dpidsBar.copy()
    
    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        if contourFromSaved:
            U[0:5,:,:] = np.load(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy')
        
        U, pi, thetaBar, dpidsBar, phiBar = setGhostNodes(U)

        printMinAndMax(t, time.time()-et, U, pi, thetaBar, dpidsBar, phiBar)

        et = time.time()
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours or plotBackgroundState:
            figName = "{0:04d}.png" . format(np.int(np.round(t) + 1e-12))
            contourSomething(U, t, pi, thetaBar, dpidsBar, phiBar \
            , whatToPlot, figName, contours)
            if (i == nTimesteps) and (not plotBackgroundState):
                #Save contour plots of variables at final time:
                for v in ["u","w","theta","dpids","phi","P","pi","T","rho"]:
                    contourSomething(U, t, pi, thetaBar, dpidsBar, phiBar \
                    , v, v+".png", 20)
    
    if contourFromSaved:
        t = t + dt
    else:
        if rks == 3:
            t, U = rk.rk3(t, U, odefun, dt, q1, q2)
        elif rks == 4:
            t, U = rk.rk4(t, U, odefun, dt, q1, q2, q3, q4)
        else:
            raise ValueError("Please use RK3 or RK4 for this problem.  " \
            + "RK1 and RK2 are unstable in time, since their stability " \
            + "regions have no imaginary axis coverage.")

###########################################################################
