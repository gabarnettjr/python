import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../../../site-packages')
from gab import rk
from gab.nonhydro import common

###########################################################################

#Switches to be set by user:

saveArrays          = True
saveContours        = True
contourFromSaved    = False
plotNodesAndExit    = False
plotBackgroundState = False

###########################################################################

def printHelp():
    sys.exit("\n\
REQUIRED ARGUMENTS\n\n\
arg 1 (name of test case)\n\
        risingBubble\n\
        densityCurrent\n\
        inertiaGravityWaves\n\
        tortureTest\n\
        scharMountainWaves\n\n\
arg 2 (vertical frame of reference, Eulerian or Lagrangian)\n\
        vEul\n\
        vLag\n\n\
arg 3 (refinement level)\n\
        0\n\
        1\n\
        2\n\
        3\n\
        4\n\n\
OPTIONAL ARGUMENTS\n\n\
arg 4 (what to plot)\n\
        u\n\
        w\n\
        theta\n\
        dpids\n\
        phi\n\
        P\n\n\
arg 5 (contour levels)\n\
        number of contours (integer)\n\
        range of contours (using np.arange or np.linspace for example)\n\n\
EXAMPLE\n\n\
python main.py risingBubble vEul 1 P np.arange(-51,51,2)")

###########################################################################

#Parse required inputs:

try:
    testCase = sys.argv[1]
    #Choose "vEul" or "vLag":
    if sys.argv[2] == "vLag":
        verticallyLagrangian = True
    elif sys.argv[2] == "vEul":
        verticallyLagrangian = False
    else:
        raise ValueError("The second argument should be either 'vLag' " \
        + "or 'vEul'.")
    #Choose 0, 1, 2, 3, or 4:
    refinementLevel = np.int64(sys.argv[3])
except:
    printHelp()

###########################################################################

#Parse optional inputs:

start = 4
try:
    whatToPlot = sys.argv[4]
except:
    if testCase == "scharMountainWaves":
        whatToPlot = "w"
        contours = np.arange(-.725, .775, .05)
    elif testCase == "risingBubble":
        whatToPlot = "theta"
        contours = np.arange(-.15, 2.25, .1)
    elif testCase == "inertiaGravityWaves":
        whatToPlot = "theta"
        contours = np.arange(-.0015, .0037, .0002)
    elif testCase == "densityCurrent":
        whatToPlot = "theta"
        contours = np.arange(-17.5, 2.5, 1)
    elif testCase == "steadyState":
        whatToPlot = "P"
        contours = 20
    elif testCase == "tortureTest":
        whatToPlot = "u"
        contours = np.arange(-1.05, 1.15, .1)
    else:
        raise ValueError("Invalid testCase string.")
else:
    start = 5
    try:
        contours = eval(sys.argv[5])
    except:
        contours = 20
    else:
        start = 6

###########################################################################

#Get string for saving results:

saveString = "./results/" + testCase + "_"

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

#Definitions of atmospheric constants:
Cp, Cv, Rd, g, Po, th0, N = common.constants(testCase)

#Test-specific parameters describing domain and initial perturbation:
left, right, bottom, top, dx, nLev, dt, tf, saveDel \
, zSurf, thetaPtb \
= common.domainParameters(testCase, refinementLevel, g, Cp, th0)




#TEMPORARY OVER-WRITE OF SOME VARIABLES FOR TESTING PURPOSES:
# tf = 50.
# saveDel = 5
# def zSurf(x):
#     return np.zeros(np.shape(x))




#Some other important parameters:
nCol = np.int(np.round((right - left) / dx))             #number of columns
t = 0.                                                        #initial time
nTimesteps = np.int(np.round(tf / dt))                #number of time-steps
x = np.linspace(left+dx/2, right-dx/2, nCol)        #array of x-coordinates

#Initial hydrostatic background state functions (given in test case):
potentialTemperature, potentialTemperatureDerivative \
, exnerPressure, inverseExnerPressure \
= common.hydrostaticProfiles(testCase, th0, g, Cp, N)

###########################################################################

#Functions for vertical derivatives (3-pt finite-differences):

def Ds(U):
    V = np.zeros(np.shape(U))
    V[0,:] = -3./2./ds*U[0,:] + 2./ds*U[1,:] - 1./2./ds*U[2,:]
    V[1:-1,:] = (U[2:,:] - U[0:-2,:]) / (2.*ds)
    V[-1,:] = 3./2./ds*U[-1,:] - 2./ds*U[-2,:] + 1./2./ds*U[-3,:]
    return V

def HVs(U):
    return 0. * 2.**-14./ds * (U[0:-2,:] - 2.*U[1:-1,:] + U[2:,:])

###########################################################################

#Weights and functions for lateral derivatives (7-pt finite-differences):

wd1 = np.array([-1./60., .15, -.75,   0., .75, -.15, 1./60.]) / dx
wd6 = np.array([     1., -6.,  15., -20., 15.,  -6.,     1.]) / dx**6.

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

def HVa(U):
    d = len(np.shape(U))
    if d == 1:
        U = np.hstack((U[[-3,-2,-1]], U, U[[0,1,2]]))
        U = wd6[0]*U[0:-6] + wd6[1]*U[1:-5] + wd6[2]*U[2:-4] \
          + wd6[3]*U[3:-3] \
          + wd6[4]*U[4:-2] + wd6[5]*U[5:-1] + wd6[6]*U[6:]
    elif d == 2:
        U = np.hstack((U[:,[-3,-2,-1]], U, U[:,[0,1,2]]))
        U = wd6[0]*U[:,0:-6] + wd6[1]*U[:,1:-5] + wd6[2]*U[:,2:-4] \
          + wd6[3]*U[:,3:-3] \
          + wd6[4]*U[:,4:-2] + wd6[5]*U[:,5:-1] + wd6[6]*U[:,6:]
    else:
        raise ValueError("U should have either one or two dimensions.")
    return 0. * 2.**-1.*dx**5. * U            #multiply by dissipation parameter

###########################################################################

#Equally spaced array of vertical coordinate values (s):

piTop  = Po * exnerPressure(top)  ** (Cp/Rd)            #HS pressure at top
piSurf = Po * exnerPressure(zSurf(x)) ** (Cp/Rd)    #HS pressure at surface
sTop = piTop / Po                             #value of s on upper boundary
ds = (1. - sTop) / nLev                                            #delta s
s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)         #equally-spaced s values

###########################################################################

#Prescribed functions for setting hybrid coordinate levels:

def A(s):
    return (1. - s) / (1. - sTop) * sTop

def B(s):
    return (s - sTop) / (1. - sTop)

def Aprime(s):
    return -sTop / (1. - sTop)

def Bprime(s):
    return 1. / (1. - sTop)

ss = np.tile(s, (nCol, 1)).T                          #s-mesh on mid-levels
ssInt = (ss[0:-1,:] + ss[1:,:]) / 2.                  #s-mesh on interfaces

###########################################################################

#Iterate to get initial vertical height levels zz:

#dpi/ds defined using vertical coordinate on interior mid-levels:
dpids = Aprime(ss[1:-1,:]) * Po \
+ Bprime(ss[1:-1,:]) * np.tile(piSurf, (nLev,1))

#integrate dpids to get HS pressure, as will be done during time-stepping:
tmp = piTop * np.ones((nCol))
pi = np.zeros((nLev+1, nCol))
pi[0,:] = tmp.copy()
for i in range(nLev):
    tmp = tmp + dpids[i,:] * ds
    pi[i+1,:] = tmp.copy()

zz0 = inverseExnerPressure((pi/Po) ** (Rd/Cp))               #initial guess
zz = zz0.copy()
xx = np.tile(x, (nLev+1,1))

pi = (pi[:-1,:] + pi[1:,:]) / 2.             #avg pi to interior mid-levels

xxMid = np.tile(x, (nLev, 1))

for j in range(10):
    zzMid = (zz[:-1,:] + zz[1:,:]) / 2.
    theta = potentialTemperature(zzMid) + thetaPtb(xxMid, zzMid)
    integrand = dpids * Rd * (pi/Po)**(Rd/Cp) * theta / g / pi
    tmp = zz[-1,:].copy()
    for i in range(nLev):
        tmp = tmp + integrand[-(i+1),:] * ds
        zz[-(i+2),:] = tmp.copy()
    # plt.figure()
    # plt.contourf(xx, zz0, zz-zz0, 20)
    # plt.colorbar()
    # plt.show()

###########################################################################

if plotNodesAndExit:
    plt.figure()
    plt.plot(xx.flatten(), zz.flatten(), marker=".", linestyle="none")
    plt.plot(x, zz[-1,:], color="red", linestyle="-")
    plt.plot(x, zz[0,:], color="red", linestyle="-")
    plt.plot([left,left], [zSurf(left),top], color="red" \
    , linestyle="-")
    plt.plot([right,right], [zSurf(right),top], color="red" \
    , linestyle="-")
    plt.axis("image")
    plt.show()
    sys.exit("Finished plotting.")

###########################################################################

#Assignment of hydrostatic background states and initial perturbations:

theta0 = potentialTemperature(zzMid) + thetaPtb(xxMid, zzMid)
phi0   = g * zz
dpids0 = Aprime(ss) * Po + Bprime(ss) * np.tile(piSurf, (nLev+2,1))

###########################################################################

#Assignment of initial conditions:

U = np.zeros((6, nLev+2, nCol))

if testCase == "inertiaGravityWaves":
    U[0,:,:] \
    = 20. * np.ones((nLev+2, nCol))     #horizontal velocity u (mid-levels)

U[1,0:-1,:] = np.zeros((nLev+1, nCol))    #vertical velocity w (interfaces)

U[2,1:-1,:] = theta0.copy()       #potential temperature theta (mid-levels)

U[3,:,:] = dpids0.copy()                 #pseudo-density dpids (mid-levels)

U[4,0:-1,:] = phi0.copy()                    #geopotential phi (interfaces)

U[5,:,:] = np.zeros((nLev+2, nCol))   #pressure perturbation P (mid-levels)

###########################################################################

mass0 = np.sum(U[3,1:-1,:] / g * ds * dx)                     #initial mass

###########################################################################

#Initialize a figure of the appropriate size:

if saveContours:
    if testCase == "inertiaGravityWaves":
        fig = plt.figure(figsize = (18,3))
    else:
        fig = plt.figure(figsize = (18,14))

###########################################################################

#Create and save a contour plot of the field specified by whatToPlot:

def contourSomething(U, t, thetaBar, pi, dpidsBar, phiBar):

    if plotBackgroundState:
        if whatToPlot == "u":
            tmp = np.zeros((nLev, nCol))
        elif whatToPlot == "w":
            tmp = np.zeros((nLev+1, nCol))
        elif whatToPlot == "theta":
            tmp = thetaBar[1:-1,:].copy()
        elif whatToPlot == "dpids":
            tmp = dpidsBar[1:-1,:].copy()
        elif whatToPlot == "phi":
            tmp = phiBar.copy()
        elif whatToPlot == "P":
            tmp = (pi[:-1,:] + pi[1:,:]) / 2.
        else:
            raise ValueError("Invalid whatToPlot string.")
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
            # tmp = U[4,0:-1,:] - phi0
            tmp = U[4,0:-1,:] - phiBar
        elif whatToPlot == "P":
            tmp = U[5,1:-1,:]
        else:
            raise ValueError("Invalid whatToPlot string.")

    if (whatToPlot == "phi") | (whatToPlot == "w"):
        xxTmp = xx
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
    or (testCase == "scharMountainWaves"):
        plt.colorbar(orientation="horizontal")
    elif (testCase == "densityCurrent"):
        plt.axis("image")
        plt.colorbar(orientation="horizontal")
    else:
        plt.axis("image")
        plt.colorbar(orientation="vertical")
    plt.axis([left-250., right+250. \
    , np.min(zz[-1,:])-250., np.max(zz[0,:])+250.])
    if plotBackgroundState:
        plt.show()
        sys.exit("\nDone plotting the requested background state.")
    else:
        fig.savefig( "{0:04d}.png".format(np.int(np.round(t)+1e-12)) \
        , bbox_inches="tight" )                       #save figure as a png

###########################################################################

#Initialize the output array for the vertical re-map function:

if verticallyLagrangian:
    Vmidlevels  = np.zeros((2, nLev+2, nCol))
    Vinterfaces = np.zeros((2, nLev+1, nCol))

###########################################################################

#This will be used inside the setGhostNodes() function to quickly find
#background states on the moving vertical levels:

def fastBackgroundStates(phi, dpids):
    
    #Get thetaBar on mid-levels from the given background state function:
    tmp = (phi[:-1,:] + phi[1:,:]) / 2.
    tmp = np.vstack((2.*phi[0,:] - tmp[0,:] \
    , tmp \
    , 2.*phi[-1,:] - tmp[-1,:]))
    thetaBar = potentialTemperature(tmp/g)

    #Integrate dpi/ds to get hydrostatic pressure pi at interfaces:
    tmp = piTop * np.ones((nCol))
    pi = np.zeros((nLev+1, nCol))
    pi[0,:] = tmp.copy()
    for i in range(nLev):
        tmp = tmp + dpids[i,:] * ds
        pi[i+1,:] = tmp.copy()

    #Get background state for pseudo-density dpi/ds on mid-levels:
    dpidsBar = Aprime(ss) * Po \
    + Bprime(ss) * np.tile(pi[-1,:], (nLev+2, 1))

    #Integrate EOS to get background state for geopotential on interfaces:
    tmp = phi[-1,:].copy()
    phiBar = np.zeros((nLev+1, nCol))
    phiBar[-1,:] = tmp.copy()
    integrand = Rd * thetaBar[1:-1,:] * dpidsBar[1:-1,:] / Po**(Rd/Cp) \
    / ((pi[:-1,:]+pi[1:,:])/2.)**(Cv/Cp)
    for i in range(nLev):
        tmp = tmp + integrand[-(i+1),:] * ds
        phiBar[-(i+2),:] = tmp.copy()

    return thetaBar, pi, dpidsBar, phiBar

###########################################################################

def setGhostNodes(U):

    #Get background states:
    thetaBar, pi, dpidsBar, phiBar \
    = fastBackgroundStates(U[4,:-1,:], U[3,1:-1,:])

    #Extrapolate dpids to bottom ghost nodes:
    U[3,-1,:] = 2. * (U[3,-2,:]-dpidsBar[-2,:]) - (U[3,-3,:]-dpidsBar[-3,:]) \
    + dpidsBar[-1,:]

    #Extrapolate dpids to top ghost nodes:
    U[3,0,:] = 2. * (U[3,1,:]-dpidsBar[1,:]) - (U[3,2,:]-dpidsBar[2,:]) \
    + dpidsBar[0,:]

    #Extrapolate theta to bottom ghost nodes:
    U[2,-1,:] = 2. * (U[2,-2,:]-thetaBar[-2,:]) - (U[2,-3,:]-thetaBar[-3,:]) \
    + thetaBar[-1,:]

    #Extrapolate theta to top ghost nodes:
    U[2,0,:] = 2. * (U[2,1,:]-thetaBar[1,:]) - (U[2,2,:]-thetaBar[2,:]) \
    + thetaBar[0,:]
    
    #Get pressure perturbation on mid-levels using equation of state:
    U[5,1:-1,:] = (-U[3,1:-1,:] / ((U[4,1:-1,:]-U[4,0:-2,:])/ds) * Rd \
    * U[2,1:-1,:] / Po**(Rd/Cp)) ** (Cp/Cv) \
    - (pi[:-1,:]+pi[1:,:])/2.

    dphida = Da(U[4,-2,:])
    #set pressure perturbation on bottom ghost nodes using Neumann BC:
    dPda = Da(pi[-1,:]) + Da(3./2.*U[5,-2,:] - 1./2.*U[5,-3,:])
    dpids = 3./2.*U[3,-2,:] - 1./2.*U[3,-3,:]
    dphids = (3./2.*U[4,-2,:] - 2.*U[4,-3,:] + 1./2.*U[4,-4,:]) / ds
    RHS = (dPda * dphids - dphida * dpids) * dphida
    RHS = RHS / (g**2. + dphida**2.)
    U[5,-1,:] = U[5,-2,:] + ds * RHS
    #extrapolate u to bottom ghost nodes:
    U[0,-1,:] = 2.*U[0,-2,:] - U[0,-3,:]
    #get w on bottom boundary nodes:
    U[1,-2,:] = (U[0,-1,:]+U[0,-2,:])/2. / g * dphida
    # U[1,-2,:] = (3./2.*U[0,-2,:]-1./2.*U[0,-3,:]) / g * dphida

    #set pressure perturbation on top ghost nodes using zero Dirichlet BC:
    U[5,0,:] = -U[5,1,:]
    #extrapolate u to top ghost nodes:
    U[0,0,:] = 2.*U[0,1,:] - U[0,2,:]

    return U, thetaBar, pi, dpidsBar, phiBar

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
    U, thetaBar, pi, dpidsBar, phiBar = setGhostNodes(U)

    uInt = (U[0,:-1,:] + U[0,1:,:]) / 2.                   #u on interfaces
    dpidsInt = (U[3,:-1,:] + U[3,1:,:]) / 2.          #dpi/ds on interfaces

    #Get sDot:

    if verticallyLagrangian:
        sDot = np.zeros((nLev+1, nCol))
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
    * (Da((pi[:-1,:]+pi[1:,:])/2.) + Da(U[5,1:-1,:])) \
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
    dUdt[1,0,:] = -uInt[0,:] * Da(U[1,0,:]) \
    + g * ((U[5,1,:]-U[5,0,:])/ds) / dpidsInt[0,:] \
    + HVa(U[1,0,:])                                            #dw/dt (top)

    tmp = (U[2,1:,:] - U[2,0:-1,:]) / ds              #dth/ds on interfaces
    tmp = sDot * tmp                             #sDot*dth/ds on interfaces
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.         #sDot*dth/ds on mid-levels
    dUdt[2,1:-1,:] = -U[0,1:-1,:] * Da(U[2,1:-1,:]) - tmp
    + HVa(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    + HVs(U[2,:,:] - thetaBar)                         #dth/dt (mid-levels)

    tmp = sDot * dpidsInt                         #sDot*dpids on interfaces
    tmp = (tmp[1:,:] - tmp[0:-1,:]) / ds    #d(sDot*dpids)/ds on mid-levels
    dUdt[3,1:-1,:] = -Da(U[3,1:-1,:] * U[0,1:-1,:]) - tmp \
    + HVa(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    + HVs(U[3,:,:] - dpidsBar)                    #d(dpids)/dt (mid-levels)

    tmp = (U[4,1:-1,:] - U[4,0:-2,:]) / ds           #dphi/ds on mid-levels
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.    #dphi/ds on interior interfaces
    tmp = sDot[1:-1,:] * tmp           #sDot*dphi/ds on interior interfaces
    dUdt[4,1:-2,:] = -uInt[1:-1,:] * Da(U[4,1:-2,:]) - tmp \
    + g * U[1,1:-2,:] \
    + HVa(U[4,1:-2,:] - phiBar[1:-1,:]) \
    + HVs(U[4,0:-1,:] - phiBar)              #dphi/dt (interior interfaces)
    dUdt[4,0,:] = -uInt[0,:] * Da(U[4,0,:]) \
    + g * U[1,0,:] \
    + HVa(U[4,0,:] - phiBar[0,:])                            #dphi/dt (top)

    return dUdt

###########################################################################

def printMinAndMax(t, et, U, thetaBar, dpidsBar, phiBar):
    
    print("t = {0:5d},  et = {1:6.2f},  MIN:  u = {2:+.2e},  \
w = {3:+.2e},  th = {4:+.2e},  dpids = {5:+.2e},  \
phi = {6:+.2e},  P = {7:+.2e}" \
    . format(np.int(np.round(t)) \
    , et \
    , np.min(U[0,1:-1,:]) \
    , np.min(U[1,0:-1,:]) \
    , np.min(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.min(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    , np.min(U[4,0:-1,:] - phiBar) \
    , np.min(U[5,1:-1,:])))

    print("                          MAX:  u = {0:+.2e},  \
w = {1:+.2e},  th = {2:+.2e},  dpids = {3:+.2e},  \
phi = {4:+.2e},  P = {5:+.2e}\n" \
    . format(np.max(U[0,1:-1,:]) \
    , np.max(U[1,0:-1,:]) \
    , np.max(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.max(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    , np.max(U[4,0:-1,:] - phiBar) \
    , np.max(U[5,1:-1,:])))

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,3) == 0) and (testCase != "inertiaGravityWaves") \
    and (testCase != "scharMountainWaves"):

        U, thetaBar, pi, dpidsBar, phiBar = setGhostNodes(U)
        piSurf = pi[-1,:].copy()

        #new coordinate on interfaces:
        piNew = A(ssInt) * Po \
        + B(ssInt) * np.tile(piSurf, (nLev+1, 1))

        #re-map w and phi:
        U[[1,4],0:-1,:] = common.verticalRemap(U[[1,4],0:-1,:] \
        , pi, piNew, Vinterfaces)
        
        #avg pi to mid-levels:
        pi = (pi[0:-1,:] + pi[1:,:])/2.
        pi = np.vstack((2.*piTop - pi[0,:] \
        , pi \
        , 2.*piSurf - pi[-1,:]))

        #new coordinate on mid-levels:
        piNew = A(ss) * Po \
        + B(ss) * np.tile(piSurf, (nLev+2, 1))

        #re-map u and theta:
        U[[0,2],:,:] = common.verticalRemap(U[[0,2],:,:] \
        , pi, piNew, Vmidlevels)
        
        #re-set the pseudo-density:
        U[3,:,:] = dpidsBar.copy()
    
    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        if contourFromSaved :
            U[0:5,:,:] = np.load(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy')
        
        U, thetaBar, pi, dpidsBar, phiBar = setGhostNodes(U)

        printMinAndMax(t, time.time()-et, U, thetaBar, dpidsBar, phiBar)

        # print("|massDiff| = {0:.2e}" \
        # . format(np.abs(np.sum(U[3,1:-1,:] \
        # / g * ds * dx) - mass0) / mass0))
        
        et = time.time()
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours or plotBackgroundState:
            contourSomething(U, t, thetaBar, pi, dpidsBar, phiBar)
    
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
