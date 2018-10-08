import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../../../site-packages')
from gab import rk, phs1
from gab.nonhydro import common

###########################################################################

#This block contains the only variables that the user should be required
#to modify when running the code, unless they want to add a new test case.

#Choose "risingBubble", "densityCurrent", "inertiaGravityWaves", or
#"steadyState":
testCase = "risingBubble"

#Choose True or False:
verticallyLagrangian = False

#Choose 0, 1, 2, 3, or 4:
refinementLevel = 1

#Switches to control what happens:
saveArrays          = True
saveContours        = True
contourFromSaved    = False
plotNodesAndExit    = False
plotBackgroundState = False

#Choose which variable to plot ("u", "w", "theta", "dpids", "phi", "P"):
whatToPlot = "P"

#Choose either a number of contours, or a range of contours:
contours = 20
# contours = np.arange(-20.5, 1.5, 1)

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
, zSurfFunc, thetaPtb \
= common.domainParameters(testCase, refinementLevel, g, Cp, th0)




#TEMPORARY OVER-WRITE OF SOME VARIABLES FOR TESTING PURPOSES:
# tf = 50.
# saveDel = 5
# def zSurfFunc(x):
#     return np.zeros(np.shape(x))




#Some other important parameters:
nCol = np.int(np.round((right - left) / dx))             #number of columns
t = 0.                                                        #initial time
nTimesteps = np.int(np.round(tf / dt))                #number of time-steps
x = np.linspace(left+dx/2, right-dx/2, nCol)        #array of x-coordinates
zSurf = zSurfFunc(x)                 #array of values along bottom boundary

#ones and zeros to avoid repeated initialization getting background states:
e = np.ones((nLev+2, nCol))
null = np.zeros((nLev+2, nCol))

#Hydrostatic background state functions:
potentialTemperature, potentialTemperatureDerivative \
, exnerPressure, inverseExnerPressure \
= common.hydrostaticProfiles(testCase, th0, g, Cp, N, e, null)

###########################################################################

#Some approximation weights:

Wa = phs1.getPeriodicDM(z=x, x=x, m=1 \
, phs=11, pol=5, stc=11, period=right-left)

Whva = phs1.getPeriodicDM(z=x, x=x, m=6 \
, phs=11, pol=5, stc=11, period=right-left)
Whva = 2.**-9.*300. * dx**5. * Whva

def Ds(U):
    V = np.zeros(np.shape(U))
    V[0,:] = (-3./2.*U[0,:] + 2.*U[1,:] - 1./2.*U[2,:]) / ds
    V[1:-1,:] = (U[2:,:] - U[0:-2,:]) / (2.*ds)
    V[-1,:] = (3./2.*U[-1,:] - 2.*U[-2,:] + 1./2.*U[-3,:]) / ds
    return V

def HVs(U):
    return 2.**-22.*300. * (U[0:-2,:] - 2.*U[1:-1,:] + U[2:,:]) / ds

def Da(U):
    return Wa.dot(U.T).T

def HVa(U):
    return Whva.dot(U.T).T

###########################################################################

#Equally spaced array of vertical coordinate values (s):

piTop  = exnerPressure(top, 1., 0.)
pTop  = Po * piTop  ** (Cp/Rd)                 #hydrostatic pressure at top
piSurf = exnerPressure(zSurf, e[0,:], null[0,:])
pSurf = Po * piSurf ** (Cp/Rd)             #hydrostatic pressure at surface
sTop = pTop / Po                              #value of s on upper boundary
ds = (1. - sTop) / nLev
s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)

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

#Iterate to get levels zz:

zz0 = inverseExnerPressure((A(ssInt) \
+ B(ssInt) * np.tile(pSurf,(nLev+1,1)) / Po) ** (Rd/Cp))     #initial guess

zz = zz0.copy()

xx = np.tile(x, (nLev+1,1))

#dp/ds defined using vertical coordinate on interior mid-levels:
dpds = Aprime(ss[1:-1,:]) * Po + Bprime(ss[1:-1,:]) * np.tile(pSurf, (nLev,1))

#integrate dpds to get p, as will be done during time-stepping:
tmp = pTop * np.ones((nCol))
p = np.zeros((nLev+1, nCol))
p[0,:] = tmp.copy()
for i in range(nLev):
    tmp = tmp + dpds[i,:] * ds
    p[i+1,:] = tmp.copy()
p = (p[:-1,:] + p[1:,:]) / 2.                     #p on interior mid-levels

tmp1 = np.ones((nLev, nCol))
tmp2 = np.zeros((nLev, nCol))

xxMid = np.tile(x, (nLev, 1))

for j in range(10):
    zzMid = (zz[:-1,:] + zz[1:,:]) / 2.
    theta = potentialTemperature(zzMid, tmp1, tmp2) + thetaPtb(xxMid, zzMid)
    integrand = -dpds * Rd * (p/Po)**(Rd/Cp) * theta / g / p
    tmp = zz[0,:].copy()
    for i in range(nLev):
        tmp = tmp + integrand[i,:] * ds
        zz[i+1,:] = tmp.copy()
    # plt.figure()
    # plt.contourf(xx, zz0, zz-zz0, 20)
    # plt.colorbar()
    # plt.show()

###########################################################################

#Check that the pressure is okay:

# #thing = A(ss[1:-1,:]) * Po + B(ss[1:-1,:]) * np.tile(pSurf,(nLev,1))
# thing = Aprime(ss[1:-1,:])*Po + Bprime(ss[1:-1,:])*np.tile(pSurf,(nLev,1))
# pHydro = np.zeros((nLev+1, nCol))
# pHydro[0,:] = pTop * np.ones((nCol))
# tmp = pHydro[0,:].copy()
# for i in range(nLev):
#     tmp = tmp + thing[i,:] * ds
#     pHydro[i+1,:] = tmp.copy()
# thing = (pHydro[0:-1,:] + pHydro[1:,:]) / 2.
# thing = thing - Po * exnerPressure(zz[1:-1,:],e,null) ** (Cp/Rd)
# plt.figure()
# plt.contourf(xx[1:-1,:], zz[1:-1,:], thing, 20)
# plt.colorbar()
# plt.show()
# sys.exit("\ndone for now\n")

###########################################################################

if plotNodesAndExit:
    plt.figure()
    plt.plot(xx.flatten(), zz.flatten(), marker=".", linestyle="none")
    plt.plot(x, zz[-1,:], color="red", linestyle="-")
    plt.plot(x, top*np.ones(np.shape(x)), color="red", linestyle="-")
    plt.plot([left,left], [zSurfFunc(left),top], color="red" \
    , linestyle="-")
    plt.plot([right,right], [zSurfFunc(right),top], color="red" \
    , linestyle="-")
    plt.axis("image")
    plt.show()
    sys.exit("Finished plotting.")

###########################################################################

#Assignment of hydrostatic background states and initial perturbations:

theta0 = potentialTemperature(zzMid, tmp1, tmp2) + thetaPtb(xxMid,zzMid)

# piBar = exnerPressure(zz, e, null)
# piPtb = np.zeros((nLev+2, nCol))
# Tbar = piBar * thetaBar
# Tptb = (piBar + piPtb) * (thetaBar + thetaPtb(xx,zz)) - Tbar
# Pbar = Po * piBar ** (Cp/Rd)
# Pptb = Po * (piBar + piPtb) ** (Cp/Rd) - Pbar
# rhoBar = Pbar / Rd / Tbar
# rhoPtb = (Pbar + Pptb) / Rd / (Tbar + Tptb) - rhoBar

phi0 = g * zz

dpids0 = Aprime(ss) * Po + Bprime(ss) * np.tile(pSurf, (nLev+2,1))

###########################################################################

#Assignment of initial conditions:

U = np.zeros((6, nLev+2, nCol))

if testCase == "inertiaGravityWaves":
    U[0,:,:] =  20. * np.ones((nLev+2, nCol))          #horizontal velocity

U[1,0:-1,:] = np.zeros((nLev+1, nCol))     #vertical velocity on interfaces

U[2,1:-1,:] = theta0.copy()                          #potential temperature

U[3,:,:] = dpids0.copy()                                 #pseudo-density

U[4,0:-1,:] = phi0.copy()                    #geopotential on interfaces

U[5,:,:] = np.zeros((nLev+2, nCol))                  #pressure perturbation

mass0 = np.sum(U[3,1:-1,:] / g * ds * dx)

###########################################################################

#Initialize a figure of the appropriate size:

if saveContours:
    if testCase == "inertiaGravityWaves":
        fig = plt.figure(figsize = (18,3))
    else:
        fig = plt.figure(figsize = (18,14))

###########################################################################

#Create and save a contour plot of the field specified by whatToPlot:

def contourSomething(U, t, pHydro, dpidsBar, thetaBar):

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
            tmp = phi0.copy()
        elif whatToPlot == "P":
            tmp = (pHydro[:-1,:] + pHydro[1:,:]) / 2.
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
            tmp = U[4,0:-1,:] - phi0
        elif whatToPlot == "P":
            tmp = U[5,1:-1,:]
        else:
            raise ValueError("Invalid whatToPlot string.")

    if (whatToPlot == "phi") | (whatToPlot == "w"):
        xxTmp = xx
        zzTmp = U[4,0:-1,:] / g                          #changing z-levels
    else:
        xxTmp = (xx[:-1,:] + xx[1:,:]) / 2.
        zzTmp = (U[4,0:-2,:]+U[4,1:-1,:])/2. / g
    
    plt.clf()
    plt.contourf(xxTmp, zzTmp, tmp, contours)
    plt.plot(x, zz[-1,:], linestyle="-", color="red")
    plt.plot(x, top*np.ones(np.shape(x)), linestyle="-", color="red")
    if testCase == "inertiaGravityWaves":
        plt.colorbar(orientation="horizontal")
    elif testCase == "densityCurrent":
        plt.axis("image")
        plt.colorbar(orientation="horizontal")
    else:
        plt.axis("image")
        plt.colorbar(orientation="vertical")
    # if plotBackgroundState:
    #     plt.show()
    #     sys.exit("\nDone plotting the requested background state.")
    # else:
    fig.savefig( "{0:04d}.png".format(np.int(np.round(t)+1e-12)) \
    , bbox_inches="tight" )                       #save figure as a png

###########################################################################

#Initialize the output array for the vertical re-map function:

if verticallyLagrangian:
    V = np.zeros((2, nLev+2, nCol))

###########################################################################

#This will be used inside the setGhostNodes() function to quickly find
#background states on possibly changing vertical levels:

def fastBackgroundStates(zz, dpids):
    
    thetaBar = potentialTemperature(zz, e, null)

    # dthetaBarDz = potentialTemperatureDerivative(zz, e, null)
    # piBar = exnerPressure(zz, e, null)
    # Pbar = Po * piBar ** (Cp/Rd)
    # Tbar = piBar * thetaBar
    # rhoBar = Pbar / Rd / Tbar
    
    tmp = pTop * np.ones((nCol))
    pHydro = np.zeros((nLev+1, nCol))
    pHydro[0,:] = tmp.copy()
    for i in range(nLev):
        tmp = tmp + dpids[i,:] * ds
        pHydro[i+1,:] = tmp.copy()

    dpidsBar = Aprime(ss) * Po \
    + Bprime(ss) * np.tile(pHydro[-1,:], (nLev+2, 1))

    return thetaBar, pHydro, dpidsBar

###########################################################################

def setGhostNodes(U):

    #Avg phi to mid-levels:
    phi = (U[4,0:-2,:] + U[4,1:-1,:]) / 2.
    phi = np.vstack((2.*U[4,0,:] - phi[0,:] \
    , phi \
    , 2.*U[4,-2,:] - phi[-1,:]))

    #Get background states:
    thetaBar, pHydro, dpidsBar = fastBackgroundStates(phi/g, U[3,1:-1,:])

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
    - (pHydro[:-1,:]+pHydro[1:,:])/2.

    dphida = Da(U[4,-2,:])
    #set pressure perturbation on bottom ghost nodes using Neumann BC:
    dPda = Da(pHydro[-1,:]) + Da(3./2.*U[5,-2,:] - 1./2.*U[5,-3,:])
    dpids = 3./2.*U[3,-2,:] - 1./2.*U[3,-3,:]
    dphids = (3./2.*U[4,-2,:] - 2.*U[4,-3,:] + 1./2.*U[4,-4,:]) / ds
    RHS = (dPda * dphids - dphida * dpids) * dphida
    RHS = RHS / (g**2. + dphida**2.)
    U[5,-1,:] = U[5,-2,:] + ds * RHS
    #extrapolate u to bottom ghost nodes:
    U[0,-1,:] = 2.*U[0,-2,:] - U[0,-3,:]
    #get w on bottom boundary nodes:
    U[1,-2,:] = (3./2.*U[0,-2,:]-1./2.*U[0,-3,:]) / g * dphida

    #set pressure perturbation on top ghost nodes using Dirichlet BC:
    U[5,0,:] = -U[5,1,:]
    # dphida = Da(U[4,0,:])
    # #set pressure perturbation on top ghost nodes using Neumann BC:
    # dPda = Da(pHydro[0,:]) + Da(3./2.*U[5,1,:] - 1./2.*U[5,2,:])
    # dpids = 3./2.*U[3,1,:] - 1./2.*U[3,2,:]
    # dphids = (-3./2.*U[4,0,:] + 2.*U[4,1,:] - 1./2.*U[4,2,:]) / ds
    # RHS = (dPda * dphids - dphida * dpids ) * dphida
    # RHS = RHS / (g**2. + dphida**2.)
    # U[5,0,:] = U[5,1,:] - ds * RHS
    #extrapolate u to top ghost nodes:
    U[0,0,:] = 2.*U[0,1,:] - U[0,2,:]

    return U, thetaBar, pHydro, dpidsBar

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

sDotNull = np.zeros((nLev+1, nCol))

###########################################################################

#This describes the RHS of the system of ODEs in time that will be solved:

def odefun(t, U, dUdt):
    
    #Get some ghost node values and background states:
    U, thetaBar, pHydro, dpidsBar = setGhostNodes(U)

    dpidsInt = (U[3,0:-1,:] + U[3,1:,:]) / 2.          #dpids on interfaces

    #Get sDot:

    if verticallyLagrangian:
        sDot = sDotNull
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
    * (Da((pHydro[:-1,:]+pHydro[1:,:])/2.) + Da(U[5,1:-1,:])) \
    - (U[3,1:-1,:] + Ds(U[5,:,:])[1:-1,:]) \
    * Da((U[4,0:-2,:]+U[4,1:-1,:])/2.)) \
    + HVa(U[0,1:-1,:]) \
    + HVs(U[0,:,:])                                     #du/dt (mid-levels)
    
    tmp = (U[1,1:-1,:] - U[1,0:-2,:]) / ds             #dw/ds on mid-levels
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.      #dw/ds on interior interfaces
    tmp = sDot[1:-1,:] * tmp             #sDot*dw/ds on interior interfaces
    dUdt[1,1:-2,:] = -(U[0,1:-2,:]+U[0,2:-1,:])/2. * Da(U[1,1:-2,:]) - tmp \
    + g * ((U[5,2:-1,:]-U[5,1:-2,:])/ds) / dpidsInt[1:-1,:] \
    + HVa(U[1,1:-2,:]) \
    + HVs(U[1,0:-1,:])                         #dw/dt (interior interfaces)
    dUdt[1,0,:] = -(U[0,0,:]+U[0,1,:])/2. * Da(U[1,0,:]) \
    + g * ((U[5,1,:]-U[5,0,:])/ds) / dpidsInt[0,:] \
    + HVa(U[1,0,:])                                            #dw/dt (top)

    tmp = (U[2,1:,:] - U[2,0:-1,:]) / ds              #dth/ds on interfaces
    tmp = sDot * tmp                             #sDot*dth/ds on interfaces
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.         #sDot*dth/ds on mid-levels
    dUdt[2,1:-1,:] = -U[0,1:-1,:] * Da(U[2,1:-1,:]) - tmp \
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
    dUdt[4,1:-2,:] = -(U[0,1:-2,:]+U[0,2:-1,:])/2. * Da(U[4,1:-2,:]) - tmp \
    + g * U[1,1:-2,:] \
    + HVa(U[4,1:-2,:] - phi0[1:-1,:]) \
    + HVs(U[4,0:-1,:] - phi0)                #dphi/dt (interior interfaces)
    dUdt[4,0,:] = -(U[0,0,:]+U[0,1,:])/2. * Da(U[4,0,:]) \
    + g*U[1,0,:] \
    + HVa(U[4,0,:] - phi0[0,:])                              #dphi/dt (top)

    return dUdt

###########################################################################

def printMinAndMax(t, et, U, dpidsBar, thetaBar):
    
    print("t = {0:5d},  et = {1:6.2f},  MIN:  u = {2:+.2e},  \
w = {3:+.2e},  th = {4:+.2e},  dpids = {5:+.2e},  \
phi = {6:+.2e},  P = {7:+.2e}" \
    . format(np.int(np.round(t)) \
    , et \
    , np.min(U[0,1:-1,:]) \
    , np.min(U[1,0:-1,:]) \
    , np.min(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.min(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    , np.min(U[4,0:-1,:] - phi0) \
    , np.min(U[5,1:-1,:])))

    print("                          MAX:  u = {0:+.2e},  \
w = {1:+.2e},  th = {2:+.2e},  dpids = {3:+.2e},  \
phi = {4:+.2e},  P = {5:+.2e}\n" \
    . format(np.max(U[0,1:-1,:]) \
    , np.max(U[1,0:-1,:]) \
    , np.max(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.max(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    , np.max(U[4,0:-1,:] - phi0) \
    , np.max(U[5,1:-1,:])))

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,3) == 0) and (testCase != "inertiaGravityWaves"):

        U, thetaBar, pHydro, dpidsBar = setGhostNodes(U)
        pHydroSurf = pHydro[-1,:].copy()

        #new coordinate on interfaces:
        pHydroNew = A(ssInt) * Po \
        + B(ssInt) * np.tile(pHydroSurf, (nLev+1, 1))

        #re-map w and phi:
        U[[1,4],0:-1,:] = common.verticalRemap(U[[1,4],0:-1,:] \
        , pHydro, pHydroNew, np.zeros((2,nLev+1,nCol)))
        
        #move pHydro to mid-levels:
        pHydro = (pHydro[0:-1,:] + pHydro[1:,:])/2.
        pHydro = np.vstack((2.*pTop - pHydro[0,:] \
        , pHydro \
        , 2.*pHydroSurf - pHydro[-1,:]))

        #new coordinate on mid-levels:
        pHydroNew = A(ss) * Po \
        + B(ss) * np.tile(pHydroSurf, (nLev+2, 1))

        #re-map u and theta:
        U[[0,2],:,:] = common.verticalRemap(U[[0,2],:,:] \
        , pHydro, pHydroNew, V)
        
        #re-set the pseudo-density:
        U[3,:,:] = dpidsBar.copy()
    
    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        if contourFromSaved :
            U[0:5,:,:] = np.load(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy')
        
        U, thetaBar, pHydro, dpidsBar = setGhostNodes(U)

        printMinAndMax(t, time.time()-et, U, dpidsBar, thetaBar)

        # print("|massDiff| = {0:.2e}" \
        # . format(np.abs(np.sum(U[3,1:-1,:] \
        # / g * ds * dx) - mass0) / mass0))
        
        et = time.time()
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours or plotBackgroundState:
            contourSomething(U, t, pHydro, dpidsBar, thetaBar)
    
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
