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
refinementLevel = 2

#Switches to control what happens:
saveArrays          = True
saveContours        = True
contourFromSaved    = False
plotNodesAndExit    = False
plotBackgroundState = False

#Choose which variable to plot
#("u", "w", "theta", "dpids", "phi", "P"):
whatToPlot = "theta"

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
# tf = 100.
# saveDel = 10
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

#Equally spaced array of vertical coordinate values (s):

piTop  = exnerPressure(top, 1., 0.)
pTop  = Po * piTop  ** (Cp/Rd)                 #hydrostatic pressure at top
piSurf = exnerPressure(zSurf, e[0,:], null[0,:])
pSurf = Po * piSurf ** (Cp/Rd)             #hydrostatic pressure at surface
sTop = pTop / Po                              #value of s on upper boundary
ds = (1. - sTop) / nLev
s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)

###########################################################################

ss = np.tile(s, (nCol, 1)).T

###########################################################################

#Some approximation weights:

Wa = phs1.getPeriodicDM(z=x, x=x, m=1 \
, phs=11, pol=5, stc=11, period=right-left)

Whva = phs1.getPeriodicDM(z=x, x=x, m=6 \
, phs=11, pol=5, stc=11, period=right-left)
Whva = 2.**-9.*300. * dx**5. * Whva

def Ds(U):
    V = np.zeros((nLev+2, nCol))
    V[0,:] = (-3./2.*U[0,:] + 2.*U[1,:] - 1./2.*U[2,:]) / ds
    V[1:-1,:] = (U[2:,:] - U[0:-2,:]) / (2.*ds)
    V[-1,:] = (3./2.*U[-1,:] - 2.*U[-2,:] + 1./2.*U[-3,:]) / ds
    return V

if verticallyLagrangian:
    def HVs(U):
        return  2.**-18.*300. * (U[0:-2,:] - 2.*U[1:-1,:] + U[2:,:]) / ds
else:
    def HVs(U):
        return  2.**-17.*300. * (U[0:-2,:] - 2.*U[1:-1,:] + U[2:,:]) / ds

def Da(U):
    return Wa.dot(U.T).T

def HVa(U):
    return Whva.dot(U.T).T

###########################################################################

#Initial vertical levels zz:

def A(s):
    return (1. - s) / (1. - sTop) * sTop

def B(s):
    return (s - sTop) / (1. - sTop)

def Aprime(s):
    return -sTop / (1. - sTop)

def Bprime(s):
    return 1. / (1. - sTop)

ssInt = (ss[0:-1,:] + ss[1:,:]) / 2.                  #s-mesh on interfaces
p = A(ssInt) * Po + B(ssInt) * np.tile(pSurf,(nLev+1,1))
pi = (p / Po) ** (Rd/Cp)
zz0 = inverseExnerPressure(pi)
zz = zz0.copy()
xx = np.tile(x, (nLev+1, 1))

#Iterate to satisfy initial conditions and hydrostatic condition:
tmp1 = np.ones(np.shape(zz))
tmp2 = np.zeros(np.shape(zz))
for j in range(10):
    T = exnerPressure(zz, tmp1, tmp2) \
    * (potentialTemperature(zz, tmp1, tmp2) + thetaPtb(xx, zz))
    # T = exnerPressure(zz, tmp1, tmp2) * potentialTemperature(zz, tmp1, tmp2)
    integrand = -Rd * T * (Aprime(ssInt) * Po + Bprime(ssInt) * pSurf) \
    / (A(ssInt) * Po + B(ssInt) * pSurf) / g
    integrand = (integrand[0:-1,:] + integrand[1:,:]) / 2.
    tmp = zz[0,:].copy()
    for i in range(nLev):
        tmp = tmp + integrand[i,:] * ds                #midpoint quadrature
        zz[i+1,:] = tmp

top = zz[0,0]                             #slightly different top of domain
zSurf = zz[-1,:]                       #slightly different bottom of domain

#move zz from interfaces to midpoints, and get (x,z) mesh:
zz = (zz[0:-1,:] + zz[1:,:]) / 2.
zz = np.vstack(( 2.*top - zz[0,:] \
, zz \
, 2.*zSurf - zz[-1,:]))
xx = np.tile(x, (nLev+2, 1))

###########################################################################

zSurfPrime = Da(zSurf)              #consistent derivative of topo function

###########################################################################

if plotNodesAndExit:
    plt.figure()
    plt.plot(xx.flatten(), zz.flatten(), marker=".", linestyle="none")
    plt.plot(x, zSurf, color="red", linestyle="-")
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
    
thetaBar = potentialTemperature(zz, e, null)
piBar = exnerPressure(zz, e, null)
piPtb = null.copy()
Tbar = piBar * thetaBar
Tptb = (piBar + piPtb) * (thetaBar + thetaPtb(xx,zz)) - Tbar
Pbar = Po * piBar ** (Cp/Rd)
Pptb = Po * (piBar + piPtb) ** (Cp/Rd) - Pbar
rhoBar = Pbar / Rd / Tbar
rhoPtb = (Pbar + Pptb) / Rd / (Tbar + Tptb) - rhoBar
phiBar = g * (zz[0:-1,:] + zz[1:,:]) / 2.
# dpidsBar = -rhoBar * Ds(phiBar)
# dpidsPtb = -rhoPtb * Ds(phiBar)
dpidsBar = Aprime(ss) * Po + Bprime(ss) * np.tile(pSurf, (nLev+2,1))
# dpidsBar = Ds(Pbar)
# dpidsPtb = -(rhoBar+rhoPtb) * Ds(phiBar) - dpidsBar

###########################################################################

#Assignment of initial conditions:

U = np.zeros((6, nLev+2, nCol))

if testCase == "inertiaGravityWaves":
    U[0,:,:] =  20. * np.ones((nLev+2, nCol))          #horizontal velocity

U[1,:,:] = np.zeros((nLev+2, nCol))                      #vertical velocity

U[2,:,:] = thetaBar + thetaPtb(xx,zz)                #potential temperature

U[3,:,:] = dpidsBar                                         #pseudo-density

U[4,:-1,:] = phiBar                             #geopotential on interfaces

U[5,:,:] = np.zeros((nLev+2, nCol))                               #pressure

mass0 = 1./g * np.sum(U[3,1:-1,:] * dx)

###########################################################################

#Normal vectors on bottom and top boundaries.  These will be used in the
#implementation of the Neumann boundary condition for pressure.

TzBot = zSurfPrime
TxBot = np.ones((nCol))
tmp = np.sqrt(TxBot**2 + TzBot**2)
TxBot = TxBot / tmp                  #x-component of unit tangent on bottom
TzBot = TzBot / tmp                  #z-component of unit tangent on bottom

NxBot = -TzBot.copy()                 #x-component of unit normal on bottom
NzBot = TxBot.copy()                 #z-component of unit tangent on bottom

NxTop = np.zeros((nCol))                 #x-component of unit normal on top
NzTop = np.ones((nCol))                  #z-component of unit normal on top

###########################################################################

#Initialize a figure of the appropriate size:

if saveContours:
    if testCase == "inertiaGravityWaves":
        fig = plt.figure(figsize = (18,3))
    else:
        fig = plt.figure(figsize = (18,14))

###########################################################################

#Create and save a contour plot of the field specified by whatToPlot:

def contourSomething(U, t, thetaBar, rhoBar, Pbar):

    if plotBackgroundState:
        if whatToPlot == "u":
            tmp = np.zeros((nLev+2, nCol))
        elif whatToPlot == "w":
            tmp = np.zeros((nLev+2, nCol))
        elif whatToPlot == "theta":
            tmp = thetaBar
        elif whatToPlot == "rho":
            tmp = rhoBar
        elif whatToPlot == "phi":
            tmp = phiBar
        elif whatToPlot == "P":
            tmp = Pbar
        else:
            raise ValueError("Invalid whatToPlot string.")
    else:
        if whatToPlot == "u":
            tmp = U[0,:,:]
        elif whatToPlot == "w":
            tmp = U[1,:,:]
        elif whatToPlot == "theta":
            tmp = U[2,:,:] - thetaBar
        elif whatToPlot == "rho":
            tmp = -U[3,:,:] / ((U[4,1:-1,:]-U[4,0:-2,:])/ds) - rhoBar
        elif whatToPlot == "phi":
            tmp = U[4,0:-1,:] - phiBar
        elif whatToPlot == "P":
            tmp = U[5,:,:] - Pbar
        else:
            raise ValueError("Invalid whatToPlot string.")

    if whatToPlot == "phi":
        xxTmp = xx[0:-1,:]
        zzTmp = U[4,0:-1,:] / g                          #changing z-levels
    else:
        xxTmp = xx
        zzTmp = (U[4,0:-2,:]+U[4,1:-1,:])/2. / g
    
    plt.clf()
    plt.contourf(xxTmp, zzTmp, tmp, contours)
    plt.plot(x, zSurf, linestyle="-", color="red")
    plt.plot(x, top*np.ones(np.shape(x)), linestyle="-", color="red")
    if testCase == "inertiaGravityWaves":
        plt.colorbar(orientation="horizontal")
    elif testCase == "densityCurrent":
        plt.axis("image")
        plt.colorbar(orientation="horizontal")
    else:
        plt.axis("image")
        plt.colorbar(orientation="vertical")
    if plotBackgroundState:
        plt.show()
        sys.exit("\nDone plotting the requested background state.")
    else:
        fig.savefig( "{0:04d}.png".format(np.int(np.round(t)+1e-12)) \
        , bbox_inches="tight" )                       #save figure as a png

###########################################################################

#Initialize the output array for the vertical re-map function:

if verticallyLagrangian:
    V = np.zeros((5, nLev+2, nCol))

###########################################################################

#This will be used inside the setGhostNodes() function to quickly find
#background states on possibly changing vertical levels:

def fastBackgroundStates(zz, dpids):
    
    thetaBar = potentialTemperature(zz, e, null)
    piBar = exnerPressure(zz, e, null)
    Pbar = Po * piBar ** (Cp/Rd)
    Tbar = piBar * thetaBar
    rhoBar = Pbar / Rd / Tbar
    
    pHydroSurf = pTop + np.sum(dpids*ds, 0)
    dpidsBar = Aprime(ss) * Po \
    + Bprime(ss) * np.tile(pHydroSurf, (nLev+2, 1))

    return thetaBar, rhoBar, Pbar, dpidsBar

###########################################################################

def setGhostNodes(U):

    #Extrapolate theta to bottom ghost nodes:
    U[2,-1,:] = 2. * U[2,-2,:] - U[2,-3,:]

    #Extrapolate theta to top ghost nodes:
    U[2,0,:] = 2. * U[2,1,:] - U[2,2,:]
    
    #Extrapolate dpids to bottom ghost nodes:
    U[3,-1,:] = 2. * U[3,-2,:] - U[3,-3,:]

    #Extrapolate dpids to top ghost nodes:
    U[3,0,:] = 2. * U[3,1,:] - U[3,2,:]

    phi = null
    phi[1:-1,:] = (U[4,0:-2,:] + U[4,1:-1,:]) / 2.
    #Enforce phi=g*z on bottom boundary (s=1):
    phi[-1,:] = 2.*g*zSurf - phi[-2,:]
    #Enforce phi=g*z on top boundary (s=sTop):
    phi[0,:] = 2.*g*top - phi[1,:]

    #Get background states on possibly changing z levels:
    thetaBar, rhoBar, Pbar, dpidsBar = fastBackgroundStates(phi \
    , U[3,1:-1,:])
    
    # #get pressure on all mid-levels using equation of state:
    # U[5,:,:] = (-U[3,:,:] / Ds(U[4,:,:]) * Rd \
    # * U[2,:,:] / Po**(Rd/Cp)) ** (Cp/Cv)

    #get pressure on interior mid-levels using equation of state:
    U[5,1:-1,:] = (-U[3,1:-1,:] / ((U[4,1:-1,:]-U[4,0:-2,:])/ds) * Rd \
    * U[2,1:-1,:] / Po**(Rd/Cp)) ** (Cp/Cv)

    dphida = Da(U[4,-2,:])
    #set pressure on bottom ghost nodes using Neumann BC:
    dPda = Da(3./2.*U[5,-2,:] - 1./2.*U[5,-3,:])
    dpids = 3./2.*U[3,-2,:] - 1./2.*U[3,-3,:]
    dphids = (U[4,-2,:] - U[4,-3,:]) / ds
    RHS = g * dpids * NzBot - dPda * dphids * NxBot
    RHS = RHS / (g*NzBot - dphida*NxBot)
    U[5,-1,:] = U[5,-2,:] + ds * RHS
    #extrapolate u to bottom ghost nodes:
    U[0,-1,:] = 2.*U[0,-2,:] - U[0,-3,:]
    #get w on bottom ghost nodes:
    U[1,-1,:] = 2.*(U[0,-1,:] + U[0,-2,:])/2. / g * dphida - U[1,-2,:]

    dphida = Da(U[4,0,:])
    #set pressure on top ghost nodes using Neumann BC:
    dPda = Da(3./2.*U[5,1,:] - 1./2.*U[5,2,:])
    dpids = 3./2.*U[3,1,:] - 1./2.*U[3,2,:]
    dphids = (U[4,1,:] - U[4,0,:]) / ds
    RHS = g * dpids * NzTop - dPda * dphids * NxTop
    RHS = RHS / (g*NzTop - dphida*NxTop)
    U[5,0,:] = U[5,1,:] - ds * RHS
    #extrapolate u to top ghost nodes:
    U[0,0,:] = 2.*U[0,1,:] - U[0,2,:]
    #get w on top ghost nodes:
    U[1,0,:] = 2.*(U[0,0,:] + U[0,1,:])/2. / g * dphida - U[1,1,:]

    return U, thetaBar, rhoBar, Pbar, dpidsBar

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
    
    #Get some ghost node values and some other things:
    U, thetaBar, rhoBar, Pbar, dpidsBar = setGhostNodes(U)

    dpidsInt = (U[3,0:-1,:] + U[3,1:,:]) / 2.          #dpids on interfaces

    #Get sDot:

    if verticallyLagrangian:
        sDot = sDotNull
    else:
        #please see this overleaf document for a complete derivation,
        #starting from the governing equation for pseudo-density dpids:
        #https://www.overleaf.com/read/gcfkprynxvkw
        integrand = Da((U[3,:,:]) * U[0,:,:])[1:-1,:]
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
    + 1./(U[3,1:-1,:]) \
    * (((U[4,1:-1,:]-U[4,0:-2,:])/ds) * Da(U[5,:,:])[1:-1,:] \
    - Ds(U[5,:,:])[1:-1,:] * Da((U[4,0:-2,:]+U[4,1:-1,:])/2.)) \
    + HVa(U[0,1:-1,:]) \
    + HVs(U[0,:,:])                                                  #du/dt
    
    tmp = (U[1,1:,:] - U[1,0:-1,:]) / ds               #dw/ds on interfaces
    tmp = sDot * tmp                              #sDot*dw/ds on interfaces
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.          #sDot*dw/ds on mid-levels
    dUdt[1,1:-1,:] = -U[0,1:-1,:] * Da(U[1,1:-1,:]) - tmp \
    - g * (1. - Ds(U[5,:,:])[1:-1,:] / U[3,1:-1,:]) \
    + HVa(U[1,1:-1,:]) \
    + HVs(U[1,:,:])                                                  #dw/dt

    tmp = (U[2,1:,:] - U[2,0:-1,:]) / ds              #dth/ds on interfaces
    tmp = sDot * tmp                             #sDot*dth/ds on interfaces
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.         #sDot*dth/ds on mid-levels
    dUdt[2,1:-1,:] = -U[0,1:-1,:] * Da(U[2,1:-1,:]) - tmp \
    + HVa(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    + HVs(U[2,:,:] - thetaBar)                                      #dth/dt

    tmp = sDot * dpidsInt                         #sDot*dpids on interfaces
    tmp = (tmp[1:,:] - tmp[0:-1,:]) / ds    #d(sDot*dpids)/ds on mid-levels
    dUdt[3,1:-1,:] = -Da(U[3,:,:] * U[0,:,:])[1:-1,:] - tmp \
    + HVa(U[3,1:-1,:] - dpidsBar[1:-1,:]) \
    + HVs(U[3,:,:] - dpidsBar)                                 #d(dpids)/dt

    tmp = (U[4,1:,:] - U[4,0:-1,:]) / ds             #dphi/ds on mid-levels
    tmp = (tmp[0:-1,:] + tmp[1:,:]) / 2.             #dphi/ds on interfaces
    tmp = sDot[1:-1,:] * tmp                    #sDot*dphi/ds on interfaces
    dUdt[4,1:-2,:] = -(U[0,1:-2,:]+U[0,2:-1,:])/2. * Da(U[4,1:-1,:]) - tmp \
    + g * (U[1,1:-2,:]+U[1,2:-1,:])/2.
    # + HVa(U[4,1:-1,:] - phiBar[1:-1,:]) \
    # + HVs(U[4,:,:] - phiBar)                                       #dphi/dt

    return dUdt

###########################################################################

def printMinAndMax(t, et, U, thetaBar, rhoBar, Pbar):
    
    print("t = {0:5d},  et = {1:6.2f},  MIN:  u = {2:+.2e},  \
w = {3:+.2e},  th = {4:+.2e},  rho = {5:+.2e},  \
phi = {6:+.2e},  P = {7:+.2e}" \
    . format(np.int(np.round(t)) \
    , et \
    , np.min(U[0,1:-1,:]) \
    , np.min(U[1,1:-1,:]) \
    , np.min(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.min(-U[3,1:-1,:] / ((U[4,1:-1,:]-U[4,0:-2,:])/ds) - rhoBar[1:-1,:]) \
    , np.min(U[4,0:-1,:] - phiBar) \
    , np.min(U[5,1:-1,:] - Pbar[1:-1,:])))

    print("                          MAX:  u = {0:+.2e},  \
w = {1:+.2e},  th = {2:+.2e},  rho = {3:+.2e},  \
phi = {4:+.2e},  P = {5:+.2e}\n" \
    . format(np.max(U[0,1:-1,:]) \
    , np.max(U[1,1:-1,:]) \
    , np.max(U[2,1:-1,:] - thetaBar[1:-1,:]) \
    , np.max(-U[3,1:-1,:] / ((U[4,1:-1,:]-U[4,0:-2,:])/ds) - rhoBar[1:-1,:]) \
    , np.max(U[4,0:-1,:] - phiBar) \
    , np.max(U[5,1:-1,:] - Pbar[1:-1,:])))

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,1) == 0) and (testCase != "inertiaGravityWaves"):
        U, thetaBar, rhoBar, Pbar, dpidsBar = setGhostNodes(U)
        tmp = pTop * np.ones((nCol))
        integrand = U[3,1:-1,:].copy()
        pHydro = np.zeros((nLev+1, nCol))
        pHydro[0,:] = tmp.copy()
        for j in range(nLev):
            tmp = tmp + integrand[j,:] * ds
            pHydro[j+1,:] = tmp.copy()
        pHydroSurf = tmp.copy()
        pHydro = (pHydro[0:-1,:] + pHydro[1:,:])/2.
        pHydro = np.vstack((2.*pTop - pHydro[0,:] \
        , pHydro \
        , 2.*pHydroSurf - pHydro[-1,:]))

        pHydroNew = A(ss) * Po \
        + B(ss) * np.tile(pHydroSurf, (nLev+2, 1))

        U[0:5,:,:] = common.verticalRemap(U[0:5,:,:] \
        , pHydro, pHydroNew, V)

        U[3,:,:] = dpidsBar.copy()
        # U[3,:,:] = Aprime(ss) * Po \
        # + Bprime(ss) * np.tile(pHydroSurf, (nLev+2, 1))        #reset dpids
    
    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        if contourFromSaved :
            U[0:5,:,:] = np.load(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy')
        
        U, thetaBar, rhoBar, Pbar, dpidsBar = setGhostNodes(U)

        printMinAndMax(t, time.time()-et, U, thetaBar, rhoBar, Pbar)

        # print("|massDiff| = {0:.2e}" \
        # . format(np.abs(np.sum(U[3,1:-1,:] \
        # / g * ds * dx) - mass0) / mass0))
        
        et = time.time()
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours or plotBackgroundState:
            contourSomething(U, t, thetaBar, rhoBar, Pbar)
    
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
