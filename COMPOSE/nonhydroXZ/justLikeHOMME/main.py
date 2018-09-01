import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append('../../../site-packages')
from gab import rk
from gab.nonhydro import common

###########################################################################

#This block contains the only variables that the user should be required
#to modify when running the code, unless they want to add a new test case.

#Choose "risingBubble", "densityCurrent", "inertiaGravityWaves", or
#"steadyState":
testCase = "inertiaGravityWaves"

#Choose True or False:
verticallyLagrangian = False

#Choose 0, 1, 2, 3, or 4:
refinementLevel = 1

#Switches to control what happens:
saveArrays          = False
saveContours        = True
contourFromSaved    = False
plotNodesAndExit    = False
plotBackgroundState = False

#Choose which variable to plot
#("u", "w", "theta", "dpids", "phi", "P", "T", "pi"):
whatToPlot = "dpids"

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
Cp, Cv, Rd, g, Po, th0, N = common.constants()

#Test-specific parameters describing domain and initial perturbation:
left, right, bottom, top, dx, nLev, dt, tf, saveDel \
, zSurf, thetaPtb \
= common.domainParameters(testCase, refinementLevel, g, Cp, th0)

#Some other important parameters:
nCol = np.int(np.round((right - left) / dx))             #number of columns
t = 0.                                                        #initial time
nTimesteps = np.int(np.round(tf / dt))                #number of time-steps
x = np.linspace(left+dx/2, right-dx/2, nCol)        #array of x-coordinates
zSurf = zSurf(x)            #over-write zSurf function with array of values

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
piSurf = exnerPressure(zSurf, e[0,:], null[0,:])
pTop  = Po * piTop  ** (Cp/Rd)             #hydrostatic pressure at top
pSurf = Po * piSurf ** (Cp/Rd)         #hydrostatic pressure at surface
sTop = pTop / Po                          #value of s on upper boundary
ds = (1. - sTop) / nLev
s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)

###########################################################################

xx, ss = np.meshgrid(x, s)

###########################################################################

#Some approximation weights:

Wa, stc, wItop, wEtop, wDtop, wHtop, wIbot, wEbot, wDbot, wHbot \
, Da, Ds, HV \
= common.derivativeApproximations(x, dx, left, right, s, ds)

Wa = getPeriodicDM(z=x, x=x, m=1 \
, phs=7, pol=6, stc=7, period=right-left)

Whva = getPeriodicDM(z=x, x=x, m=6 \
, phs=7, pol=6, stc=7, period=right-left)
Whva = dx**5./120. * Whva

def Ds(U):
    return (U[2:,:] - U[0:-2,:]) / (2.*ds)

def HVs(U):
    return (U[0:-2,:] - 2.*U[1:-1,:] + U[2:,:]) / 2. / ds

def Da(U):
    return Wa.dot(U[1:-1,:].T).T

def HVa(U):
    return Whva.dot(U[1:-1,:].T).T

###########################################################################

#Initial vertical levels zz:

def A(s):
    return (1. - s) / (1. - sTop) * sTop

def Aprime(s):
    return -sTop / (1. - sTop)

def B(s):
    return (s - sTop) / (1. - sTop)

def Bprime(s):
    return 1. / (1. - sTop)

ssInt = (ss[0:-1,:] + ss[1:,:]) / 2.                  #s-mesh on interfaces
p = A(ssInt) * Po + B(ssInt) * np.tile(pSurf,(nLev+1,1))
pi = (p / Po) ** (Rd/Cp)
zz0 = inverseExnerPressure(pi)
zz = zz0.copy()

#Iterate to satisfy initial conditions and hydrostatic condition:
tmp1 = np.ones(np.shape(zz))
tmp2 = np.zeros(np.shape(zz))
for j in range(10):
    T = exnerPressure(zz, tmp1, tmp2) \
    * potentialTemperature(zz, tmp1, tmp2)
    # T = exnerPressure(zz, tmp1, tmp2) \
    # * (potentialTemperature(zz, tmp1, tmp2) + thetaPtb(xx[1:,:],zz))
    integrand = -Rd * T / (A(ssInt) * Po + B(ssInt) * pSurf) \
    * (Aprime(ssInt) * Po + Bprime(ssInt) * pSurf) / g
    integrand = (integrand[0:-1,:] + integrand[1:,:]) / 2.
    tmp = zz[0,:].copy()
    for i in range(nLev):
        tmp = tmp + integrand[i,:] * ds            #midpoint quadrature
        zz[i+1,:] = tmp

top = zz[0,:]                             #slightly different top of domain
zSurf = zz[-1,:]                       #slightly different bottom of domain

#move zz from interfaces to midpoints:
zz = (zz[0:-1,:] + zz[1:,:]) / 2.
zz = np.vstack((2*top - zz[0,:] \
, zz \
, 2*zSurf - zz[-1,:]))

###########################################################################

zSurfPrime = Da(zSurf)              #consistent derivative of topo function

###########################################################################

if plotNodesAndExit:
    plt.figure()
    plt.plot(xx.flatten(), zz.flatten(), marker=".", linestyle="none")
    plt.plot(x, zSurf, color="red", linestyle="-")
    plt.plot(x, top*np.ones(np.shape(x)), color="red", linestyle="-")
    plt.axis("image")
    plt.show()
    sys.exit("Finished plotting.")

###########################################################################

#Assignment of hydrostatic background states and initial perturbations:
    
thetaBar = potentialTemperature(zz[1:-1,:], e[1:-1,:], null[1:-1,:])
piBar = exnerPressure(zz, e, null)
piPtb = null
Tbar = piBar * thetaBar
Tptb = (piBar + piPtb) * (thetaBar + thetaPtb(xx,zz)) - Tbar
Pbar = Po * piBar ** (Cp/Rd)
Pptb = Po * (piBar + piPtb) ** (Cp/Rd) - Pbar
rhoBar = Pbar / Rd / Tbar
rhoPtb = (Pbar + Pptb) / Rd / (Tbar + Tptb) - rhoBar
phiBar = g * zz
dpidsBar = -rhoBar * Ds(phiBar)
dpidsPtb = -rhoPtb * Ds(phiBar)
    
###########################################################################

#Assignment of initial conditions:

U = np.zeros((6, nLev+2, nCol))

if testCase == "inertiaGravityWaves":
    U[0,:,:] =  20. * np.ones((nLev+2, nCol))          #horizontal velocity
U[1,:,:] = np.zeros((nLev+2, nCol))                      #vertical velocity
U[2,:,:] = thetaBar + thetaPtb(xx,zz)   #potential temperature perturbation
U[3,:,:] = dpidsBar + dpidsPtb                              #pseudo-density
U[4,:,:] = phiBar.copy()                                      #geopotential
U[5,:,:] = np.zeros((nLev+2, nCol))                               #pressure

mass0 = 1./g * np.sum(U[3,1:-1,:] * dx)

###########################################################################

#Tangent and normal vectors on bottom and top boundaries:

TxBot, TzBot, NxBot, NzBot \
, TxTop, TzTop, NxTop, NzTop \
= common.tangentsAndNormals(zSurfPrime, stc)

###########################################################################

#Initialize a figure of the appropriate size:

if saveContours:
    if testCase == "inertiaGravityWaves":
        fig = plt.figure(figsize = (18,3))
    else:
        fig = plt.figure(figsize = (18,14))

###########################################################################

#Create and save a contour plot of the field specified by whatToPlot:

def contourSomething(U, t, rhoBar):

    if plotBackgroundState:
        if whatToPlot == "u":
            tmp = np.zeros((nLev+2, nCol))
        elif whatToPlot == "w":
            tmp = np.zeros((nLev+2, nCol))
        elif whatToPlot == "theta":
            tmp = thetaBar
        elif whatToPlot == "dpids":
            tmp = dpidsBar
        elif whatToPlot == "phi":
            tmp = phiBar
        elif whatToPlot == "P":
            tmp = Pbar
        elif whatToPlot == "T":
            tmp = Tbar
        elif whatToPlot == "pi":
            tmp = piBar
        else:
            raise ValueError("Invalid whatToPlot string.")
    else:
        if whatToPlot == "u":
            tmp = U[0,:,:]
        elif whatToPlot == "w":
            tmp = U[1,:,:]
        elif whatToPlot == "theta":
            tmp = U[2,:,:] - thetaBar
        elif whatToPlot == "dpids":
            tmp = U[3,:,:] - dpidsBar
        elif whatToPlot == "phi":
            tmp = U[4,:,:] - phiBar
        elif whatToPlot == "P":
            tmp = U[5,:,:] - Pbar
        elif whatToPlot == "pi":
            tmp = (U[5,:,:] / Po) ** (Rd/Cp) - piBar
        else:
            raise ValueError("Invalid whatToPlot string.")

    zz = U[4,:,:] / g                           #possibly changing z-levels
    
    plt.clf()
    plt.contourf(xx, zz, tmp, contours)
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

def fastBackgroundStates(zz):
    
    thetaBar = potentialTemperature(zz, e, null)
    piBar = exnerPressure(zz, e, null)
    dthetaBarDz = potentialTemperatureDerivative(zz, e, null)
    
    Tbar = piBar * thetaBar
    Pbar = Po * piBar ** (Cp/Rd)
    rhoBar = Pbar / Rd / Tbar
    
    dpiBarDz = -g / Cp / thetaBar                    #hydrostatic condition
    dTbarDz = piBar * dthetaBarDz + thetaBar * dpiBarDz
    dPbarDz = Po * Cp/Rd * piBar**(Cp/Rd-1.) * dpiBarDz
    drhoBarDz = (dPbarDz - Rd*rhoBar*dTbarDz) / (Rd * Tbar)
    
    return Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz

###########################################################################

def setGhostNodes(U):

    #Enforce phi=g*z on bottom boundary (s=1):
    U[4,-1,:] = 2. * g*zSurf - U[4,-2,:]
    
    #Enforce phi=g*z on top boundary (s=sTop):
    U[4,0,:] = 2. * g*top - U[4,1,:]
    
    #Get background states on possibly changing vertical levels:
    Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz \
    = fastBackgroundStates(U[4,:,:] / g)
    
    #get pressure on interior nodes using the equation of state:
    U[5,1:-1,:] = (-U[3,1:-1,:] / ((U[4,2:,:]-U[4,0:-2,:])/2.) * Rd \
    * U[2,1:-1,:] / Po**(Rd/Cp)) ** (Cp/Cv)
    
    #set pressure on bottom ghost nodes:
    dPda = Da(3./2.*U[5,-2,:] - 1./2.*U[5,-3,:])
    dpi = 3./2.*U[3,-2,:] - 1./2.*U[3,-3,:]
    dphida = Da((U[4,-1,:] + U[4,-2,:]) / 2.)
    dphi = U[4,-1,:] - U[4,-2,:]
    RHS = g * dpi * NzBot[0,:] - dPda * dphi * NxBot[0,:]
    RHS = RHS / (g * NzBot[0,:] - dphida * NxBot[0,:])
    U[5,-1,:] = U[5,-2,:] + RHS
    
    #set pressure on top ghost nodes:
    dPda = Da(3./2.*U[5,1,:] - 1./2.*U[5,2,:])
    dpi = 3./2.*U[3,1,:] - 1./2.*U[3,2,:]
    dphida = Da((U[4,0,:] + U[4,1,:]) / 2.)
    dphi = U[4,0,:] - U[4,1,:]
    RHS = g * dpi * NzTop[0,:] - dPda * dphi * NxTop[0,:]
    RHS = RHS / (g * NzTop[0,:] - dphida * NxTop[0,:])
    U[5,0,:] = U[5,1,:] - RHS
    
    return U, Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz

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

sDotNull = np.zeros((nLev+2, nCol))

###########################################################################

#This describes the RHS of the system of ODEs in time that will be solved:

def odefun(t, U, dUdt):
    
    #Preliminaries:
    
    U, Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz = setGhostNodes(U)
    
    rhoInv = 1. / U[3,:,:]
    # rhoInv = 1. / (rhoBar + U[3,:,:])
    duda   = Da(U[0,:,:])
    duds   = Ds(U[0,:,:])
    dwda   = Da(U[1,:,:])
    dwds   = Ds(U[1,:,:])
    dphids = Ds(U[4,:,:])
    dPds   = Ds(U[5,:,:])
    
    dsdx = -Da(U[4,:,:]) / dphids
    dsdz = g / dphids
    divU = (duda + duds * dsdx) + (dwds * dsdz)
    uDotGradS = U[0,:,:] * dsdx + U[1,:,:] * dsdz
    
    if verticallyLagrangian:
        sDot = sDotNull
    else:
        #please see this overleaf document for a complete derivation,
        #starting from the governing equation for pseudo-density dpids:
        #https://www.overleaf.com/read/gcfkprynxvkw
        sDot = sDotNull
        integrand = Da(U[3,:,:] * U[0,:,:])[1:-1,:]
        sDot[0:-1,:] = B(ssInt) \
        * np.tile(np.sum(integrand,0), (nLev+1,1))
        tmp = np.zeros((nCol))
        for j in range(nLev):
            tmp = tmp + integrand[j,:]
            sDot[j+1,:] = sDot[j+1,:] - tmp
        sDot[0:-1,:] = sDot[0:-1,:] / 
        sDot[1:-1,:] = (sDot[0:-2,:] + sDot[1:-1,:]) / 2.#avg to midpts
        sDot[-1,:] = (0. - wIbot[1:stc].dot(sDot[-2:-1-stc:-1,:])) \
        / wIbot[0]                                    #sDot=0 on bottom
        sDot[0,:] = (0. - wItop[1:stc].dot(sDot[1:stc,:])) \
        / wItop[0]                                       #sDot=0 on top
    
    #Main part:
    
    dUdt[0,1:-1,:] = (-U[0,:,:] * duda - sDot * duds \
    - rhoInv * (Da(U[5,:,:]) + dPds * dsdx))[1:-1,:] \
    + HV(U[0,:,:])                                                   #du/dt
    
    dUdt[1,1:-1,:] = (-U[0,:,:] * dwda - sDot * dwds \
    - rhoInv * (dPds * dsdz + (U[3,:,:]-rhoBar) * g))[1:-1,:] \
    + HV(U[1,:,:])                                                   #dw/dt

    dUdt[2,1:-1,:] = (-U[0,:,:] * Da(U[2,:,:]) - sDot * Ds(U[2,:,:]) \
    - U[1,:,:] * dTbarDz - Rd/Cv * (Tbar + U[2,:,:]) * divU)[1:-1,:] \
    + HV(U[2,:,:])                                                   #dT/dt

    dUdt[3,1:-1,:] = (-U[0,:,:] * Da(U[3,:,:]) - sDot * Ds(U[3,:,:]) \
    - U[3,:,:] * divU)[1:-1,:] \
    + HV(U[3,:,:] - rhoBar)                                        #drho/dt
    # dUdt[3,1:-1,:] = (-U[0,:,:] * Da(U[3,:,:]) - sDot * Ds(U[3,:,:]) \
    # - U[1,:,:] * drhoBarDz - (rhoBar + U[3,:,:]) * divU)[1:-1,:] \
    # + HV(U[3,:,:])                                                 #drho/dt

    dUdt[4,1:-1,:] = ((uDotGradS - sDot) * dphids)[1:-1,:] \
    + HV(U[4,:,:] - phiBar)                                        #dphi/dt
    
    return dUdt

###########################################################################

#Main time-stepping loop:

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,4) == 0) and (testCase != "inertiaGravityWaves"):
        if verticalCoordinate == "height":
            U = setGhostNodes(U)[0]
            U[0:5,:,:] = common.verticalRemap(U[0:5,:,:] \
            , U[4,:,:], phiBar, V)
        else:
            tmp = setGhostNodes(U)
            U = tmp[0]
            rhoBar = tmp[2]
            integrand = (-U[3,:,:] * Ds(U[4,:,:]))[1:-1,:]
            # integrand = (-(rhoBar+U[3,:,:]) * Ds(U[4,:,:]))[1:-1,:]
            tmp = pTop * np.ones((nCol))
            pHydro = np.zeros((nLev+1, nCol))
            pHydro[0,:] = tmp.copy()
            for j in range(nLev):
                tmp = tmp + integrand[j,:] * ds
                pHydro[j+1,:] = tmp.copy()

            pHydroSurf = tmp.copy()

            pHydro = (pHydro[0:-1,:] + pHydro[1:,:])/2.
            tmp = np.zeros((nLev+2, nCol))
            tmp[1:-1,:] = pHydro
            tmp[-1,:] = (pHydroSurf \
            - wIbot[1:stc].dot(tmp[-2:-1-stc:-1,:])) / wIbot[0]
            tmp[0,:] = (pTop - wItop[1:stc].dot(tmp[1:stc,:])) / wItop[0]
            pHydro = tmp.copy()

            pHydroNew = A(ss) * Po \
            + B(ss) * np.tile(pHydroSurf, (nLev+2, 1))
            pHydroNew[-1,:] = (pHydroSurf \
            - wIbot[1:stc].dot(pHydroNew[-2:-1-stc:-1,:])) / wIbot[0]
            pHydroNew[0,:] = (pTop - wItop[1:stc].dot(pHydroNew[1:stc,:])) / wItop[0]

            U[0:5,:,:] = common.verticalRemap(U[0:5,:,:] \
            , pHydro, pHydroNew, V)
            U[3,:,:] = -Ds(pHydroNew) / Ds(U[4,:,:])
    
    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        tmp = setGhostNodes(U)
        U = tmp[0]
        rhoBar = tmp[2]

        common.printMinAndMax(t, time.time()-et, U, rhoBar, phiBar)

        # print("|massDiff| = {0:.2e}" \
        # . format(np.abs(np.sum(U[3,1:-1,:] * Ds(U[4,:,:])[1:-1,:] \
        # / g * ds * dx) - mass0) / mass0))
        
        et = time.time()
        
        if contourFromSaved :
            U[0:5,:,:] = np.load(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy')
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours:
            contourSomething(U, t, rhoBar)
    
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
