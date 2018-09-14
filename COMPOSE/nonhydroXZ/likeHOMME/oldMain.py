import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

sys.path.append('../../../site-packages')
from gab import rk
from gab.nonhydro import common

###########################################################################

#This block contains the only variables that the user should be required
#to modify when running the code, unless they want to add a new test case.

#Choose "risingBubble", "densityCurrent", "inertiaGravityWaves",
#"steadyState", or "scharMountainWaves":
testCase = sys.argv[1]

#Choose "pressure" or "height":
verticalCoordinate = sys.argv[2]

if sys.argv[3] == "vLag":
    verticallyLagrangian = True
elif sys.argv[3] == "vEul":
    verticallyLagrangian = False
else:
    raise ValueError("The third argument should be either 'vLag' " \
    + "or 'vEul'.")

#Choose 0, 1, 2, 3, or 4:
refinementLevel = np.int64(sys.argv[4])

#Switches to control what happens:
saveArrays          = True
saveContours        = True
contourFromSaved    = True
plotNodesAndExit    = False
plotBackgroundState = False

#Choose which variable to plot
#("u", "w", "T", "rho", "phi", "P", "theta", "pi"):
try:
    whatToPlot = sys.argv[5]
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
    else:
        raise ValueError("Invalid testCase string.")
else:
    try:
        contours = eval(sys.argv[6])
    except:
        contours = 20

###########################################################################

if contourFromSaved:
    saveArrays = False
    saveContours = True

###########################################################################

#Get string for saving results:

saveString = "./oldResults/" + testCase + "_" + verticalCoordinate + "_"

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

###########################################################################

#Definitions of atmospheric constants:
Cp, Cv, Rd, g, Po, th0, N = common.constants(testCase)

#Test-specific parameters describing domain and initial perturbation:
left, right, bottom, top, dx, nLev, dt, tf, saveDel \
, zSurfFunc, thetaPtb \
= common.domainParameters(testCase, refinementLevel, g, Cp, th0)



# tf = 30.
# saveDel = 3



#Some other important parameters:
nCol = np.int(np.round((right - left) / dx))             #number of columns
t = 0.                                                        #initial time
nTimesteps = np.int(np.round(tf / dt))                #number of time-steps
x = np.linspace(left+dx/2, right-dx/2, nCol)        #array of x-coordinates
zSurf = zSurfFunc(x)             #array of topo values along bottom surface

#ones and zeros to avoid repeated initialization getting background states:
e = np.ones((nLev+2, nCol))
null = np.zeros((nLev+2, nCol))

#Hydrostatic background state functions:
potentialTemperature, potentialTemperatureDerivative \
, exnerPressure, inverseExnerPressure \
= common.hydrostaticProfiles(testCase, th0, g, Cp, N, e, null)

###########################################################################

#Equally spaced array of vertical coordinate values (s):

if verticalCoordinate == "height":
    #This is a strange height coordinate, because it starts at zero
    #at the top and goes up to 1 at the bottom.  This is so it mimics
    #the behavior of the pressure coordinate, so that we can use the
    #same setGhostNodes() function for both coordinates.
    ds = 1. / nLev
    sTop = 0.
    s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)
elif verticalCoordinate == "pressure":
    piTop  = exnerPressure(top, 1., 0.)
    pTop  = Po * piTop  ** (Cp/Rd)             #hydrostatic pressure at top
    piSurf = exnerPressure(zSurf, e[0,:], null[0,:])
    pSurf = Po * piSurf ** (Cp/Rd)         #hydrostatic pressure at surface
    sTop = pTop / Po                          #value of s on upper boundary
    ds = (1. - sTop) / nLev
    s = np.linspace(sTop-ds/2, 1+ds/2, nLev+2)
else:
    raise ValueError("Invalid verticalCoordinate string.  Please " \
    + "choose either 'height' or 'pressure'.")

###########################################################################

xx, ss = np.meshgrid(x, s)

###########################################################################

#All of the weights and functions associated with derivative approximation:

Wa, stc, wItop, wEtop, wDtop, wHtop, wIbot, wEbot, wDbot, wHbot \
, Da, Ds, HV, rayleighDamping, numDampedLayers \
= common.derivativeApproximations(x, dx, left, right, s, ds)

###########################################################################

#Initial vertical levels zz (easy in height coord, hard in pressure coord):

if verticalCoordinate == "height":

    zz = np.zeros((nLev+2, nCol))

    for j in range(nCol):
        dz = (top - zSurf[j]) * ds
        zz[:,j] = np.flipud(np.linspace(zSurf[j]-dz/2, top+dz/2, nLev+2))

elif verticalCoordinate == "pressure":
    
    def A(s):
        return (1. - s) / (1. - sTop) * sTop
    
    def Aprime(s):
        return -sTop / (1. - sTop)

    def B(s):
        return (s - sTop) / (1. - sTop)
    
    def Bprime(s):
        return 1. / (1. - sTop)
    
    ssInt = (ss[0:-1,:] + ss[1:,:]) / 2.              #s-mesh on interfaces
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
            zz[i+1,:] = tmp.copy()

    top = zz[0,0]                         #slightly different top of domain
    zSurf = zz[-1,:]                   #slightly different bottom of domain

    #move zz from interfaces to midpoints:
    tmp = np.zeros((nLev+2, nCol))
    tmp[1:-1,:] = (zz[0:-1,:] + zz[1:,:]) / 2.
    tmp[-1,:] = (zSurf - wIbot[1:stc].dot(tmp[-2:-1-stc:-1,:])) / wIbot[0]
    tmp[0,:] = (top - wItop[1:stc].dot(tmp[1:stc,:])) / wItop[0]
    zz = tmp.copy()

###########################################################################

zSurfPrime = Wa.dot(zSurf.T).T      #consistent derivative of topo function

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
piPtb = null
Tbar = piBar * thetaBar
Tptb = (piBar + piPtb) * (thetaBar + thetaPtb(xx,zz)) - Tbar
Pbar = Po * piBar ** (Cp/Rd)
Pptb = Po * (piBar + piPtb) ** (Cp/Rd) - Pbar
rhoBar = Pbar / Rd / Tbar
rhoPtb = (Pbar + Pptb) / Rd / (Tbar + Tptb) - rhoBar
phiBar = g * zz
    
###########################################################################

#Assignment of initial conditions:

U = np.zeros((6, nLev+2, nCol))                       #3D array of unknowns

if testCase == "inertiaGravityWaves":
    U[0,:,:] =  20. * np.ones((nLev+2, nCol))
elif testCase == "scharMountainWaves":
    U[0,:,:] = 10. * np.ones((nLev+2, nCol))
else:
    U[0,:,:] = np.zeros((nLev+2, nCol))                #horizontal velocity
U[1,:,:] = np.zeros((nLev+2, nCol))                      #vertical velocity
U[2,:,:] = Tptb                                   #temperature perturbation
U[3,:,:] = rhoPtb                                                  #density
U[4,:,:] = phiBar.copy()                                      #geopotential
U[5,:,:] = np.zeros((nLev+2, nCol))                  #pressure perturbation

mass0 = -np.sum(((rhoBar+U[3,:,:]) * Ds(U[4,:,:]))[1:-1,:] / g * ds * dx)

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

def contourSomething(U, t):

    if plotBackgroundState:
        if whatToPlot == "u":
            tmp = np.zeros((nLev+2, nCol))
        elif whatToPlot == "w":
            tmp = np.zeros((nLev+2, nCol))
        elif whatToPlot == "T":
            tmp = Tbar
        elif whatToPlot == "rho":
            tmp = rhoBar
        elif whatToPlot == "phi":
            tmp = phiBar
        elif whatToPlot == "P":
            tmp = Pbar
        elif whatToPlot == "theta":
            tmp = thetaBar
        elif whatToPlot == "pi":
            tmp = piBar
        else:
            raise ValueError("Invalid whatToPlot string.")
    else:
        if whatToPlot == "u":
            tmp = U[0,:,:]
        elif whatToPlot == "w":
            tmp = U[1,:,:]
        elif whatToPlot == "T":
            tmp = U[2,:,:]
        elif whatToPlot == "rho":
            tmp = U[3,:,:]
        elif whatToPlot == "phi":
            tmp = U[4,:,:] - phiBar
        elif whatToPlot == "P":
            tmp = U[5,:,:]
        elif whatToPlot == "theta":
            tmp = (U[2,:,:]+Tbar) / ((U[5,:,:]+Pbar) / Po) ** (Rd/Cp) \
            - thetaBar
        elif whatToPlot == "pi":
            tmp = ((U[5,:,:]+Pbar) / Po) ** (Rd/Cp) - piBar
        else:
            raise ValueError("Invalid whatToPlot string.")

    zz = U[4,:,:] / g                           #possibly changing z-levels
    
    if testCase != "inertiaGravityWaves":
        matplotlib.rcParams.update({'font.size': 22})#not igw
    else:
        matplotlib.rcParams.update({'font.size': 14})#igw

    plt.clf()
    plt.contourf(xx[1:-1,:], zz[1:-1,:], tmp[1:-1,:], contours)
    lw = .75
    plt.plot([np.min(x),np.max(x)], [top,top], linestyle="-", color="red" \
    , linewidth=lw)
    plt.plot(x,zSurf, linestyle="-", color="red", linewidth=lw)
    plt.plot([left,left], [zSurfFunc(left),top], linestyle="-" \
    , color="red", linewidth=lw)
    plt.plot([right,right], [zSurfFunc(right),top], linestyle="-" \
    , color="red", linewidth=lw)
    # plt.plot(xx[1:-1,:].flatten(), zz[1:-1,:].flatten(), marker="." \
    # , linestyle="none", color="black", markersize=1)
    if testCase == "inertiaGravityWaves":
        plt.colorbar(orientation="horizontal")
    elif testCase == "densityCurrent":
        plt.axis("image")
        plt.axis([0.,19200.,0.,4800.])
        plt.colorbar(orientation="horizontal")
    elif testCase == "scharMountainWaves":
        plt.axis("image")
        plt.axis([-25000,25000,-250,20000])
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

#Initialize the output array for the vertical re-map function.  For some
#reason, re-setting the mapping variable (phi in height coord, rho in
#pressure coord) works fine for the height coordinate, but causes problems
#in the pressure coordinate.  Not sure why.

if verticallyLagrangian:
    if verticalCoordinate == "pressure":
        V = np.zeros((5, nLev+2, nCol))
    else:
        V = np.zeros((4, nLev+2, nCol))

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
    U[4,-1,:] = (g*zSurf - wIbot[1:stc].dot(U[4,-2:-1-stc:-1,:])) \
    / wIbot[0]
    
    #Enforce phi=g*z on top boundary (s=sTop):
    U[4,0,:] = (g*top - wItop[1:stc].dot(U[4,1:stc,:])) \
    / wItop[0]
    
    #Get background states on possibly changing vertical levels:
    Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz \
    = fastBackgroundStates(U[4,:,:] / g)
    
    #extrapolate tangent velocity uT to bottom ghost nodes:
    uT = U[0,-2:-stc-2:-1,:] * TxBot + U[1,-2:-stc-2:-1,:] * TzBot
    uT = wEbot.dot(uT)
    
    #get normal velocity uN on bottom ghost nodes:
    uN = U[0,-2:-1-stc:-1,:] * NxBot + U[1,-2:-1-stc:-1,:] * NzBot
    uN = -wIbot[1:stc].dot(uN) / wIbot[0]
    
    #use uT and uN to get (u,w) on bottom ghost nodes:
    U[0,-1,:] = uT*TxBot[0,:] + uN*NxBot[0,:]
    U[1,-1,:] = uT*TzBot[0,:] + uN*NzBot[0,:]
    
    #get (u,w) on top ghost nodes (easier because it's flat):
    U[0,0,:] = wEtop.dot(U[0,1:stc+1,:])
    U[1,0,:] = -wItop[1:stc].dot(U[1,1:stc,:]) / wItop[0]
    
    #get pressure on interior nodes using the equation of state:
    U[5,1:-1,:] = ((rhoBar+U[3,:,:]) * Rd * (Tbar+U[2,:,:]) - Pbar)[1:-1,:]
    
    #set pressure on bottom ghost nodes:
    dPda = Da(wHbot.dot(U[5,-2:-2-stc:-1,:]))
    rho = wHbot.dot(U[3,-2:-2-stc:-1,:])
    dphida = Da(wIbot.dot(U[4,-1:-1-stc:-1,:]))
    dphids = wDbot.dot(U[4,-1:-1-stc:-1,:])
    dsdx = -dphida / dphids
    dsdz = g / dphids
    RHS = -rho * g * NzBot[0,:] - dPda * NxBot[0,:]
    RHS = RHS / (NxBot[0,:] * dsdx + NzBot[0,:] * dsdz)
    U[5,-1,:] = (RHS - wDbot[1:stc].dot(U[5,-2:-1-stc:-1,:])) / wDbot[0]
    
    #set pressure on top ghost nodes:
    dPda = Da(wHtop.dot(U[5,1:stc+1,:]))
    rho = wHtop.dot(U[3,1:stc+1,:])
    dphida = Da(wItop.dot(U[4,0:stc,:]))
    dphids = wDtop.dot(U[4,0:stc,:])
    dsdx = -dphida / dphids
    dsdz = g / dphids
    RHS = -rho * g * NzTop[0,:] - dPda * NxTop[0,:]
    RHS = RHS / (NxTop[0,:] * dsdx + NzTop[0,:] * dsdz)
    U[5,0,:] = (RHS - wDtop[1:stc].dot(U[5,1:stc,:])) / wDtop[0]
    
    #extrapolate temperature to bottom and top ghost nodes:
    U[2,-1,:] = wEbot.dot(U[2,-2:-(stc+2):-1,:])
    U[2,0,:] = wEtop.dot(U[2,1:stc+1,:])
    
    #extrapolate density to bottom and top ghost nodes using EOS:
    U[3,-1,:] = (Pbar[-1,:]+U[5,-1,:]) / Rd / (Tbar[-1,:]+U[2,-1,:]) \
    - rhoBar[-1,:]
    U[3,0,:] = (Pbar[0,:]+U[5,0,:]) / Rd / (Tbar[0,:]+U[2,0,:]) \
    - rhoBar[0,:]
    
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
    
    rhoInv = 1. / (rhoBar + U[3,:,:])
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
    
    #Get sDot:

    if verticallyLagrangian:
        sDot = sDotNull
    else:
        if verticalCoordinate == "height":
            sDot = uDotGradS
        else:
            #please see this overleaf document for a complete derivation,
            #starting from the governing equation for pseudo-density dpids:
            #https://www.overleaf.com/read/gcfkprynxvkw
            sDot = sDotNull
            dpids = -(rhoBar+U[3,:,:]) * Ds(U[4,:,:])#hydrostatic condition
            integrand = Da(dpids * U[0,:,:])[1:-1,:]
            dpids = (dpids[0:-1,:] + dpids[1:,:]) / 2.   #avg to interfaces
            sDot[0:-1,:] = B(ssInt) \
            * np.tile(np.sum(integrand*ds,0), (nLev+1,1))
            tmp = np.zeros((nCol))
            for j in range(nLev):
                tmp = tmp + integrand[j,:] * ds
                sDot[j+1,:] = sDot[j+1,:] - tmp
            sDot[0:-1,:] = sDot[0:-1,:] / dpids
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
    - rhoInv * (dPds * dsdz + U[3,:,:] * g))[1:-1,:] \
    + HV(U[1,:,:])                                                   #dw/dt

    dUdt[2,1:-1,:] = (-U[0,:,:] * Da(U[2,:,:]) - sDot * Ds(U[2,:,:]) \
    - U[1,:,:] * dTbarDz - Rd/Cv * (Tbar + U[2,:,:]) * divU)[1:-1,:] \
    + HV(U[2,:,:])                                                   #dT/dt

    dUdt[3,1:-1,:] = (-U[0,:,:] * Da(U[3,:,:]) - sDot * Ds(U[3,:,:]) \
    - U[1,:,:] * drhoBarDz - (rhoBar + U[3,:,:]) * divU)[1:-1,:] \
    + HV(U[3,:,:])                                                 #drho/dt

    # dUdt[4,1:-1,:] = (-U[0,:,:] * Da(U[4,:,:]) - sDot * Ds(U[4,:,:]) \
    # + g * U[1,:,:])[1:-1,:] \
    # + HV(U[4,:,:] - phiBar)                                        #dphi/dt
    dUdt[4,1:-1,:] = ((uDotGradS - sDot) * dphids)[1:-1,:] \
    + HV(U[4,:,:] - phiBar)                                        #dphi/dt

    #Rayleigh damping in scharMountainWaves test case:

    # if testCase == "scharMountainWaves":
    #     for k in range(5):
    #         dUdt[k,1:numDampedLayers,:] = dUdt[k,1:numDampedLayers,:] \
    #         + rayleighDamping(U[k,:,:])
    
    return dUdt

###########################################################################

#Time-stepping loop:

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,4) == 0) and (testCase != "inertiaGravityWaves"):
        if verticalCoordinate == "height":
            U = setGhostNodes(U)[0]
            U[0:4,:,:] = common.verticalRemap(U[0:4,:,:] \
            , U[4,:,:], phiBar, V)
            U[4,:,:] = phiBar
        else:
            tmp = setGhostNodes(U)
            U = tmp[0]
            rhoBar = tmp[2]
            integrand = (-(rhoBar+U[3,:,:]) * Ds(U[4,:,:]))[1:-1,:]
            tmp = pTop * np.ones((nCol))
            pHydro = np.zeros((nLev+1, nCol))
            pHydro[0,:] = tmp
            for j in range(nLev):
                tmp = tmp + integrand[j,:] * ds
                pHydro[j+1,:] = tmp

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

            U[0:5,:,:] = common.verticalRemap(U[0:5,:,:] \
            , pHydro, pHydroNew, V)

            # rhoBar = setGhostNodes(U)[2]
            # rho = Aprime(ss) * Po \
            # + Bprime(ss) * np.tile(pHydroSurf, (nLev+2, 1))
            # rho = -rho / Ds(U[4,:,:])
            # U[3,:,:] = rho - rhoBar

    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        if contourFromSaved:
            U[0:5,:,:] = np.load(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy')
        
        tmp = setGhostNodes(U)
        U = tmp[0]
        rhoBar = tmp[2]
        
        common.printMinAndMax(t, time.time()-et, U, rhoBar, phiBar)

        # print("t = {0:5d},  et = {1:6.2f},  relativeMassChange = {2:.2e}" \
        # . format(np.int(np.round(t)) \
        # , time.time() - et \
        # , (-np.sum(((rhoBar+U[3,:,:]) * Ds(U[4,:,:]))[1:-1,:] \
        # / g * ds * dx) - mass0) / mass0))
        
        et = time.time()
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours or plotBackgroundState:
            # tmp = U[3,:,:].copy()
            # dpids = -((rhoBar+U[3,:,:]) * Ds(U[4,:,:]))[1:-1,:]
            # pHydroSurf = pTop * np.ones((nCol))
            # for j in range(nLev):
            #     pHydroSurf = pHydroSurf + dpids[j,:] * ds
            # dpidsBar = Aprime(ss) * Po \
            # + Bprime(ss) * np.tile(pHydroSurf, (nLev+2, 1))
            # U[3,:,:] = -(rhoBar+U[3,:,:])*Ds(U[4,:,:]) - dpidsBar
            contourSomething(U, t)
            # U[3,:,:] = tmp
    
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
