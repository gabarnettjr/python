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

#Choose "risingBubble", "densityCurrent", or "inertiaGravityWaves":
testCase = "densityCurrent"

#Choose "pressure" or "height":
verticalCoordinate = "height"
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
#("u", "w", "T", "rho", "phi", "P", "theta", "pi", "phi"):
whatToPlot = "P"

#Choose either a number of contours, or a range of contours:
contours = 20
# contours = np.arange(-.15, 2.25, .1)

###########################################################################

#Get string for saving results:

saveString = "./results/" + testCase + "_" + verticalCoordinate + "_"

if verticallyLagrangian:
    saveString = saveString + "vLag" + "_"
else:
    saveString = saveString + "vEul" + "_"

saveString = saveString + str(refinementLevel) + "/"

###########################################################################

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

#Definitions of constants:

Cp = 1004.                   #specific heat of dry air at constant pressure
Cv = 717.                      #specific heat dry of air at constant volume
Rd = Cp - Cv                                      #gas constant for dry air
g  = 9.81                                           #gravitational constant
Po = 10.**5.                                 #reference pressure in Pascals

th0 = 300.                       #reference potential temperature in kelvin
N   = .01           #Brunt-Vaisala frequency for inertia gravity waves case

###########################################################################

#All of the test cases are defined in terms of hydrostatic background
#states for potential temperature (th) and Exner pressure (pi), and these
#are functions of z only.  In the pressure coordinate case, the initial
#z-levels are defined using the inverse Exner pressure function.

if (testCase == "risingBubble") or (testCase == "densityCurrent"):
    
    def potentialTemperature(z):
        return th0 * np.ones(np.shape(z))
    
    def potentialTemperatureDerivative(z):
        return np.zeros(np.shape(z))
    
    def exnerPressure(z):
        return 1. - g/Cp/th0 * z
    
    def inverseExnerPressure(pi):
        return (1. - pi) * Cp*th0/g
    
elif testCase == "inertiaGravityWaves":
    
    def potentialTemperature(z):
        return th0 * np.exp(N**2/g * z)
    
    def potentialTemperatureDerivative(z):
        return th0 * N**2/g * np.exp(N**2/g * z)
    
    def exnerPressure(z):
        return 1. + g**2/Cp/th0/N**2 * (np.exp(-N**2/g*z) - 1.)
    
    def inverseExnerPressure(pi):
        return -g/N**2. * np.log(1. + (pi-1.) * Cp*th0*N**2/g**2.)
    
else:
    
    raise ValueError("Invalid test case string.  Please choose " \
    + "'risingBubble', 'densityCurrent', or 'inertiaGravityWaves'.")

###########################################################################

#Get some test-specific parameters, such as the size of the domain, the
#horizontal node spacing dx, and the number of vertical levels nLev, the
#time between saves saveDel, the topogrophy function zSurf(x), and the
#initial perturbation in potential temperature thetaPtb(x,z):

if testCase == "risingBubble":
    left    = -5000.
    right   = 5000.
    bottom  = 0.
    top     = 10000.
    dx      = 400. / 2**refinementLevel
    nLev    = 25 * 2**refinementLevel
    dt      = 1./2. / 2**refinementLevel
    tf      = 1000.
    saveDel = 100
    def zSurf(x):
        return 1000. * np.sin(2.*np.pi / 10000. * x)
    def thetaPtb(x, z):
        return 2. * np.exp(-1e-6*(x**2. + (z-4000.)**2.))
elif testCase == "densityCurrent":
    left    = -25600.
    right   = 25600.
    bottom  = 0.
    top     = 6400.
    dx      = 400. / 2**refinementLevel
    nLev    = 16 * 2**refinementLevel
    dt      = 1./3. / 2**refinementLevel
    tf      = 900.
    saveDel = 50
    def zSurf(x):
        return np.zeros(np.shape(x))
    def thetaPtb(x, z):
        return -20. * np.exp(-7e-7*(x**2. + (z-3000.)**2.))
elif testCase == "inertiaGravityWaves":
    left    = -150000.
    right   = 150000.
    bottom  = 0.
    top     = 10000.
    dx      = 1000. / 2**refinementLevel
    nLev    = 10 * 2**refinementLevel
    dt      = 1. / 2**refinementLevel
    tf      = 3000.
    saveDel = 250
    def zSurf(x):
        return np.zeros(np.shape(x))
    def thetaPtb(x, z):
        thetaC = .01
        hC = 10000.
        aC = 5000.
        xC = -50000.
        return thetaC * np.sin(np.pi * z / hC) / (1. + ((x - xC) / aC)**2.)
else:
    raise ValueError("Invalid test case string.  Please choose " \
    + "'risingBubble', 'densityCurrent', or 'inertiaGravityWaves'.")

###########################################################################

t = 0.

nTimesteps = np.int(np.round(tf / dt))

nCol = np.int(np.round((right - left) / dx))

x = np.linspace(left+dx/2, right-dx/2, nCol)

zSurf = zSurf(x)            #over-write zSurf function with array of values

###########################################################################

def getSvalues():

    if verticalCoordinate == "height":
        #This is a strange height coordinate, because it starts at zero
        #at the top and goes up to 1 at the bottom.  This is so it mimics
        #the behavior of the pressure coordinate, so that we can use the
        #same setGhostNodes() function for both coordinates.
        ds = 1. / nLev
        s = np.linspace(-ds/2, 1+ds/2, nLev+2)
        #These are not explicitly needed in height coordinate case:
        pSurf = 0.
        sTop = 0.
        pTop = 0.
    elif verticalCoordinate == "pressure":
        piTop  = exnerPressure(top)
        piSurf = exnerPressure(zSurf)
        pSurf = Po * piSurf ** (Cp/Rd)     #hydrostatic pressure at surface
        pTop  = Po * piTop  ** (Cp/Rd)         #hydrostatic pressure at top
        sTop = pTop / Po                      #value of s on upper boundary
        ds = (1. - sTop) / nLev
        s = np.linspace(sTop-ds/2., 1.+ds/2., nLev+2)
    else:
        raise ValueError("Invalid verticalCoordinate string.  Please " \
        + "choose either 'height' or 'pressure'.")

    return s, ds, sTop, pTop, pSurf

###########################################################################

s, ds, sTop, pTop, pSurf = getSvalues()

xx, ss = np.meshgrid(x, s)

###########################################################################

#All of the polyharmonic spline radial basis function weights:

phs = 11                                 #lateral PHS exponent (odd number)
pol = 5                         #highest degree polynomial in lateral basis
stc = 11                                              #lateral stencil size
alp = 2.**-9. * 300.                       #lateral dissipation coefficient
Wa   = phs1.getPeriodicDM(z=x, x=x, m=1 \
, phs=phs, pol=pol, stc=stc, period=right-left) #lateral derivative weights
Whva = phs1.getPeriodicDM(z=x, x=x, m=pol+1 \
, phs=phs, pol=pol, stc=stc, period=right-left)
Whva = alp * dx**pol * Whva                    #lateral dissipation weights

phs = 5                                              #vertical PHS exponent
pol = 3                        #highest degree polynomial in vertical basis
stc = 5                                              #vertical stencil size
if verticalCoordinate == "pressure":
    alp = -2.**-22. * 300.                #vertical dissipation coefficient
else:
    alp = -2.**-22. * 300.          #much larger in height coordinate case?
Ws   = phs1.getDM(z=s,       x=s, m=1 \
, phs=phs, pol=pol, stc=stc)                   #vertical derivative weights
Whvs = phs1.getDM(z=s[1:-1], x=s, m=pol+1 \
, phs=phs, pol=pol, stc=stc)
Whvs = alp * ds**pol * Whvs                   #vertical dissipation weights

wItop = phs1.getWeights(z=(s[0]+s[1])/2.,   x=s[0:stc],        m=0 \
, phs=phs, pol=pol)                            #interpolate to top boundary
wEtop = phs1.getWeights(z=s[0],             x=s[1:stc+1],      m=0 \
, phs=phs, pol=pol)                         #extrapolate to top ghost nodes
wDtop = phs1.getWeights(z=(s[0]+s[1])/2.,   x=s[0:stc],        m=1 \
, phs=phs, pol=pol)                             #derivative on top boundary
wHtop = phs1.getWeights(z=(s[0]+s[1])/2.,   x=s[1:stc+1],      m=0 \
, phs=phs, pol=pol)                            #extrapolate to top boundary

wIbot = phs1.getWeights(z=(s[-2]+s[-1])/2., x=s[-1:-1-stc:-1], m=0 \
, phs=phs, pol=pol)                         #interpolate to bottom boundary
wEbot = phs1.getWeights(z=s[-1],            x=s[-2:-2-stc:-1], m=0 \
, phs=phs, pol=pol)                      #extrapolate to bottom ghost nodes
wDbot = phs1.getWeights(z=(s[-2]+s[-1])/2., x=s[-1:-1-stc:-1], m=1 \
, phs=phs, pol=pol)                          #derivative on bottom boundary
wHbot = phs1.getWeights(z=s[-1],            x=s[-2:-2-stc:-1], m=0 \
, phs=phs, pol=pol)                         #extrapolate to bottom boundary

#Lateral derivative on all levels:
def Da(U):
    return Wa.dot(U.T).T

#Vertical first derivative on all levels:
def Ds(U):
    return Ws.dot(U)

#Total dissipation on non-ghost levels:
def HV(U):
    return Whva.dot(U[1:-1,:].T).T + Whvs.dot(U)

###########################################################################

def getVerticalLevels(zSurf, top):

    if verticalCoordinate == "height":
        
        zz = np.zeros((nLev+2, nCol))
        
        for j in range(nCol):
            dz = (zSurf[j] - top) * ds               #Note:  dz is negative
            zz[:,j] \
            = np.flipud(np.linspace(zSurf[j]+dz/2, top-dz/2, nLev+2))

        #These are not explicitly needed in the height coordinate case:
        A = 0.
        B = 0.
        ssInt = 0.
        
    elif verticalCoordinate == "pressure":
        
        def A(s):
            return (1. - s) / (1. - sTop) * sTop
        
        def Aprime(s):
            return -sTop / (1. - sTop)

        def B(s):
            return (s - sTop) / (1. - sTop)
        
        def Bprime(s):
            return 1. / (1. - sTop)
        
        ssInt = (ss[0:-1,:] + ss[1:,:]) / 2.          #s-mesh on interfaces
        p = A(ssInt) * Po + B(ssInt) * np.tile(pSurf,(nLev+1,1))
        pi = (p / Po) ** (Rd/Cp)
        zz0 = inverseExnerPressure(pi)
        zz = zz0.copy()

        #Iterate to satisfy initial conditions and hydrostatic condition:
        for j in range(10):
            T = exnerPressure(zz) * potentialTemperature(zz)
            # T = exnerPressure(zz) \
            # * (potentialTemperature(zz) + thetaPtb(xx[1:,:],zz))
            integrand = -Rd * T / (A(ssInt) * Po + B(ssInt) * pSurf) \
            * (Aprime(ssInt) * Po + Bprime(ssInt) * pSurf) / g
            integrand = (integrand[0:-1,:] + integrand[1:,:]) / 2.
            tmp = zz[0,:].copy()
            for i in range(nLev):
                tmp = tmp + integrand[i,:] * ds        #midpoint quadrature
                zz[i+1,:] = tmp.copy()
            # plt.figure()
            # plt.clf()
            # plt.contourf(xx[1:,:], zz0, zz-zz0, 20)
            # plt.colorbar()
            # # plt.axis("image")
            # plt.title("{0:g}".format(np.max(np.abs(zSurf-zz[-1,:]))))
            # plt.show()

        top = zz[0,:]                     #slightly different top of domain
        zSurf = zz[-1,:]               #slightly different bottom of domain
        zz = (zz[0:-1,:] + zz[1:,:]) / 2.
        zz = np.vstack((2.*top - zz[0,:] \
        , zz \
        , 2.*zSurf - zz[-1,:]))     #interfaces -> layer midpts

    zSurfPrime = Wa.dot(zSurf.T).T #approximate derivative of topo function

    return zz, A, B, zSurf, top, zSurfPrime, ssInt

###########################################################################

zz, A, B, zSurf, top, zSurfPrime, ssInt = getVerticalLevels(zSurf, top)

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

def backgroundStatesAndPerturbations(zz):
    
    thetaBar = potentialTemperature(zz)
    piBar = exnerPressure(zz)
    piPtb = np.zeros((nLev+2, nCol))
    Tbar = piBar * thetaBar
    Tptb = (piBar + piPtb) * (thetaBar + thetaPtb(xx,zz)) - Tbar
    Pbar = Po * piBar ** (Cp/Rd)
    Pptb = Po * (piBar + piPtb) ** (Cp/Rd) - Pbar
    rhoBar = Pbar / Rd / Tbar
    rhoPtb = (Pbar + Pptb) / Rd / (Tbar + Tptb) - rhoBar
    phiBar = g * zz
    
    return thetaBar, piBar, Tbar, Pbar, rhoBar, phiBar \
    , Tptb, rhoPtb

###########################################################################

#Assignment of hydrostatic background states and initial perturbations:
thetaBar, piBar, Tbar, Pbar, rhoBar, phiBar \
, Tptb, rhoPtb = backgroundStatesAndPerturbations(zz)

#Assignment of initial conditions:
U = np.zeros((6, nLev+2, nCol))
if testCase == "inertiaGravityWaves":
    U[0,:,:] =  20. * np.ones((nLev+2, nCol))          #horizontal velocity
U[1,:,:] = np.zeros((nLev+2, nCol))                      #vertical velocity
U[2,:,:] = Tptb                                   #temperature perturbation
U[3,:,:] = rhoPtb                                     #density perturbation
U[4,:,:] = phiBar.copy()                                      #geopotential
U[5,:,:] = np.zeros((nLev+2, nCol))                  #pressure perturbation

###########################################################################

#Unit tangent and unit normal vectors along bottom and top boundaries:

TzBot = zSurfPrime
TxBot = np.ones((nCol))
tmp = np.sqrt(TxBot**2 + TzBot**2)
TxBot = TxBot / tmp                  #x-component of unit tangent on bottom
TzBot = TzBot / tmp                  #z-component of unit tangent on bottom

NxBot = np.tile(-TzBot, (stc-1,1))    #x-component of unit normal on bottom
NzBot = np.tile(TxBot, (stc-1,1))     #z-component of unit normal on bottom

#Tile the tangent and normal vectors vertically according to the stencil
#size, so that they can be more effectively used inside the
#setGhostNodes() function below.  Inside this function, the tangent
#velocity will be extrapolated to the ghost nodes, while the normal
#velocity will be found by forcing the boundary value to be zero (no-flux
#boundary condition):

TxBot = np.tile(TxBot, (stc,1))
TzBot = np.tile(TzBot, (stc,1))

NxTop = np.zeros((stc-1, nCol))
NzTop = np.ones((stc-1, nCol))

TxTop = np.ones((stc, nCol))
TzTop = np.zeros((stc, nCol))

###########################################################################

if saveContours :
    #Initialize a figure of the appropriate size:
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

if verticallyLagrangian:
    if verticalCoordinate == "height":
        V = np.zeros((5, nLev+2, nCol))
    else:
        V = np.zeros((5, nLev+2, nCol))

###########################################################################

def verticalRemap(U, z, Z, V):          #used only in vertically Lagrangian
    """
    Interpolate columns of 3D array U from z to Z using quadratics
    nLev is the number of interior levels of U
    z and Z are 2D arrays such that each row is a new z-level
    V is the output array
    """
    z = np.tile(z, (np.shape(U)[0], 1, 1))
    Z = np.tile(Z, (np.shape(U)[0], 1, 1))

    #quadratic on bottom:
    z0 = z[:,0,:]                                               #first node
    z1 = z[:,1,:]                                              #second node
    z2 = z[:,2,:]                                               #third node
    ZZ = Z[:,0,:]                                         #evaluation point
    V[:,0,:] = \
      (ZZ - z1) * (ZZ - z2) * U[:,0,:] / (z0 - z1) / (z0 - z2) \
    + (ZZ - z0) * (ZZ - z2) * U[:,1,:] / (z1 - z0) / (z1 - z2) \
    + (ZZ - z0) * (ZZ - z1) * U[:,2,:] / (z2 - z0) / (z2 - z1)

    #quadratic on interior:
    z0 = z[:,0:nLev+0,:]
    z1 = z[:,1:nLev+1,:]
    z2 = z[:,2:nLev+2,:]
    ZZ = Z[:,1:nLev+1,:]
    V[:,1:nLev+1,:] = \
      (ZZ - z1) * (ZZ - z2) * U[:,0:nLev+0,:] / (z0 - z1) / (z0 - z2) \
    + (ZZ - z0) * (ZZ - z2) * U[:,1:nLev+1,:] / (z1 - z0) / (z1 - z2) \
    + (ZZ - z0) * (ZZ - z1) * U[:,2:nLev+2,:] / (z2 - z0) / (z2 - z1)

    #quadratic on top:
    z0 = z[:,nLev-1,:]
    z1 = z[:,nLev+0,:]
    z2 = z[:,nLev+1,:]
    ZZ = Z[:,nLev+1,:]
    V[:,nLev+1,:] = \
      (ZZ - z1) * (ZZ - z2) * U[:,nLev-1,:] / (z0 - z1) / (z0 - z2) \
    + (ZZ - z0) * (ZZ - z2) * U[:,nLev+0,:] / (z1 - z0) / (z1 - z2) \
    + (ZZ - z0) * (ZZ - z1) * U[:,nLev+1,:] / (z2 - z0) / (z2 - z1)

    return V

###########################################################################

def fastBackgroundStates(phi):
    
    zz = phi / g
    
    thetaBar = potentialTemperature(zz)
    piBar = exnerPressure(zz)
    dthetaBarDz = potentialTemperatureDerivative(zz)
    
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
    U[4,0,:] = (g*top - wItop[1:stc].dot(U[4,1:stc,:])) / wItop[0]
    
    #Get background states on possibly changing geopotential levels:
    Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz \
    = fastBackgroundStates(U[4,:,:])
    
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
    U[5,1:-1,:] = ((rhoBar+U[3,:,:]) * Rd \
    * (Tbar+U[2,:,:]) - Pbar)[1:-1,:]
    
    #set pressure on bottom ghost nodes:
    dPda = Wa.dot(wHbot.dot(U[5,-2:-2-stc:-1,:]).T).T
    rho = wHbot.dot(U[3,-2:-2-stc:-1,:])
    dphida = Wa.dot(wIbot.dot(U[4,-1:-1-stc:-1,:]).T).T
    dphids = wDbot.dot(U[4,-1:-1-stc:-1,:])
    dsdx = -dphida / dphids
    dsdz = g / dphids
    RHS = -rho * g * NzBot[0,:] - dPda * NxBot[0,:]
    RHS = RHS / (NxBot[0,:] * dsdx + NzBot[0,:] * dsdz)
    U[5,-1,:] = (RHS - wDbot[1:stc].dot(U[5,-2:-1-stc:-1,:])) / wDbot[0]
    
    #set pressure on top ghost nodes:
    dPda = Wa.dot(wHtop.dot(U[5,1:stc+1,:]).T).T
    rho = wHtop.dot(U[3,1:stc+1,:])
    dphida = Wa.dot(wItop.dot(U[4,0:stc,:]).T).T
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
    
    if verticallyLagrangian:
        sDot = np.zeros((nLev+2, nCol))
    else:
        if verticalCoordinate == "height":
            sDot = uDotGradS
        else:
            #please see this overleaf document for a complete derivation,
            #starting from the governing equation for pseudo-density dpids:
            #https://www.overleaf.com/read/gcfkprynxvkw
            sDot = np.zeros((nLev+1, nCol))
            dpids = -(rhoBar+U[3,:,:]) * Ds(U[4,:,:])#hydrostatic condition
            integrand = Da(dpids * U[0,:,:])[1:-1,:]
            dpids = (dpids[0:-1,:] + dpids[1:,:]) / 2.   #avg to interfaces
            sDot = B(ssInt) * np.tile(np.sum(integrand*ds,0), (nLev+1,1))
            tmp = 0.
            for j in range(nLev):
                tmp = tmp + integrand[j,:] * ds
                sDot[j+1,:] = sDot[j+1,:] - tmp
            sDot = sDot / dpids
            sDot = (sDot[0:-1,:] + sDot[1:,:]) / 2.
            sDot = np.vstack((2. * 0. - sDot[0,:] \
            , sDot \
            , 2. * 0. - sDot[-1,:]))   #sDot=0 on bottom and top boundaries
    
    #Main part:
    
    dUdt[0,1:-1,:] = (-U[0,:,:] * duda - sDot * duds \
    - rhoInv * (Da(U[5,:,:]) + dPds * dsdx))[1:-1,:] \
    + HV(U[0,:,:])                                                   #du/dt
    
    dUdt[1,1:-1,:] = (-U[0,:,:] * dwda - sDot * dwds \
    - rhoInv * (dPds * dsdz) - U[3,:,:] * g * rhoInv)[1:-1,:] \
    + HV(U[1,:,:])                                                   #dw/dt

    dUdt[2,1:-1,:] = (-U[0,:,:] * Da(U[2,:,:]) - sDot * Ds(U[2,:,:]) \
    - U[1,:,:] * dTbarDz - Rd/Cv * (Tbar + U[2,:,:]) * divU)[1:-1,:] \
    + HV(U[2,:,:])                                                   #dT/dt

    dUdt[3,1:-1,:] = (-U[0,:,:] * Da(U[3,:,:]) - sDot * Ds(U[3,:,:]) \
    - U[1,:,:] * drhoBarDz - (rhoBar + U[3,:,:]) * divU)[1:-1,:] \
    + HV(U[3,:,:])                                                 #drho/dt

    dUdt[4,1:-1,:] = ((uDotGradS - sDot) * dphids)[1:-1,:] \
    + 0.*HV(U[4,:,:] - phiBar)                                     #dphi/dt
    
    return dUdt

###########################################################################

#Main time-stepping loop:

# U = setGhostNodes(U)[0]
# plt.contourf(xx, zz, U[5,:,:], 20)
# plt.colorbar()
# plt.show()
# sys.exit("Done for now.")

et = time.time()

for i in np.arange(0, nTimesteps+1):
    
    #Vertical re-map:
    if verticallyLagrangian and not contourFromSaved \
    and (np.mod(i,4) == 0) and (testCase != "inertiaGravityWaves"):
        if verticalCoordinate == "height":
            U = setGhostNodes(U)[0]
            U[0:5,:,:] = verticalRemap(U[0:5,:,:], U[4,:,:], phiBar, V)
            # U[4,:,:] = phiBar.copy()
        else:
            tmp = setGhostNodes(U)
            U = tmp[0]
            rhoBar = tmp[2]
            integrand = (-(rhoBar+U[3,:,:]) * Ds(U[4,:,:]))[1:-1,:]
            # integrand = -(rhoBar + U[3,:,:])[1:-1,:]
            # dPhi = (U[4,0:-1,:] + U[4,1:,:]) / 2.
            # dPhi = dPhi[1:,:] - dPhi[0:-1,:]
            tmp = pTop * np.ones((nCol))
            pHydro = np.zeros((nLev+1, nCol))
            pHydro[0,:] = tmp.copy()
            for j in range(nLev):
                tmp = tmp + integrand[j,:] * ds
                # tmp = tmp + integrand[j,:] * dPhi[j,:]
                pHydro[j+1,:] = tmp.copy()
            pHydroSurf = tmp.copy()
            pHydro = (pHydro[0:-1,:] + pHydro[1:,:])/2.
            pHydro = np.vstack((2.*pTop - pHydro[0,:] \
            , pHydro \
            , 2.*pHydroSurf - pHydro[-1,:]))
            
            pHydroNew = A(ss) * Po \
            + B(ss) * np.tile(pHydroSurf, (nLev+2, 1))

            # dP = A(ssInt) * Po \
            # + B(ssInt) * np.tile(pHydroSurf, (nLev+1,1))
            # dP = dP[1:,:] - dP[0:-1,:]
            U[0:5,:,:] = verticalRemap(U[0:5,:,:], pHydro, pHydroNew, V)
            # tmp = setGhostNodes(U)
            # U = tmp[0]
            # rhoBar = tmp[2]
            # dPhi = (U[4,0:-1,:] + U[4,1:,:]) / 2.
            # dPhi = dPhi[1:,:] - dPhi[0:-1,:]
            # U[3,1:-1,:] = -dP / dPhi - rhoBar[1:-1,:]

            # plt.clf()
            # plt.contourf(xx, zz, pHydro-pHydroNew, 20)
            # plt.colorbar()
            # plt.axis('image')
            # plt.show()
    
    if np.mod(i, np.int(np.round(saveDel/dt))) == 0:
        
        print("t = {0:5d} | et = {1:6.2f} | maxAbsRho = {2:.2e}" \
        . format(np.int(np.round(t)) \
        , time.time()-et \
        , np.max(np.abs(U[3,:,:]))))
        
        et = time.time()
        
        if contourFromSaved :
            U[0:5,:,:] = np.load( saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy' )
        
        if saveArrays or saveContours:
            U = setGhostNodes(U)[0]
        
        if saveArrays:
            np.save(saveString \
            + '{0:04d}'.format(np.int(np.round(t))) + '.npy', U[0:5,:,:])
        
        if saveContours:
            contourSomething(U, t)
    
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
