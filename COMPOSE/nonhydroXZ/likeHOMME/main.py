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
testCase = "risingBubble"

#Choose "pressure" or "height":
verticalCoordinate = "height"

#Choose 0, 1, 2, 3, or 4:
refinementLevel = 1

verticallyLagrangian           = False
plotNodesAndExit               = False
contourBackgroundStatesAndExit = False

###########################################################################

#Definitions of constants:

Cp = 1004.                       #specific heat of air at constant pressure
Cv = 717.                          #specific heat of air at constant volume
Rd = Cp - Cv                                      #gas constant for dry air
g  = 9.81                                           #gravitational constant
Po = 10.**5.                                 #reference pressure in Pascals

th0 = 300.                                 #reference potential temperature
N   = .01                                          #Brunt-Vaisala frequency

###########################################################################

#All of the test cases are defined in terms of hydrostatic background
#states for potential temperature (th) and Exner pressure (pi), and these
#are functions of z only.  In the pressure coordinate case, the initial
#s-levels are defined using the inverse Exner pressure function.

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
    
    raise ValueError("Invalid test case string.")

###########################################################################

#Get some test-specific parameters, such as the size of the domain, the
#horizontal node spacing dx, and the number of vertical levels nLev:

if testCase == "risingBubble":
    left   = -5000.
    right  = 5000.
    bottom = 0.
    top    = 10000.
    dx     = 400. / 2**refinementLevel
    nLev   = 25 * 2**refinementLevel
    dt     = 1./4. / 2**refinementLevel
    tf     = 1200.
    def zSurf(x):
        return 1000. * np.sin(2.*np.pi / 10000. * x)
    def zSurfPrime(x):
        return 1000. * 2.*np.pi/10000 * np.cos(2.*np.pi / 10000. * x)
    def thetaPrime(x, z):
        return 2. * np.exp(-7e-7*(x**2. + (z-5000.)**2.))
elif testCase == "densityCurrent":
    left   = -25600.
    right  = 25600.
    bottom = 0.
    top    = 6400.
    dx     = 400. / 2**refinementLevel
    nLev   = 16 * 2**refinementLevel
    dt     = 1./6. / 2**refinementLevel
    tf     = 900.
    def zSurf(x):
        return np.zeros(np.shape(x))
    def zSurfPrime(x):
        return np.zeros(np.shape(x))
    def thetaPrime(x, z):
        return -20. * np.exp(-7e-7*(x**2. + (z-3000.)**2.))
elif testCase == "inertiaGravityWaves":
    left   = -150000.
    right  = 150000.
    bottom = 0.
    top    = 10000.
    dx     = 1000. / 2**refinementLevel
    nLev   = 10 * 2**refinementLevel
    dt     = 1. / 2**refinementLevel
    tf     = 3000.
    def zSurf(x):
        return np.zeros(np.shape(x))
    def zSurfPrime(x):
        return np.zeros(np.shape(x))
    def thetaPrime(x, z):
        thetaC = .01
        hC = 10000.
        aC = 5000.
        xC = -50000.
        return thetaC * np.sin(np.pi * z / hC) / (1. + ((x - xC) / aC)**2.)
else:
    raise ValueError("Invalid test case string.")

###########################################################################

nTimesteps = np.int(np.round(tf / dt))

nCol = np.int(np.round((right - left) / dx))

x = np.linspace(left+dx/2, right-dx/2, nCol)

###########################################################################

def getLevels(verticalCoordinate):
        
    if verticalCoordinate == "height":
        
        ds = (top - bottom) / nLev
        s = np.linspace(bottom-ds/2, top+ds/2, nLev+2)
        zz = np.zeros((nLev+2, nCol))
        
        for j in range(nCol):
            dz = (top - zSurf(x[j])) / (top - bottom) * ds
            zz[:,j] = np.linspace(zSurf(x[j])-dz/2, top+dz/2, nLev+2)
        
    elif verticalCoordinate == "pressure":
        
        piSurf = exnerPressure(zSurf(x))
        piTop  = exnerPressure(top)
        pSurf = Po * piSurf ** (Cp/Rd)     #hydrostatic pressure at surface
        pTop  = Po * piTop  ** (Cp/Rd)         #hydrostatic pressure at top
        sTop = pTop / Po                      #value of s on upper boundary
        ds = (1. - sTop) / nLev
        s = np.linspace(sTop-ds/2., 1.+ds/2., nLev+2)
        s = np.flipud(s)
        ss = np.tile(s, (nCol,1)).T
        def A(s):
            return (1. - s) / (1. - sTop) * sTop
        def B(s):
            return (s - sTop) / (1. - sTop)
        p = A(ss) * Po + B(ss) * np.tile(pSurf,(nLev+2,1))
        pi = (p / Po) ** (Rd/Cp)
        zz = inverseExnerPressure(pi)
        
    return s, ds, zz

###########################################################################

s, ds, zz = getLevels(verticalCoordinate)

xx, ss = np.meshgrid(x, s)

if plotNodesAndExit:
    plt.figure()
    plt.plot(xx.flatten(), zz.flatten(), '.')
    plt.plot(x, zSurf(x), 'r-')
    plt.plot(x, top*np.ones(np.shape(x)), 'r-')
    plt.axis("image")
    plt.show()
    sys.exit("Finished plotting.")

###########################################################################

#Assignment of hydrostatic background states:
thetaBar = potentialTemperature(zz)
piBar = exnerPressure(zz)
piPrime = np.zeros((nLev+2, nCol))
Tbar = piBar * thetaBar
Tprime = (piBar + piPrime) * (thetaBar + thetaPrime(xx,zz)) - Tbar
Pbar = Po * piBar ** (Cp/Rd)
Pprime = Po * (piBar + piPrime) ** (Cp/Rd) - Pbar
rhoBar = Pbar / Rd / Tbar
rhoPrime = (Pbar + Pprime) / Rd / (Tbar + Tprime) - rhoBar
phiBar = g * zz

#Assignment of initial conditions:
U = np.zeros((6, nLev+2, nCol))
if testCase == "igw":
    U[0,:,:] =  20. * np.ones((nLev+2, nCol))
U[2,:,:] = Tprime
U[3,:,:] = rhoPrime
U[4,:,:] = phiBar

if contourBackgroundStatesAndExit:
    plt.figure()
    plt.contourf(xx, zz, U[2,:,:], 20)
    plt.axis("image")
    plt.colorbar()
    plt.show()
    sys.exit("Finished contour plots.")

###########################################################################

#All of the polyharmonic spline radial basis function weights:

phs = 11
pol = 5
stc = 11
alp = 2.**-10.
Wa   = phs1.getPeriodicDM(period=right-left, x=x, X=x, m=1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc)
Whva = phs1.getPeriodicDM(period=right-left, x=x, X=x, m=pol+1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc)
Whva = alp * dx**pol * Whva
phs = 5
pol = 3
stc = 7
alp = -2.**-7.
Ws   = phs1.getDM(x=s, X=s,       m=1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc)
Whvs = phs1.getDM(x=s, X=s[1:-1], m=pol+1 \
, phsDegree=phs, polyDegree=pol, stencilSize=stc)
Whvs = alp * ds**pol * Whvs

wIbot = phs1.getWeights(0.,   s[0:stc],   0, phs, pol)
wEbot = phs1.getWeights(s[0], s[1:stc+1], 0, phs, pol)
wDbot = phs1.getWeights(0.,   s[0:stc],   1, phs, pol)
wHbot = phs1.getWeights(0.,   s[1:stc+1], 0, phs, pol)

wItop = phs1.getWeights(top,   s[-1:-1-stc:-1], 0, phs, pol)
wEtop = phs1.getWeights(s[-1], s[-2:-2-stc:-1], 0, phs, pol)
wDtop = phs1.getWeights(top,   s[-1:-1-stc:-1], 1, phs, pol)
wHtop = phs1.getWeights(s[-1], s[-2:-2-stc:-1], 0, phs, pol)

def Da(U):
    return Wa.dot(U.T).T

def Ds(U):
    return Ws.dot(U)

def HV(U):
    return Whva.dot(U[1:-1,:].T).T + Whvs.dot(U)

###########################################################################

#Unit tangent and unit normal vectors along bottom and top boundaries:

TzBot = zSurfPrime(x)
TxBot = np.ones((nCol))
tmp = np.sqrt(TxBot**2 + TzBot**2)
TxBot = TxBot / tmp
TzBot = TzBot / tmp

NxBot = np.tile(-TzBot, (stc-1,1))
NzBot = np.tile(TxBot, (stc-1,1))

TxBot = np.tile(TxBot, (stc,1))
TzBot = np.tile(TzBot, (stc,1))

NxTop = np.zeros((stc-1, nCol))
NzTop = np.ones((stc-1, nCol))

TxTop = np.ones((stc, nCol))
TzTop = np.zeros((stc, nCol))

###########################################################################

def fastBackgroundStates(phi):
    
    z = phi / g
    
    thetaBar = potentialTemperature(z)
    piBar = exnerPressure(z)
    dthetaBarDz = potentialTemperatureDerivative(z)
    
    Tbar = piBar * thetaBar
    Pbar = Po * piBar ** (Cp/Rd)
    rhoBar = Pbar / Rd / Tbar
    
    dpiBarDz = -g / Cp / thetaBar                    #hydrostatic condition
    dTbarDz = piBar * dthetaBarDz + thetaBar * dpiBarDz
    dPbarDz = Po * Cp/Rd * piBar**(Cp/Rd-1.) * dpiBarDz
    drhoBarDz = ( dPbarDz - Rd*rhoBar*dTbarDz ) / ( Rd * Tbar )
    
    return Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz

###########################################################################

def setGhostNodes(U):
    
    #Enforce phi=g*z on bottom boundary:
    U[4,0,:] = (g*zSurf(x) - wIbot[1:stc].dot(U[4,1:stc,:])) / wIbot[0]
    
    #Enforce phi=g*z on top boundary:
    U[4,-1,:] = (g*top - wItop[1:stc].dot(U[4,-2:-1-stc:-1,:])) / wItop[0]
    
    #Get background states on possibly changing geopotential levels:
    Pbar, rhoBar, Tbar, drhoBarDz, dTbarDz = fastBackgroundStates(U[4,:,:])
    
    #extrapolate tangent velocity uT to bottom ghost nodes:
    uT = U[0,1:stc+1,:] * TxBot + U[1,1:stc+1,:] * TzBot
    uT = wEbot.dot(uT)
    
    #get normal velocity uN on bottom ghost nodes:
    uN = U[0,1:stc,:] * NxBot + U[1,1:stc,:] * NzBot
    uN = -wIbot[1:stc].dot(uN) / wIbot[0]
    
    #use uT and uN to get (u,w) on bottom ghost nodes, then get (u,w) on
    #top ghost nodes:
    U[0,0,:] = uT*TxBot[0,:] + uN*NxBot[0,:]
    U[1,0,:] = uT*TzBot[0,:] + uN*NzBot[0,:]
    U[0,-1,:] = wEtop.dot(U[0,-2:-(stc+2):-1,:])
    U[1,-1,:] = -wItop[1:stc].dot(U[1,-2:-(stc+1):-1,:]) / wItop[0]
    
    #get pressure on interior nodes using the equation of state:
    U[5,1:-1,:] = ((rhoBar+U[3,:,:]) * Rd * (Tbar+U[2,:,:]) - Pbar)[1:-1,:]
    
    #set pressure on bottom ghost nodes:
    dPda = Wa.dot(wHbot.dot(U[5,1:stc+1,:]).T).T
    rho = wHbot.dot(U[3,1:stc+1,:])
    dphida = Wa.dot(wIbot.dot(U[4,0:stc,:]).T).T
    dphids = wDbot.dot(U[4,0:stc,:])
    dsdx = -dphida / dphids
    dsdz = g / dphids
    RHS = -rho * g * NzBot[0,:] - dPda * NxBot[0,:]
    RHS = RHS / (NxBot[0,:] * dsdx + NzBot[0,:] * dsdz)
    U[5,0,:] = (RHS - wDbot[1:stc].dot(U[5,1:stc,:])) / wDbot[0]
    
    #set pressure on top ghost nodes:
    dPda = Wa.dot(wHtop.dot(U[5,-2:-2-stc:-1,:]).T).T
    rho = wHtop.dot(U[3,-2:-2-stc:-1,:])
    dphida = Wa.dot(wItop.dot(U[4,-1:-1-stc:-1,:]).T).T
    dphids = wDtop.dot(U[4,-1:-1-stc:-1,:])
    dsdx = -dphida / dphids
    dsdz = g / dphids
    RHS = -rho * g * NzTop[0,:] - dPda * NxTop[0,:]
    RHS = RHS / (NxTop[0,:] * dsdx + NzTop[0,:] * dsdz)
    U[5,-1,:] = (RHS - wDtop[1:stc].dot(U[5,-2:-1-stc:-1,:])) / wDtop[0]
    
    #extrapolate temperature to bottom and top ghost nodes:
    U[2,0,:] = wEbot.dot(U[2,1:stc+1,:])
    U[2,-1,:] = wEtop.dot(U[2,-2:-(stc+2):-1,:])
    
    #extrapolate density to bottom and top ghost nodes using EOS:
    U[3,0,:] = (Pbar[0,:]+U[5,0,:]) / Rd / (Tbar[0,:]+U[2,0,:]) \
    - rhoBar[0,:]
    U[3,-1,:] = (Pbar[-1,:]+U[5,-1,:]) / Rd / (Tbar[0,:]+U[2,-1,:]) \
    - rhoBar[-1,:]
    
    return U

###########################################################################

def odefun(t, U, dUdt):
    
    U = setGhostNodes(U)
    
    rhoInv = 1. / (rhoBar + U[3,:,:])
    dPds = Ds(U[5,:,:])
    duda = Da(U[0,:,:])
    duds = Ds(U[0,:,:])
    dwda = Da(U[1,:,:])
    dwds = Ds(U[1,:,:])
    dphids = Ds(U[4,:,:])
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
            sDot = getSdot(U)
    
    dUdt[0,:,:] = (-U[0,:,:] * duda - sDot * duds \
    - rhoInv * (Da(U[5,:,:]) + dPds * dsdx))[1:-1,:] \
    + HV(U[0,:,:])
    
    dUdt[1,:,:] = (-U[0,:,:] * dwda - sDot * dwds \
    - rhoInv * (dPds * dsdz) - U[3,:,:] * g * rhoInv)[1:-1,:] \
    + HV(U[1,:,:])

    dUdt[2,:,:] = (-U[0,:,:] * Da(U[2,:,:]) - sDot * Ds(U[2,:,:]) \
    - U[1,:,:] * dTbarDz - Rd/Cv * (Tbar + U[2,:,:]) * divU)[1:-1,:] \
    + HV(U[2,:,:])

    dUdt[3,:,:] = (-U[0,:,:] * Da(U[3,:,:]) - sDot * Ds(U[3,:,:]) \
    - U[1,:,:] * drhoBarDz - (rhoBar + U[3,:,:]) * divU)[1:-1,:] \
    + HV(U[3,:,:])
    
    return dUdt

###########################################################################

#Main time-stepping loop:

# for i in range(1, nTimesteps+1):
    
    

###########################################################################
