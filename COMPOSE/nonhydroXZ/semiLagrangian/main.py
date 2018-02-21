import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import sparse

sys.path.append( '../../../site-packages' )
from gab import nonhydro, rk, phs2

###########################################################################

#"bubble", "igw", "densityCurrent", "doubleDensityCurrent",
#or "movingDensityCurrent":
testCase = "doubleDensityCurrent"

#"exner" or "hydrostaticPressure":
formulation = "exner"

semiLagrangian = 0                   #Set this to zero.  SL not working yet
rbfDerivatives = 0                  #Set this to zero.  RBF not working yet
dx = 50.
ds = 50.
FD = 4                                    #Order of lateral FD (2, 4, or 6)
rbfOrder    = 3
polyOrder   = 1
stencilSize = 9
K           = FD/2+1                #determines exponent in HV for RBF case
rkStages = 3
plotNodes = 0                               #if 1, plot nodes and then exit
saveDel = 50                              #print/save every saveDel seconds

var           = 3                        #determines what to plot (0,1,2,3)
saveArrays    = 1 
saveContours  = 1
plotFromSaved = 0                   #if 1, results are loaded, not computed

###########################################################################

t = 0.

saveString = './results/' + testCase + '/' \
+ 'dx' + '{0:1d}'.format(np.int(np.round(dx)+1e-12)) \
+ 'ds' + '{0:1d}'.format(np.int(np.round(ds)+1e-12)) + '/'

###########################################################################

#QUESTION:  Does this weird function thing make a difference?

Cp, Cv, Rd, g, Po = nonhydro.getConstants()

###########################################################################

xLeft, xRight, nLev, nCol, zTop, zSurf, zSurfPrime, x, z \
= nonhydro.getSpaceDomain( testCase, dx, ds, FD )

xVec = x.flatten()
zVec = z.flatten()

tf, dt, dtEul, nTimesteps = nonhydro.getTimeDomain( testCase, dx, ds )

s, dsdx, dsdz = nonhydro.getHeightCoordinate( zTop, zSurf, zSurfPrime )

FDo2 = np.int( FD/2 )
ii = np.arange( 1, nLev+1 )
jj = np.arange( FDo2, nCol+FDo2 )
i0 = ii[0]
i1 = ii[-1] + 1
j0 = jj[0]
j1 = jj[-1] + 1

Tx, Tz, Nx, Nz = nonhydro.getTanNorm( zSurfPrime, x[0,jj] )

U, thetaBar, piBar, dpidsBar \
= nonhydro.getInitialConditions( testCase, formulation \
, nLev, nCol, FD, x, z \
, Cp(), Cv(), Rd(), g(), Po() \
, dsdz )

ind = nonhydro.getIndexes( x, z, xLeft, xRight, zSurf, zTop, FD \
, nLev, nCol )

if plotNodes == 1 :
    nonhydro.plotNodes( x, z, ind, testCase )
    sys.exit( "\nDone plotting nodes.\n" )
    
###########################################################################

#Derivatives of height coordinate function s:

dsdxBottom = dsdx( x[0,jj], zSurf(x[0,jj]) )
dsdzBottom = dsdz( x[0,jj], zSurf(x[0,jj]) )
dsdxAll    = dsdx( x, z )
dsdzAll    = dsdz( x, z )
dsdxVec    = dsdxAll . flatten()
dsdzVec    = dsdzAll . flatten()
dsdxEul    = dsdx( x[ii,:][:,jj], z[ii,:][:,jj] )
dsdzEul    = dsdz( x[ii,:][:,jj], z[ii,:][:,jj] )

###########################################################################

#Define finite difference (FD) weights for derivative approximation:

if FD == 2 :
    wx = np.array( [ -1./2., 0., 1./2. ] )
    wxhv = np.array( [ 1., -2., 1. ] )
    gamma = 1./2.
elif FD == 4 :
    wx = np.array( [ 1./12., -2./3., 0., 2./3., -1./12. ] )
    wxhv = np.array( [ 1., -4., 6., -4., 1. ] )
    gamma = -1./12.
elif FD == 6 :
    wx = np.array( [ -1./60., 3./20., -3./4., 0., 3./4, -3./20., 1./60. ] )
    wxhv = np.array( [ 1., -6., 15., -20., 15., -6., 1. ] )
    gamma = 1./60.
else :
    sys.exit( "\nError: FD should be 2, 4, or 6.\n" )
wx   = wx   / dx
wxhv = wxhv / dx

ws = np.array( [ -1./2., 0., 1./2. ] )
wshv = np.array( [ 1., -2., 1. ] )
ws   = ws   / ds
wshv = wshv / ds

###########################################################################

bigTx = np.tile( Tx, (2,1) )
bigTz = np.tile( Tz, (2,1) )

normGradS = np.sqrt( dsdxBottom**2. + dsdzBottom**2. )

bigNull = np.zeros(( 4, nLev+2, nCol+FD ))

###########################################################################

#Define functions for approximating derivatives:

if rbfDerivatives == 0 :
    
    def Dx( U ) :
        return nonhydro.LxFD_3D( U, wx,   j0, j1, dx, FD, FDo2 )
    
    def Dx2D( U ) :
        return nonhydro.LxFD_2D( U, wx,   j0, j1, dx, FD, FDo2 )
    
    def Ds( U ) :
        return nonhydro.LsFD_3D( U, ws,   i0, i1, ds )
    
    def Ds2D( U ) :
        return nonhydro.LsFD_2D( U, ws,   i0, i1, ds )
    
    def HVx( U ) :
        return nonhydro.LxFD_3D( U, wxhv, j0, j1, dx, FD, FDo2 )
    
    def HVs( U ) :
        return nonhydro.LsFD_3D( U, wshv, i0, i1, ds )
    
elif rbfDerivatives == 1 :
    
    stencils = phs2.getStencils( xVec, zVec, xVec[ind.m], zVec[ind.m], stencilSize )
    A = phs2.getAmatrices( stencils, rbfOrder, polyOrder )
    Wx  = phs2.getWeights( stencils, A, "1",  0 )
    Wz  = phs2.getWeights( stencils, A, "2",  0 )
    Whv = phs2.getWeights( stencils, A, "hv", K )
    
    ib = np.transpose( np.tile( np.arange(len(xVec[ind.m])), (stencilSize,1) ) )
    ib = ib.flatten()
    jb = stencils.idx
    jb = jb.flatten()
    Wx  = sparse.coo_matrix( (Wx.flatten(), (ib,jb)) \
    , shape = ( len(xVec[ind.m]), len(xVec) ) )
    Wz  = sparse.coo_matrix( (Wz.flatten(), (ib,jb)) \
    , shape = ( len(xVec[ind.m]), len(xVec) ) )
    Whv = sparse.coo_matrix( (Whv.flatten(),(ib,jb)) \
    , shape = ( len(xVec[ind.m]), len(xVec) ) )
    
    def Dx( U ) :
        return nonhydro.Lphs( U, Wx, nLev, nCol, FD, ind.m, ii, jj )
    
    def Dz( U ) :
        return nonhydro.Lphs( U, Wz, nLev, nCol, FD, ind.m, ii, jj )
    
    def Dhv( U ) :
        U = nonhydro.Lphs( U, Whv, nLev, nCol, FD, ind.m, ii, jj )
        return dx**(2*K-1) * U
    
else :
    
    sys.exit( "\nError: rbfDerivatives should be 0 or 1.\n" )

###########################################################################

#Important functions for time stepping, which may be chosen by user:

if formulation == "exner" :
    
    def setGhostNodes( U ) :
        U = nonhydro.setGhostNodes1( U \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, jj \
        , nLev, nCol, thetaBar, g(), Cp() \
        , normGradS, ds, dsdxBottom, dsdzBottom \
        , wx, j0, j1, dx, FD, FDo2 )
        P = []
        return U, P
    
    if rbfDerivatives == 0 :
        
        def odefun( t, U ) :
            return nonhydro.odefun1( t, U \
            , setGhostNodes, Dx, Ds, HVx, HVs, [], [] \
            , ii, jj, i0, i1, j0, j1 \
            , dsdxEul, dsdzEul, rbfDerivatives \
            , Cp(), Cv(), Rd(), g(), gamma \
            , bigNull )
        
    elif rbfDerivatives == 1 :
        
        def odefun( t, U ) :
            return nonhydro.odefun1( t, U \
            , setGhostNodes, Dx, [], [], [], Dhv, Dz \
            , ii, jj, i0, i1, j0, j1 \
            , dsdxEul, dsdzEul, rbfDerivatives \
            , Cp(), Cv(), Rd(), g(), gamma \
            , bigNull )
        
    else :
        
        sys.exit( "\nError: rbfDerivatives should be 0 or 1.\n" )
    
elif formulation == "hydrostaticPressure" :
    
    def setGhostNodes( U ) :
        U, P = nonhydro.setGhostNodes2( U \
        , Tx, Tz, Nx, Nz, bigTx, bigTz, jj \
        , nLev, nCol, thetaBar, dpidsBar, g(), Cp(), Po(), Rd(), Cv() \
        , normGradS, ds, dsdxBottom, dsdzBottom, dsdz(x,z) \
        , wx, j0, j1, dx, FD, FDo2 )
        return U, P
    
    def odefun( t, U ) :
        return nonhydro.odefun2( t, U \
        , setGhostNodes, Dx, Dx2D, Ds, Ds2D, HVx, HVs \
        , ii, jj, i0, i1, j0, j1 \
        , dsdxEul, dsdzEul, dsdxAll, dsdzAll \
        , Cp(), Cv(), Rd(), g(), gamma )
    
else :
    
    sys.exit( "\nError: formulation should be 'exner' or 'hydrostaticPressure'.\n" )

###########################################################################

#This is not working well yet.  Need to get semi-implicit working first.

if semiLagrangian == 1 :
    
    def semiLagrangianTimestep( Un1, U, alp, bet ) :
        U1, alp, bet = nonhydro.conventionalSemiLagrangianTimestep( Un1, U, alp, bet \
        , setGhostNodes, Dx2D, Ds2D \
        , nLev, nCol, FD, FDo2, ds \
        , Cp(), Rd(), Cv(), g(), dt \
        , x.flatten(), z.flatten(), dsdx, dsdz \
        , ind.m, i0, i1, j0, j1 \
        , rbfOrder, polyOrder, stencilSize )
        return U1, alp, bet
        # return nonhydro.mySemiLagrangianTimestep( Un1, U \
        # , setGhostNodes, Dx, Ds \
        # , x.flatten(), z.flatten(), ind.m, dt \
        # , nLev, nCol, FD, i0, i1, j0, j1 \
        # , Cp(), Rd(), Cv(), g(), dsdxVec[ind.m], dsdzVec[ind.m] \
        # , rbfOrder, polyOrder, stencilSize )

###########################################################################

#Functions that will not be changed by user:

if rkStages == 3 :
    rk = rk.rk3
elif rkStages == 4 :
    rk = rk.rk4
else :
    sys.exit( "\nError: rkStages should be 3 or 4.  rk2 is not stable for this problem.\n" )

def printInfo( U, et, t ) :
    return nonhydro.printInfo( U, et , t \
    , formulation \
    , thetaBar, piBar, dpidsBar )

#Figure size and contour levels for plotting:
fig, CL = nonhydro.setFigAndContourLevels( testCase )

def saveContourPlot( U, t ) :
    nonhydro.saveContourPlot( U, t \
    , formulation, testCase, var, fig \
    , x, z, thetaBar, piBar, dpidsBar, CL, FDo2 \
    , xLeft, xRight, zTop, dx, ds )

###########################################################################

#Eulerian time-stepping for first large time step:

print()
print("dt =",dt)
print("dtEul =",dtEul)
print()

#Save initial conditions and contour of first frame:
U, P = setGhostNodes( U )
if saveArrays == 1 :
    np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
Un1 = U
et = printInfo( U, time.clock(), t )
if saveContours == 1 :
    saveContourPlot( U, t )

#The actual Eulerian time-stepping from t=0 to t=dt:
for i in range( np.int( np.round(dt/dtEul) + 1e-12 ) ) :
    U = rk( t, U, odefun, dtEul )
    t = t + dtEul

U, P = setGhostNodes( U )
alp = 0.
bet = 0.

###########################################################################

#The rest of the time-stepping:

for i in range(1,nTimesteps+1) :
    
    if np.mod( i, np.int(np.round(saveDel/dt)) ) == 0 :
        
        if plotFromSaved == 0 :
            U, P = setGhostNodes( U )
            if saveArrays == 1 :
                np.save( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy', U )
        elif plotFromSaved == 1 :
            U = np.load( saveString+'{0:04d}'.format(np.int(np.round(t)))+'.npy' )
        else :
            sys.exit( "\nError: plotFromSaved should be 0 or 1.\n" )
        
        et = printInfo( U, et, t )
        if saveContours == 1 :
            saveContourPlot( U, t )
        
    if plotFromSaved == 0 :
        if semiLagrangian == 0 :
            U = rk( t, U, odefun, dt )
        elif semiLagrangian == 1 :
            U1, alp, bet = semiLagrangianTimestep( Un1, U, alp, bet )
            Un1 = U
            U = U1
        else :
            sys.exit( "\nError: semiLagrangian should be 0 or 1.\n" )
    
    t = t + dt

############################################################################
