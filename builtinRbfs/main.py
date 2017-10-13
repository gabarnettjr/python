#main.py

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import time
from scipy import spatial
from gab import phs

k = 10;
def trueFunction( x, y ) :
    # z = np.cos( np.pi * x ) * np.sin( np.pi * y )
    z = np.exp( -k * ( (x-.1)**2 + (y-np.pi/6)**2 ) )
    return z
def trueFunction_x( x, y ) :
    # z = -np.pi * np.sin( np.pi * x ) * np.sin( np.pi * y )
    z = -2*k*(x-.1) * trueFunction(x,y)
    return z
    
def trueFunction_y( x, y ) :
    # z = np.pi * np.cos( np.pi * x) * np.cos( np.pi * y )
    z = -2*k*(y-np.pi/6) * trueFunction(x,y)
    return z

useGlobalRbfs = 0
useLocalRbfs  = 1
approximateDerivatives = 1;
rbfParam = 3
polyorder = 1
stencilSize = 9
useFastMethod = 1

d = 1/1

n = 51
x = d * np.linspace( -1, 1, n )
y = d * np.linspace( -1, 1, n )

N = 51
X = d * np.linspace( -1, 1, N )
Y = d * np.linspace( -1, 1, N )

###########################################################################

x,y = np.meshgrid( x, y )
x = x.flatten()
y = y.flatten()
z = trueFunction( x, y )

###########################################################################

X,Y = np.meshgrid( X, Y )

start_time = time.clock()

if useGlobalRbfs == 1 :

    phi = interpolate.Rbf( x, y, z \
   # , function='thin_plate' \
   # , epsilon=1 \
   # , smooth=.00001 \
   # , norm = euclidean_norm \
    )
    Z = phi( X, Y )
    
elif useLocalRbfs == 1 :

    stencils = phs.getStencils( x, y, X.flatten(), Y.flatten(), stencilSize )

    # pts = np.transpose( np.vstack((x,y)) )
    # tree = spatial.cKDTree(pts)
    # X = X.flatten()
    # Y = Y.flatten()
    
    mn = phs.getMN()
    numPoly = np.int( ( polyorder + 1 ) * ( polyorder + 2 ) / 2 )
    
    # PTS = np.transpose( np.vstack((X,Y)) )
    # idx = tree.query( PTS, stencilSize )
    # rad = idx[0]
    # rad = rad[:,stencilSize-1]
    # idx = idx[1]
    
    if useFastMethod == 1 :
    
        Xn = x[idx] - np.transpose( np.tile( X, (stencilSize,1) ) )
        Yn = y[idx] - np.transpose( np.tile( Y, (stencilSize,1) ) )
        A = np.zeros(( len(X), stencilSize+numPoly, stencilSize+numPoly ))
        
        #the single-for-loop way (but you have to tile things):
        mn = mn[ 0:numPoly, : ]
        mn0 = np.tile( mn[:,0], (len(X),1) )
        mn1 = np.tile( mn[:,1], (len(X),1) )
        rad = np.transpose( np.tile( stencils.rad, (stencilSize,1) ) )
        for i in range(stencilSize) :
            tmpX = np.transpose( np.tile( Xn[:,i], (stencilSize,1) ) )
            tmpY = np.transpose( np.tile( Yn[:,i], (stencilSize,1) ) )
            A[ :, i, 0:stencilSize ] = phs.phi( rad, tmpX-Xn, tmpY-Yn, rbfParam )
            tmp = tmpX[:,0:numPoly]**mn0 * tmpY[:,0:numPoly]**mn1 / rad[:,0:numPoly]**(mn0+mn1)
            A[ :, i, stencilSize:stencilSize+numPoly ] = tmp
            A[ :, stencilSize:stencilSize+numPoly, i ] = tmp
        
        # #The double-for-loop way (but not looping over very much (seems a little faster)):
        # for i in range(stencilSize) :
            # for j in range(numPoly) :
                # A[:,i,j] = phs.phi( rad, Xn[:,i]-Xn[:,j], Yn[:,i]-Yn[:,j], rbfParam )
                # tmp = Xn[:,i]**mn[j,0] * Yn[:,i]**mn[j,1] / rad**(mn[j,0]+mn[j,1])
                # A[ :, i, stencilSize+j ] = tmp
                # A[ :, stencilSize+j, i ] = tmp
            # for j in range(numPoly,stencilSize) :
                # A[:,i,j] = phs.phi( rad, Xn[:,i]-Xn[:,j], Yn[:,i]-Yn[:,j], rbfParam )
        # rad = np.transpose( np.tile( rad, (stencilSize,1) ) )
        
        f = np.zeros(( len(X), stencilSize+numPoly, 1 ))
        f[ :, 0:stencilSize, 0 ] = z[idx]
        lam = np.linalg.solve( A, f )
        lam = np.reshape( lam, (len(X),stencilSize+numPoly) )
        #for interpolation:
        b = np.zeros(( len(X), stencilSize+numPoly ))
        b[:,stencilSize] = 1
        b[:,0:stencilSize] = phs.phi( rad, 0-Xn, 0-Yn, rbfParam )
        Z = np.sum( b*lam, axis=1 )
        Z = np.reshape( Z, (N,N) )
        if approximateDerivatives == 1 :
            #for first derivative in x:
            b_x = np.zeros(( len(X), stencilSize+numPoly ))
            b_x[:,stencilSize+1] = 1/rad[:,0]
            b_x[:,0:stencilSize] = phs.phi_x( rad, 0-Xn, 0-Yn, rbfParam )
            #for first derivative in y:
            b_y = np.zeros(( len(X), stencilSize+numPoly ))
            b_y[:,stencilSize+2] = 1/rad[:,0]
            b_y[:,0:stencilSize] = phs.phi_y( rad, 0-Xn, 0-Yn, rbfParam )
            #for Laplacian:
            bL = np.zeros(( len(X), stencilSize+numPoly ))
            if numPoly > 3 :
                bL[:,stencilSize+3] = 2/rad[:,0]**2
                bL[:,stencilSize+5] = 2/rad[:,0]**2
            bL[:,0:stencilSize] = phs.phiHV( rad, 0-Xn, 0-Yn, rbfParam, 1 )
            #get derivative approximations:
            Z_x = np.sum( b_x*lam, axis=1 )
            Z_x = np.reshape( Z_x, (N,N) )
            Z_y = np.sum( b_y*lam, axis=1 )
            Z_y = np.reshape( Z_y, (N,N) )
            ZL = np.sum( bL*lam, axis=1 )
            ZL = np.reshape( ZL, (N,N) )
        X = np.reshape( X, (N,N) )
        Y = np.reshape( Y, (N,N) )
        
    else :
        
        Xn = x[idx]
        Yn = y[idx]
        Zn = z[idx]
        A = np.zeros( ( stencilSize+numPoly, stencilSize+numPoly ) )
        f = np.zeros( stencilSize+numPoly )
        b = np.zeros( stencilSize+numPoly )
        b[stencilSize] = 1
        Z = np.zeros( np.shape(X) )
        for i in range( len(X) ) :
            xn = Xn[i,:] - X[i]
            yn = Yn[i,:] - Y[i]
            xx,xx = np.meshgrid( xn, xn )
            yy,yy = np.meshgrid( yn, yn )
            A[0:stencilSize,0:stencilSize] = phs.phi( rad[i], np.transpose(xx)-xx \
            , np.transpose(yy)-yy, rbfParam )
            for j in range(numPoly) :
                tmp = xn**mn[j,0] * yn**mn[j,1] / rad[i]**(mn[j,0]+mn[j,1])
                A[ 0:stencilSize, stencilSize+j ] = tmp
                A[ stencilSize+j, 0:stencilSize ] = tmp
            f[0:stencilSize] = Zn[i,:]
            lam = np.linalg.solve( A, f )
            b[0:stencilSize] = phs.phi(rad[i],0-xn,0-yn,rbfParam)
            Z[i] = np.dot( b, lam )
        X = np.reshape( X, (N,N) )
        Y = np.reshape( Y, (N,N) )
        Z = np.reshape( Z, (N,N) )

else :

    pts = np.transpose( np.vstack((x,y)) )
    Z = interpolate.griddata( pts, z, (X,Y) \
        , method='cubic' \
        , fill_value=np.nan \
        , rescale=False \
        )

print( time.clock() - start_time, "seconds" )

###########################################################################

# #THERE IS A BUG IN THIS BUILT-IN IMPLEMENTATION:
# start_time = time.clock();
# phi = interpolate.interp2d( x, y, z \
    # , kind='cubic' \
    # # , copy=True \
    # # , bounds_error=False \
    # # , fill_value=np.nan \
    # )
# Z = phi( X, Y );
# print( time.clock() - start_time, "seconds" )
# X,Y = np.meshgrid( X, Y )

###########################################################################

#plot the approximation and the error:

Zexact = trueFunction( X, Y )
print( np.max( np.max( np.abs( Z - Zexact ) ) ) )
if approximateDerivatives == 1 :
    Zexact_x = trueFunction_x( X, Y )
    print( np.max( np.max( np.abs( Z_x - Zexact_x ) ) ) )
    Zexact_y = trueFunction_y( X, Y )
    print( np.max( np.max( np.abs( Z_y - Zexact_y ) ) ) )

# contourVector = np.linspace( -1.1, 1.1, 23 )
plt.figure(1)
plt.contourf( X, Y, Z )
# plt.contourf( X, Y, (Zexact-Z)/np.max(np.max(np.abs(Zexact))) )
plt.colorbar()
plt.axis( 'equal' )

if approximateDerivatives == 1 :

    plt.figure(2)
    plt.contourf( X, Y, Z_x )
    # plt.contourf( X, Y, (Zexact_x-Z_x)/np.max(np.max(np.abs(Zexact_x))) )
    plt.colorbar()
    plt.axis( 'equal' )

    plt.figure(3)
    plt.contourf( X, Y, ZL )
    # plt.contourf( X, Y, (Zexact_y-Z_y)/np.max(np.max(np.abs(Zexact_y))) )
    plt.colorbar()
    plt.axis( 'equal' )

plt.show()
