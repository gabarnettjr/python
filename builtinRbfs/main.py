import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import time
from scipy import spatial
import rbffd

def trueFunction( x, y ) :
    z = np.cos( np.pi * x ) * np.sin( np.pi * y )
    return z

useGlobalRbfs = 0;
useLocalRbfs  = 0;
rbfParam = 5;
polyorder = 2;
stencilSize = 16;

d = 2/1

n = 23
x = d * np.linspace( -1, 1, n )
y = d * np.linspace( -1, 1, n )

N = 101
X = d * np.linspace( -1, 1, N )
Y = d * np.linspace( -1, 1, N )

###########################################################################

x,y = np.meshgrid( x, y )
x = x.flatten()
y = y.flatten()
z = trueFunction( x, y )

###########################################################################

X,Y = np.meshgrid( X, Y )

if useGlobalRbfs == 1 :
    start_time = time.clock()
    phi = interpolate.Rbf( x, y, z \
#    , function='thin_plate' \
#    , epsilon=1 \
#    , smooth=.00001 \
#    , norm = euclidean_norm \
    )
    Z = phi( X, Y )
    print( time.clock() - start_time, "seconds" )
elif useLocalRbfs == 1 :
    start_time = time.clock()
    mn = rbffd.getMN()
    numPoly = np.int( ( polyorder + 1 ) * ( polyorder + 2 ) / 2 );
    pts = np.transpose( np.vstack((x,y)) )
    tree = spatial.cKDTree(pts)
    X = X.flatten()
    Y = Y.flatten()
    PTS = np.transpose( np.vstack((X,Y)) )
    idx = tree.query( PTS, stencilSize )
    print( time.clock() - start_time, "seconds" )
    rad = idx[0]
    rad = rad[:,stencilSize-1]
    idx = idx[1]
    #the fast way:
    Xn = np.zeros(( len(X), 1, stencilSize ))
    Xn[:,0,:] = x[idx] - np.transpose( np.tile( X, (stencilSize,1) ) )
    Yn = np.zeros(( len(Y), 1, stencilSize ))
    Yn[:,0,:] = y[idx] - np.transpose( np.tile( Y, (stencilSize,1) ) )
    f = np.zeros(( len(X), stencilSize+numPoly, 1 ))
    f[ :, 0:stencilSize, 0 ] = z[idx]
    b = np.zeros( np.shape(f) )
    b[:,stencilSize,0] = 1;
    A = np.zeros(( len(X), stencilSize+numPoly, stencilSize+numPoly ));
    for i in range(stencilSize) :
        for j in range(stencilSize) :
            A[:,i,j] = rbffd.phi( rad, Xn[:,0,i]-Xn[:,0,j], Yn[:,0,i]-Yn[:,0,j], rbfParam )
            if j < numPoly :
                tmp = Xn[:,0,i]**mn[j,0] * Yn[:,0,i]**mn[j,1] / rad**(mn[j,0]+mn[j,1])
                A[ :, i, stencilSize+j ] = tmp
                A[ :, stencilSize+j, i ] = tmp
        b[:,i,0] = rbffd.phi( rad, 0-Xn[:,0,i], 0-Yn[:,0,i], rbfParam )
    lam = np.linalg.solve( A, f )
    Z = np.sum( b*lam, axis=1 )
    X = np.reshape( X, (N,N) )
    Y = np.reshape( Y, (N,N) )
    Z = np.reshape( Z, (N,N) )
    print( time.clock() - start_time, "seconds" )
#    #the slow way:
#    Xn = x[idx]
#    Yn = y[idx]
#    Zn = z[idx];
#    A = np.zeros( ( stencilSize+numPoly, stencilSize+numPoly ) )
#    f = np.zeros( stencilSize+numPoly )
#    b = np.zeros( stencilSize+numPoly )
#    b[stencilSize] = 1;
#    X = X.flatten();
#    Y = Y.flatten();
#    Z = np.zeros( np.shape(X) )
#    for i in range( len(X) ) :
#        xn = Xn[i,:] - X[i]
#        yn = Yn[i,:] - Y[i]
#        xx,xx = np.meshgrid( xn, xn )
#        yy,yy = np.meshgrid( yn, yn )
#        A[0:stencilSize,0:stencilSize] = \
#            rbffd.phi( rad[i], np.transpose(xx)-xx \
#            , np.transpose(yy)-yy, rbfParam )
#        for j in range(numPoly) :
#            tmp = xn**mn[j,0] * yn**mn[j,1] / rad[i]**(mn[j,0]+mn[j,1])
#            A[ 0:stencilSize, stencilSize+j ] = tmp
#            A[ stencilSize+j, 0:stencilSize ] = tmp
#        f[0:stencilSize] = Zn[i,:]
#        lam = np.linalg.solve( A, f )
#        b[0:stencilSize] = rbffd.phi(rad[i],0-xn,0-yn,rbfParam)
#        Z[i] = np.dot( b, lam )
#    X = np.reshape( X, (N,N) )
#    Y = np.reshape( Y, (N,N) )
#    Z = np.reshape( Z, (N,N) )
#    print( time.clock() - start_time, "seconds" )
else :
    start_time = time.clock()
    pts = np.transpose( np.vstack((x,y)) )
    Z = interpolate.griddata( pts, z, (X,Y) \
        , method='cubic' \
        , fill_value=np.nan \
        , rescale=False \
        );
    print( time.clock() - start_time, "seconds" )


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

# contourVector = np.linspace( -1.2, 1.2, 13 )

#plt.figure(1)
#plt.contourf( X, Y, Zexact, contourVector )
#plt.colorbar()

# plt.figure(2)
# plt.contourf( X, Y, Z, contourVector )
# plt.colorbar()

# plt.figure(3)
# plt.contourf( X, Y, Z-Zexact )
# plt.colorbar()
# #plt.plot( x, y, 'k.' )

# plt.show()
