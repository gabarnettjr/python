import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

def trueFunction( x, y ) :
    z = np.cos( np.pi * x ) * np.sin( np.pi * y )
    return z

d = 1/1;

n = 12
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

phi = interpolate.Rbf( x, y, z \
    # , function='cubic' \
    # , epsilon=1 \
    , smooth=.00001 \
    # , norm = euclidean_norm \
    )
X,Y = np.meshgrid( X, Y );
Z = phi( X, Y );

# pts = np.transpose( np.vstack((x,y)) )
# X,Y = np.meshgrid( X, Y )
# Z = interpolate.griddata( pts, z, (X,Y) \
    # , method='linear' \
    # , fill_value=np.nan \
    # , rescale=False \
    # );

# phi = interpolate.interp2d( x, y, z \
    # , kind='cubic' \
    # # , copy=True \
    # # , bounds_error=False \
    # # , fill_value=np.nan \
    # )
# Z = phi( X, Y );
# X,Y = np.meshgrid( X, Y )

###########################################################################

Zexact = trueFunction( X, Y )

plt.figure(1)
plt.contourf( X, Y, Zexact, np.linspace(-1.2,1.2,9) )
plt.colorbar()

plt.figure(2)
plt.contourf( X, Y, Z, np.linspace(-1.2,1.2,9) )
plt.colorbar()

plt.figure(3)
plt.contourf( X, Y, Z-Zexact )
plt.colorbar()
plt.plot( x, y, '.' )

plt.show()