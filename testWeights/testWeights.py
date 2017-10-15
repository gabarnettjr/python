import numpy as np
from gab import phs2
import matplotlib.pyplot as plt
import time

n = 51
N = 179
rbfParam = 3
polyorder = 2
stencilSize = 9
op = "hv"
K = 1
plotError = 1;

###########################################################################

k = 10
a = np.exp(-1)
b = np.pi/6

def trueFunction( x, y ) :
    z = np.exp( -k * ( (x-a)**2 + (y-b)**2 ) )
    # z = np.cos( np.pi * x ) * np.sin( np.pi * y )
    return z
    
def trueFunction_x( x, y ) :
    z = -2*k*(x-a) * trueFunction(x,y)
    # z = -np.pi * np.sin( np.pi * x ) * np.sin( np.pi * y )
    return z
    
def trueFunction_y( x, y ) :
    z = -2*k*(y-b) * trueFunction(x,y)
    # z = np.pi * np.cos( np.pi * x) * np.cos( np.pi * y )
    return z
    
def trueFunctionL( x, y ) :
    z = 4*k * ( k*a**2 - 2*k*a*x + k*b**2 - 2*k*b*y + k*x**2 + k*y**2 - 1 ) * trueFunction(x,y)
    # z = - 2*np.pi**2 * np.cos( np.pi * x ) * np.sin( np.pi * y )
    return z
    
def trueFunctionL2( x, y ) :
    z = 16*k**2*(k*(a - x)**2*(k*(a - x)**2 + k*(b - y)**2 - 1) - 3*k*(a - x)**2 \
    + k*(b - y)**2*(k*(a - x)**2 + k*(b - y)**2 - 1) - 3*k*(b - y)**2 + 2)*trueFunction(x,y)
    return z
    
###########################################################################

x = np.linspace( -1, 1, n )
h = 2/(n-1)
x, y = np.meshgrid( x, x )
x = x.flatten()
y = y.flatten()
z = trueFunction(x,y)

X = np.linspace( -1+1*h, 1-1*h, N )
X, Y = np.meshgrid( X, X )
X = X.flatten()
Y = Y.flatten()

start_time = time.clock()

stencils = phs2.getStencils( x, y, X, Y, stencilSize )
A = phs2.getAmatrices( stencils, rbfParam, polyorder )
W = phs2.getWeights( stencils, A, op, K )
Z = np.sum( W*z[stencils.idx], axis=1 )

print( time.clock() - start_time, "seconds" )

###########################################################################

X = np.reshape( X, (N,N) )
Y = np.reshape( Y, (N,N) )
Z = np.reshape( Z, (N,N) )

if op == "i" :
    Zexact = trueFunction( X, Y )
elif op == "x" :
    Zexact = trueFunction_x( X, Y )
elif op == "y" :
    Zexact = trueFunction_y( X, Y )
elif op == "hv" :
    if K == 1 :
        Zexact = trueFunctionL( X, Y )
    elif K == 2 :
        Zexact = trueFunctionL2( X, Y )

plt.figure(1)
if plotError == 1 :
    plt.contourf( X, Y, (Zexact-Z)/np.max(np.max(np.abs(Zexact))) )
else :
    plt.contourf( X, Y, Z )
plt.colorbar()
plt.axis( 'equal' )
plt.show()

###########################################################################