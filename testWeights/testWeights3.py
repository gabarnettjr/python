import numpy as np
from gab import phs3
import matplotlib.pyplot as plt
import time

ng = 1;             #number of ghost node layers
n = 21
N = n-2*ng;
rbfParam = 7
polyorder = 3
stencilSize = 60
e1 = [ np.sqrt(3/3), np.sqrt(0/3), np.sqrt(0/3) ]
e2 = [ np.sqrt(0/3), np.sqrt(3/3), np.sqrt(0/3) ]
e3 = [ np.sqrt(0/3), np.sqrt(0/3), np.sqrt(3/3) ]
op = "3"

K = 1
plotError = 1;

###########################################################################

alp = 4
a = np.exp(-1)
b = np.pi/6
c = .123

def trueFunction( x, y, z ) :
    return np.exp( -alp * ( (x-a)**2 + (y-b)**2 + (z-c)**2 ) )
    
def trueFunction_x( x, y, z ) :
    return -2*alp*(x-a) * trueFunction(x,y,z)
    
def trueFunction_y( x, y, z ) :
    return -2*alp*(y-b) * trueFunction(x,y,z)

def trueFunction_z( x, y, z ) :
    return -2*alp*(z-c) * trueFunction(x,y,z)
    
def trueFunction_1( x, y, z ) :
    return e1[0,0] * trueFunction_x(x,y,z) + e1[0,1]*trueFunction_y(x,y,z) + e1[0,2]*trueFunction_z(x,y,z)

def trueFunction_2( x, y, z ) :
    return e2[0,0] * trueFunction_x(x,y,z) + e2[0,1]*trueFunction_y(x,y,z) + e2[0,2]*trueFunction_z(x,y,z)

def trueFunction_3( x, y, z ) :
    return e3[0,0] * trueFunction_x(x,y,z) + e3[0,1]*trueFunction_y(x,y,z) + e3[0,2]*trueFunction_z(x,y,z)
    
def trueFunctionL( x, y, z ) :
    return -2*alp * (  (x-a)*trueFunction_x(x,y,z) + (y-b)*trueFunction_y(x,y,z) + (z-c)*trueFunction_z(x,y,z) + 3*trueFunction(x,y,z) )
    
###########################################################################

x = np.linspace( -1, 1, n )
h = 2/(n-1)
y,z,x = np.meshgrid( x, x, x )
x = x.flatten()
y = y.flatten()
z = z.flatten()
v = trueFunction(x,y,z)

X = np.linspace( -1+ng*h, 1-ng*h, N )
Y,Z,X = np.meshgrid( X, X, X )
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

start_time = time.clock()

stencils = phs3.getStencils( x, y, z, X, Y, Z, stencilSize )
e1 = np.tile( e1, (stencils.nE,1) )
e2 = np.tile( e2, (stencils.nE,1) )
e3 = np.tile( e3, (stencils.nE,1) )
stencils = phs3.rotateStencils( stencils, e1, e2, e3 )
A = phs3.getAmatrices( stencils, rbfParam, polyorder )
W = phs3.getWeights( stencils, A, op, K )
V = np.sum( W*v[stencils.idx], axis=1 )

print()
print( time.clock() - start_time, "seconds" )
print()
print( np.max( A.cond ) )
print()

###########################################################################

X = np.reshape( X, (N,N,N) )
Y = np.reshape( Y, (N,N,N) )
Z = np.reshape( Z, (N,N,N) )
V = np.reshape( V, (N,N,N) )

if op == "i" :
    Vexact = trueFunction( X, Y, Z )
elif op == "1" :
    Vexact = trueFunction_1( X, Y, Z )
elif op == "2" :
    Vexact = trueFunction_2( X, Y, Z )
elif op == "3" :
    Vexact = trueFunction_3( X, Y, Z )
elif op == "hv" :
    Vexact = trueFunctionL( X, Y, Z )
    
nContours = 16
if plotError == 1 :
    tmp = (Vexact-V) / np.max(np.abs(Vexact))
    cv = np.linspace( np.min(tmp), np.max(tmp), nContours )
else :
    cv = np.linspace( np.min(V), np.max(V), nContours )

plt.figure( figsize=(12,9) )
for i in range(N) :
    x = np.reshape( X[i,:,:], (N,N) )
    y = np.reshape( Y[i,:,:], (N,N) )
    v = np.reshape( V[i,:,:], (N,N) )
    ve = np.reshape( Vexact[i,:,:], (N,N) )
    if plotError == 1 :
        tmp = (ve-v) / np.max(np.abs(Vexact))
        plt.contourf( x, y, tmp, cv )
        plt.title( 'z = ' + str(Z[i,0,0]) + ', min = ' + str(np.min(tmp)) + ', max = ' + str(np.max(tmp)) )
    else :
        plt.contourf( x, y, v, cv )
        plt.title( 'z = ' + str(Z[i,0,0]) + ', min = ' + str(np.min(v)) + ', max = ' + str(np.max(v)) )
    plt.colorbar()
    plt.axis( 'equal' )
    plt.waitforbuttonpress()
    plt.clf()

print()
print( np.max(np.abs(Vexact-V)) / np.max(np.abs(Vexact)) )
print()

###########################################################################