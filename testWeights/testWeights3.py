import numpy as np
from gab import phs3
import matplotlib.pyplot as plt
import time

kap = 6;
n = 21
N = n-kap;
rbfParam = 3
polyorder = 1
stencilSize = 30
e1 = [ np.sqrt(3/3), np.sqrt(0/3), np.sqrt(0/3) ]
e2 = [ np.sqrt(0/3), np.sqrt(3/3), np.sqrt(0/3) ]
e3 = [ np.sqrt(0/3), np.sqrt(0/3), np.sqrt(3/3) ]
op = "hv"
K = 1

###########################################################################

alp = 4
a = np.exp(-1)
b = np.pi/6
c = .1

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
x,y,z = np.meshgrid( x, x, x )
x = x.flatten()
y = y.flatten()
z = z.flatten()
v = trueFunction(x,y,z)

X = np.linspace( -1+kap/2*h, 1-kap/2*h, N )
X,Y,Z = np.meshgrid( X, X, X )
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

print( time.clock() - start_time, "seconds" )

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

# plt.figure(1)
# if plotError == 1 :
    # plt.contourf( X, Y, (Zexact-Z)/np.max(np.max(np.abs(Zexact))) )
# else :
    # plt.contourf( X, Y, Z )
# plt.colorbar()
# plt.axis( 'equal' )
# plt.show()

print( np.max( np.abs( V - Vexact ) ) )

###########################################################################