import numpy as  np

###########################################################################

def getTopo( testCase ) :
    if testCase == "doubleStrakaTopo" :
        def topoFunc( x ) :
            y = 1000 * np.exp( -(16*(x-1000)/(b-a))**2 )
            return y
    elif testCase == "bubbleTopo" :
        def topoFunc( x ) :
            y = 500 * ( 1 + np.sin(2*np.pi*x/5000) )
            return y
    else :
        print( "error" )

###########################################################################

def getNodes( testCase, a, b, c, d, nx, nz ) :

    dx = (b-a) / (nx-1)
    dz = (d-c) / (nz-1)
    
    xx = np.linspace( a, b, nx )
    xx = np.transpose( np.tile( xx, (nz,1) ) )
    xxc = np.linspace( a+dx/2, b-dx/2, nx-1 )
    xxc = np.transpose( np.tile( xxc, (nz-1,1) ) )
    
    zz = np.zeros(( nx, nz ))
    for i in range(nx) :
        zz[i,:] = np.linspace(  )
    
    return xx, zz, xxc, zzc, dx, dz

###########################################################################