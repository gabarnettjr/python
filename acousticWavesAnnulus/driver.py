import os

#dimensions entirely split:

# for i in range( 5 ) :
    # ns = '{0:1d}'.format( 2**(i+3) )
    # dtinv = '{0:1d}'.format( 2**(i+1) )
    # str = "2 5 3 9 .00 3 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()

#dimensions split except for th derivative:

for i in range( 4 ) :
    ns = '{0:1d}'.format( 2**(i+3) )
    dtinv = '{0:1d}'.format( 2**(i+1) )
    str = "1 5 3 25 .30 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()