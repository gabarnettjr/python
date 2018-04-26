import os

#dimensions entirely split:
for i in range( 6 ) :
    ns = '{0:1d}'.format( 2**(i+3) )
    dtinv = '{0:1d}'.format( 2**(i) )
    str = "2 5 4 5 .00 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()

# #dimensions split except for theta derivative:
# for i in range( 5 ) :
    # ns = '{0:1d}'.format( 2**(i+3) )
    # dtinv = '{0:1d}'.format( 2**(i) )
    # str = "0 5 3 25 .00 3 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()

# #dimensions split except for theta derivative:
# for i in range( 5 ) :
    # ns = '{0:1d}'.format( 2**(i+3) )
    # dtinv = '{0:1d}'.format( 2**(i) )
    # str = "0 5 3 25 .30 3 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()