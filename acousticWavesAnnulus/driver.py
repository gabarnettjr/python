import os

#p5 regular nodes:
for i in range( 5 ) :
    ns = '{0:1d}'.format( 2**(i+4) )
    dtinv = '{0:1d}'.format( 2**(i+2) )
    str = "2 7 5 13 .00 4 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()

#p5 perturbed nodes:
for i in range( 4 ) :
    ns = '{0:1d}'.format( 2**(i+4) )
    dtinv = '{0:1d}'.format( 2**(i+2) )
    str = "2 7 5 13 .30 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()

#p3 regular nodes:
for i in range( 4 ) :
    ns = '{0:1d}'.format( 2**(i+4) )
    dtinv = '{0:1d}'.format( 2**(i+2) )
    str = "2 5 3 7 .00 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()

#p3 perturbed nodes:
for i in range( 4 ) :
    ns = '{0:1d}'.format( 2**(i+4) )
    dtinv = '{0:1d}'.format( 2**(i+2) )
    str = "2 5 3 7 .30 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()

# #p4 regular nodes (FD4) (diverges at high res):
# for i in range( 5 ) :
    # ns = '{0:1d}'.format( 2**(i+4) )
    # dtinv = '{0:1d}'.format( 2**(i+2) )
    # str = "2 5 4 5 .00 4 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()