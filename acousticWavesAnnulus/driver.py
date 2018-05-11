import os

for i in range( 5 ) :
    
    ns = '{0:1d}'.format( 12*2**i + 2 )
    dtinv = '{0:1d}'.format( 1*2**i )
    
    #################################
    
    #p5:
    
    str = "1 7 5 13 .30 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    str = "1 7 5 13 .30 4 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    str = "1 7 5 13 .00 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    str = "1 7 5 13 .00 4 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    #################################
    
    #p3:
    
    str = "1 5 3 7 .30 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    str = "1 5 3 7 .30 4 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    str = "1 5 3 7 .00 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    str = "1 5 3 7 .00 4 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    #################################
    
    #FD4:
    
    str = "1 5 4 5 .00 4 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()

#####################################

#reference solution:

i = 5
ns = '{0:1d}'.format( 12*2**i + 2 )
dtinv = '{0:1d}'.format( 1*2**i )

# str = "1 5 3 7 .00 4 " + ns + " " + dtinv
# print( str + " :" )
# os.system( "python main.py " + str )
# print()

str = "1 7 5 13 .00 4 " + ns + " " + dtinv
print( str + " :" )
os.system( "python main.py " + str )
print()

###########################################################################