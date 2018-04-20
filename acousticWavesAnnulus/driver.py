import os

for i in range( 6 ) :
    
    ns = '{0:3d}'.format( 2**(i+3) )
    
    dt = '{0:1.6f}'.format( .5 / 2**i )
    
    str1 = "4 5 4 5 .00 " + ns + " " + dt
    str2 = "4 7 6 7 .00 " + ns + " " + dt
    
    print( str1 + " :" )
    os.system( "python main.py " + str1 )
    print()
    
    print( str2 + " :" )
    os.system( "python main.py " + str2 )
    print()