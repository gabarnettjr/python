import os

for i in range( 5 ) :
    
    ns = '{0:1d}'.format( 12*2**i )
    dtinv = '{0:1d}'.format( 1*2**i )
    
    #################################
    
    #p5:
    
    # str = "2 7 5 13 .30 3 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    # str = "2 7 5 13 .30 4 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    # str = "2 7 5 13 .00 3 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    # str = "2 7 5 13 .00 4 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    #################################
    
    #p3:
    
    # str = "2 5 3 7 .30 3 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    # str = "2 5 3 7 .30 4 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    str = "2 5 3 7 .00 3 " + ns + " " + dtinv
    print( str + " :" )
    os.system( "python main.py " + str )
    print()
    
    # str = "2 5 3 7 .00 4 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()
    
    #################################
    
    #FD4:
    
    # str = "2 5 4 5 .00 4 " + ns + " " + dtinv
    # print( str + " :" )
    # os.system( "python main.py " + str )
    # print()

#####################################

#reference solutions:

i = 5
ns = '{0:1d}'.format( 12*2**i )
dtinv = '{0:1d}'.format( 1*2**i )

str = "2 5 3 7 .00 4 " + ns + " " + dtinv
print( str + " :" )
os.system( "python main.py " + str )
print()

# str = "2 9 7 21 .00 4 " + ns + " " + dtinv
# print( str + " :" )
# os.system( "python main.py " + str )
# print()

# str = "2 7 5 13 .00 4 " + ns + " " + dtinv
# print( str + " :" )
# os.system( "python main.py " + str )
# print()

###########################################################################